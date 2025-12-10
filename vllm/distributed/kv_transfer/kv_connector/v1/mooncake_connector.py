# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import httpx
import msgspec
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_global_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    get_ip,
    get_open_zmq_inproc_path,
    make_zmq_path,
    make_zmq_socket,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

EngineId = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"

logger = init_logger(__name__)

http_log_level = logger.getEffectiveLevel()
# INFO logs of http are too noisy. Silence them.
# Setting vllm log level to DEBUG if we really want to see.
if http_log_level == logging.INFO:
    http_log_level = logging.WARNING
logging.getLogger("httpx").setLevel(http_log_level)


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    req_blocks: dict[ReqId, list[int]]
    kv_caches_base_addr: list[int]


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    status: str
    err_reqs: list[ReqId] | None = None


@dataclass
class QueryReqMeta:
    local_block_ids: list[int]
    bootstrap_server_host: str
    bootstrap_server_port: int
    target_zmq_path: str | None = None


@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: threading.Event
    expire_time: float = float("inf")


@dataclass
class SendReqMeta:
    reqs: dict[ReqId, SendBlockMeta]
    lock: threading.Lock


@dataclass
class FinishedSendReqSet:
    set: set[ReqId]
    lock: threading.Lock


@dataclass
class FinishedReceiveReqSet:
    set: set[ReqId]
    lock: asyncio.Lock


class WorkerInfo(BaseModel):
    hostname: str
    port: int


class RegisterWorkerPayload(BaseModel):
    dp_rank: int
    tp_rank: int
    info: WorkerInfo


class RegisterRequestPayload(BaseModel):
    req_id: ReqId
    dp_rank: int


class QueryRequestsPayload(BaseModel):
    req_ids: list[ReqId]
    tp_rank: int
    tp_size: int


class QueryRequestsResponse(BaseModel):
    results: dict[ReqId, tuple[str, WorkerInfo | None]]


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, QueryReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = QueryReqMeta(
                local_block_ids=local_block_ids,
                bootstrap_server_host=kv_transfer_params["bootstrap_server_host"],
                bootstrap_server_port=kv_transfer_params["bootstrap_server_port"],
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.parallel_config
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        if self.kv_role != "kv_consumer":
            host, port = get_mooncake_bootstrap_addr(vllm_config)
            self.bootstrap_addr = make_zmq_path("http", host, port)

            self.sender_loop = asyncio.new_event_loop()
            self._sender_loop_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_loop_t.start()

        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.kv_role != "kv_consumer" and self.sender_loop.is_running():
            self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
            self._sender_loop_t.join()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert self.kv_role != "kv_producer"
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        if params.get("do_remote_decode"):
            assert self.kv_role != "kv_consumer"
            asyncio.run_coroutine_threadsafe(
                self.register_req_to_bootstrap(request.request_id), self.sender_loop
            )

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert self.kv_role != "kv_producer"
            if all(
                p in params for p in ("bootstrap_server_host", "bootstrap_server_port")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = (
                    blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            # Add an empty list to worker to create event.
            self._reqs_need_send[request.request_id] = []

    async def register_req_to_bootstrap(self, req_id: ReqId):
        url = self.bootstrap_addr + "/register_request"
        payload = RegisterRequestPayload(req_id=req_id, dp_rank=self.dp_rank)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload.model_dump())
                response.raise_for_status()
                logger.debug("Registered request %s to bootstrap server", req_id)
        except Exception as e:
            logger.error(
                "Failed to register request %s to bootstrap server: %s", req_id, e
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to RecvReqMeta.
        if self.kv_role != "kv_producer":
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if self.kv_role != "kv_consumer":
            for req_id, block_ids in self._reqs_need_send.items():
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params={},
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s",
            request.status,
            params,
        )
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert self.kv_role != "kv_producer"
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        assert self.kv_role != "kv_consumer"

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = block_ids

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", "rdma", "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0

        assert vllm_config.parallel_config
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_workers = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: SendReqMeta = SendReqMeta(reqs={}, lock=threading.Lock())

        # Only used by prefillers.
        host, port = get_mooncake_bootstrap_addr(vllm_config)
        self.bootstrap_addr = make_zmq_path("http", host, port)

        if self.kv_role != "kv_consumer":
            # Background thread for sending kvcaches to D.
            self._mooncake_sender_t: threading.Thread | None = None
            # Background thread for processing new sending requests.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers, thread_name_prefix="vllm-mooncake-sender"
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches", self.num_workers
            )

            # Start bootstrap server on global rank 0.
            if is_global_first_rank():
                self.bootstrap_server = MooncakeBootstrapServer(host, port)
                self.bootstrap_server.start()

        if self.kv_role != "kv_producer":
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            self._query_retry_list: dict[ReqId, QueryReqMeta] = {}
            self._query_retry_lock = asyncio.Lock()

        self.finished_sending_reqs: FinishedSendReqSet = FinishedSendReqSet(
            set(), threading.Lock()
        )
        self.finished_recving_reqs: FinishedReceiveReqSet = FinishedReceiveReqSet(
            set(), asyncio.Lock()
        )

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )
        self._use_pallas = self.kv_topo._use_pallas

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if self.kv_role != "kv_consumer":
            self._sender_executor.shutdown(wait=False)
            if self._mooncake_sender_t:
                self._mooncake_sender_t.join()
            if is_global_first_rank():
                self.bootstrap_server.shutdown()
        if self.kv_role != "kv_producer" and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    def register_worker_to_bootstrap(self):
        url = self.bootstrap_addr + "/register_worker"
        info = WorkerInfo(hostname=self.hostname, port=self.side_channel_port)
        payload = RegisterWorkerPayload(
            dp_rank=self.dp_rank, tp_rank=self.tp_rank, info=info
        )

        while True:
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                time.sleep(0.1)
            except Exception as e:
                raise RuntimeError("Could not connect to bootstrap server") from e

    def _mooncake_sender(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        frontend_path = make_zmq_path("tcp", self.hostname, 0)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.ROUTER)
        frontend_path = frontend.LAST_ENDPOINT.decode()
        self.side_channel_port = int(frontend_path.split(":")[-1])
        logger.debug("Mooncake sender starting listening on path: %s", frontend_path)

        backend_path = get_open_zmq_inproc_path()
        backend = make_zmq_socket(self.zmq_ctx, backend_path, zmq.PULL)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        self.register_worker_to_bootstrap()
        ready_event.set()

        try:
            while True:
                sockets = dict(poller.poll())

                if frontend in sockets:
                    identity, _, metadata_bytes = frontend.recv_multipart()
                    self._sender_executor.submit(
                        self._sender_worker,
                        identity,
                        metadata_bytes,
                        backend_path,
                    )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            frontend.close()
            backend.close()

    def _sender_worker(
        self, identity: bytes, metadata_bytes: bytes, worker_channel_path: str
    ):
        response = MooncakeXferResponse(status="unknown error")

        try:
            metadata = self._xfer_meta_decoder.decode(metadata_bytes)
            err_reqs = self.send_kv_to_decode(metadata)
            response.status = "ok"
            response.err_reqs = err_reqs
        except Exception as e:
            logger.error("Error processing Mooncake xfer request: %s", e)
            response.status = f"error: {e}"
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                msg = self._encoder.encode(response)
                pusher.send_multipart((identity, msg))
            except Exception as e:
                logger.warning(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def send_kv_to_decode(self, meta: MooncakeXferMetadata) -> list[ReqId]:
        send_reqs: list[tuple[ReqId, SendBlockMeta]] = []
        err_reqs: list[ReqId] = []
        wait_task: SendBlockMeta | None = None
        with self.reqs_need_send.lock:
            for req_id in meta.req_blocks:
                send_meta = self.reqs_need_send.reqs.get(req_id)
                if send_meta is None or not send_meta.ready.is_set():
                    wait_task = send_meta
                    err_reqs.append(req_id)
                    continue
                # Remove from reqs_need_send so that this req will not
                # be treated as expired.
                del self.reqs_need_send.reqs[req_id]
                send_reqs.append((req_id, send_meta))

        # Optimization: If we are the only waiter, try to wait for 50 seconds
        # which is less than decoder's RCVTIMEO (60s).
        if (
            len(meta.req_blocks) == 1
            and wait_task is not None
            and wait_task.ready.wait(50.0)
        ):
            with self.reqs_need_send.lock:
                del self.reqs_need_send.reqs[err_reqs[0]]
            send_reqs = [(err_reqs[0], wait_task)]
            err_reqs.clear()

        err_reqs += self._send_blocks(send_reqs, meta)

        with self.finished_sending_reqs.lock:
            self.finished_sending_reqs.set.update([req_id for req_id, _ in send_reqs])

        return err_reqs

    def _send_blocks(
        self,
        send_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> list[ReqId]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"
        err_reqs: list[ReqId] = []

        for req_id, send_meta in send_reqs:
            remote_block_ids = agent_meta.req_blocks[req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                err_reqs.append(req_id)
                continue
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                req_id,
                num_remote_blocks,
                remote_session,
            )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value != 0:
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

        logger.debug(
            "Sending to %s done, took %s",
            remote_session,
            time.perf_counter() - start_time,
        )

        return err_reqs

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

        # No need to launch server for D node.
        if self.kv_role == "kv_consumer":
            return

        ready_event = threading.Event()
        self._mooncake_sender_t = threading.Thread(
            target=self._mooncake_sender,
            args=(ready_event,),
            daemon=True,
            name="mooncake_sender",
        )
        self._mooncake_sender_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        async with self.finished_recving_reqs.lock:
            finished_recving_reqs = self.finished_recving_reqs.set
            self.finished_recving_reqs.set = set()
        return finished_recving_reqs

    def get_finished(
        self, finished_req_ids: set[ReqId]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        fut = None
        if self.kv_role != "kv_producer":
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            with self.finished_sending_reqs.lock:
                finished_sending_reqs = self.finished_sending_reqs.set
                self.finished_sending_reqs.set = set()
        else:
            finished_sending_reqs = set()

        finished_recving_reqs = fut.result() if fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        with self.reqs_need_send.lock:
            expired_reqs = [
                req_id
                for req_id, send_meta in self.reqs_need_send.reqs.items()
                if send_meta.expire_time < now
            ]
            for req_id in expired_reqs:
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                del self.reqs_need_send.reqs[req_id]
            if expired_reqs:
                finished_sending_reqs.update(expired_reqs)

        return finished_sending_reqs or None, finished_recving_reqs or None

    async def receive_kv(self, path: str, reqs_meta: dict[ReqId, QueryReqMeta]):
        req_ids = set(reqs_meta)
        retry_reqs: dict[ReqId, QueryReqMeta] = {}
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            req_blocks={
                req_id: meta.local_block_ids for req_id, meta in reqs_meta.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug("Sending kv transfer request for %s on path: %s", req_ids, path)

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
            ) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 60000)
                await sock.send(encoded_data)
                ret_msg = await sock.recv()
                response = self._xfer_resp_decoder.decode(ret_msg)
                if response.status != "ok" or response.err_reqs is None:
                    logger.error(
                        "Error happens during tranfering kvcache for %s: %s",  # noqa: E501
                        req_ids,
                        response.status,
                    )
                    return
                retry_reqs = {req_id: reqs_meta[req_id] for req_id in response.err_reqs}
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

        async with self.finished_recving_reqs.lock:
            self.finished_recving_reqs.set.update(req_ids)

        # Add retry_reqs to list.
        # These reqs will be received again in next start_load_kv().
        async with self._query_retry_lock:
            self._query_retry_list.update(retry_reqs)

        logger.debug("pulling kv_caches for %s finished", req_ids)

    def group_by_bootstrap_addr(
        self, reqs_to_recv: dict[ReqId, QueryReqMeta]
    ) -> dict[str, dict[ReqId, QueryReqMeta]]:
        """
        First-level grouping:
        group requests by their bootstrap server address.
        """
        grouped: dict[str, dict[ReqId, QueryReqMeta]] = defaultdict(dict)
        for req_id, meta in reqs_to_recv.items():
            bootstrap_addr = make_zmq_path(
                "http", meta.bootstrap_server_host, meta.bootstrap_server_port
            )
            grouped[bootstrap_addr][req_id] = meta
        return grouped

    def group_by_worker(
        self,
        meta_list: dict[ReqId, QueryReqMeta],
        bootstrap_resp: dict[ReqId, tuple[str, WorkerInfo | None]],
    ) -> tuple[dict[str, dict[ReqId, QueryReqMeta]], dict[ReqId, QueryReqMeta]]:
        """
        Second-level grouping:
        group requests by their final target prefiller worker's ZMQ path.
        """
        grouped: dict[str, dict[ReqId, QueryReqMeta]] = defaultdict(dict)
        retry_reqs = {}
        for req_id, meta in meta_list.items():
            if meta.target_zmq_path is not None:
                # Target zmq path is cached. Reuse it.
                grouped[meta.target_zmq_path][req_id] = meta
                continue
            response = bootstrap_resp.get(req_id)
            if not response:
                logger.warning(
                    "Bootstrap server internal error! "
                    "No address found for req %s in bootstrap response.",
                    req_id,
                )
                continue

            status, worker_info = response
            if status != "ok" or worker_info is None:
                logger.debug(
                    "Bootstrap query for req %s returned error: %s", req_id, status
                )
                retry_reqs[req_id] = meta
                continue

            path = make_zmq_path("tcp", worker_info.hostname, worker_info.port)
            # Cache target zmq path so that we will not send
            # duplicate query requests to bootstrap server.
            meta.target_zmq_path = path
            grouped[path][req_id] = meta

        return grouped, retry_reqs

    async def batch_query_requests(
        self, bootstrap_addr: str, req_ids: list[ReqId]
    ) -> dict[ReqId, tuple[str, WorkerInfo | None]]:
        url = bootstrap_addr + "/query_requests"
        payload = QueryRequestsPayload(
            req_ids=req_ids, tp_rank=self.tp_rank, tp_size=self.tp_size
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload.model_dump())
                response.raise_for_status()
                data = response.json()
                logger.debug("Received responses from bootstrap server: %s", data)
                response_data = QueryRequestsResponse.model_validate(data)
                return response_data.results
        except Exception as e:
            err_msg = (
                e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
            )
            logger.error(
                "Failed to query bootstrap server for %d requests: %s",
                len(req_ids),
                err_msg,
            )
            return {}

    async def handle_bootstrap_group(
        self, bootstrap_addr: str, meta_list: dict[ReqId, QueryReqMeta]
    ):
        # Only query reqs without cached zmq path.
        req_ids = [
            req_id for req_id, meta in meta_list.items() if meta.target_zmq_path is None
        ]
        if req_ids:
            bootstrap_resp = await self.batch_query_requests(bootstrap_addr, req_ids)
            if not bootstrap_resp:
                return
        else:
            bootstrap_resp = {}

        groups_by_worker, retry_reqs = self.group_by_worker(meta_list, bootstrap_resp)

        for worker_path, _meta_list in groups_by_worker.items():
            asyncio.create_task(self.receive_kv(worker_path, _meta_list))

        # Add retry_reqs to list.
        # These reqs will be queried again in next start_load_kv().
        async with self._query_retry_lock:
            self._query_retry_list.update(retry_reqs)

    async def do_load_kv(self, reqs_to_recv: dict[ReqId, QueryReqMeta]):
        # Merge last retry list
        async with self._query_retry_lock:
            reqs_to_recv.update(self._query_retry_list)
            self._query_retry_list.clear()

        if not reqs_to_recv:
            return

        groups_by_bootstrap = self.group_by_bootstrap_addr(reqs_to_recv)

        for bootstrap_addr, meta_list in groups_by_bootstrap.items():
            asyncio.create_task(self.handle_bootstrap_group(bootstrap_addr, meta_list))

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if self.kv_role != "kv_producer":
            asyncio.run_coroutine_threadsafe(
                self.do_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            with self.reqs_need_send.lock:
                for req_id, block_ids in metadata.reqs_to_send.items():
                    if block_ids:
                        # Already gone through request_finished()
                        send_meta = self.reqs_need_send.reqs[req_id]
                        send_meta.local_block_ids = block_ids
                        send_meta.ready.set()
                        send_meta.expire_time = (
                            time.perf_counter()
                            + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                        )
                    else:
                        # From update_state_after_alloc(),
                        # but not reach request_finished() yet
                        self.reqs_need_send.reqs[req_id] = SendBlockMeta(
                            local_block_ids=[], ready=threading.Event()
                        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# ####################################################################
# ## Mooncake Bootstrap Server
# ####################################################################


class MooncakeBootstrapServer:
    """
    A centralized server running on the global rank 0 prefiller worker.
    Its main purpose is to act as a service discovery mechanism.

    1. Prefiller workers register their connection info (IP, port, ranks) here.
    2. Prefiller workers register which requests they will be serving.
    3. Decoder workers query this server to find out which prefiller worker to
       contact for a specific request's KV cache.
    """

    def __init__(self, host: str, port: int):
        self._lock = threading.Lock()
        # store workers info: {dp_rank: {tp_rank: (hostname, port)}}
        self.workers: defaultdict[int, dict[int, WorkerInfo]] = defaultdict(dict)
        # store reqs info: {req_id: dp_rank}
        self.req_to_dp_rank: dict[ReqId, int] = {}
        self.tp_size = get_tensor_model_parallel_world_size()

        self.host = host
        self.port = port
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

    def _register_routes(self):
        self.app.post("/register_worker")(self._register_worker)
        self.app.post("/register_request")(self.register_request)
        self.app.post("/query_requests", response_model=QueryRequestsResponse)(
            self._query_requests
        )

    def start(self):
        if self.server_thread:
            return

        config = uvicorn.Config(
            app=self.app, host=self.host, port=self.port, log_level=http_log_level
        )
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        while not self.server.started:
            time.sleep(0.1)  # Wait for the server to start
        logger.info("Mooncake Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        if self.server_thread is None or self.server is None or not self.server.started:
            return

        self.server.should_exit = True
        self.server_thread.join()
        logger.info("Mooncake Bootstrap Server stopped.")

    def _register_worker(self, payload: RegisterWorkerPayload):
        """Handles registration of a prefiller worker."""
        with self._lock:
            self.workers[payload.dp_rank][payload.tp_rank] = payload.info
        logger.debug(
            "Registered worker: dp_rank=%d, tp_rank=%d at %s:%d",
            payload.dp_rank,
            payload.tp_rank,
            payload.info.hostname,
            payload.info.port,
        )
        return {"status": "ok"}

    def register_request(self, payload: RegisterRequestPayload):
        """Handles associating a request ID with a DP rank."""
        with self._lock:
            self.req_to_dp_rank[payload.req_id] = payload.dp_rank
        logger.debug(
            "Registered request '%s' to dp_rank=%d", payload.req_id, payload.dp_rank
        )
        return {"status": "ok"}

    def _query_requests(self, payload: QueryRequestsPayload) -> QueryRequestsResponse:
        """Handles a query (batch req_ids) from a decoder worker."""

        # We only support homogeneous TP now.
        if self.tp_size != payload.tp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    "heterogeneous TP is not supported yet. TP size mismatch: "
                    f"expected {self.tp_size}, got {payload.tp_size}"
                ),
            )

        results: dict[ReqId, tuple[str, WorkerInfo | None]] = {}
        with self._lock:
            for req_id in payload.req_ids:
                prefiller_dp_rank = self.req_to_dp_rank.get(req_id)
                if prefiller_dp_rank is None:
                    results[req_id] = ("Request ID not found", None)
                    continue

                prefiller_group = self.workers.get(prefiller_dp_rank)
                if not prefiller_group:
                    results[req_id] = (
                        f"Prefiller DP group {prefiller_dp_rank} not found",
                        None,
                    )
                    continue

                # todo: support heterogeneous TP.
                prefiller_tp_rank = payload.tp_rank
                worker_info = prefiller_group.get(prefiller_tp_rank)
                if worker_info is None:
                    results[req_id] = (
                        (
                            f"Prefiller TP rank {prefiller_tp_rank} not found "
                            "in DP group {prefiller_dp_rank}"
                        ),
                        None,
                    )
                    continue

                results[req_id] = ("ok", worker_info)

        return QueryRequestsResponse(results=results)


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers and requests.
    Decoders should get addr from kv_transfer_params.
    """
    assert vllm_config.parallel_config
    host = vllm_config.parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
