"""Keep a checkpoint resident outside vLLM and push it into vLLM on demand.

One process per GPU (``torchrun`` or ``srun``, any number of nodes). Each rank
holds a contiguous byte range of the checkpoint for its whole lifetime: on its
GPU (``--residency gpu``), or in page-locked host memory on huge pages
(``--residency host``: the GPU only stages the chunk being sent). vLLM
sleeping, waking or restarting never touches it. Rank 0 serves a small HTTP
control API and writes its URL to ``<control_dir>/url``::

    torchrun --nnodes N --nproc-per-node G ... resident_publisher.py CKPT CTRL
    curl -X POST "$(cat CTRL/url)" \\
        -d '{"op": "publish", "base_url": "http://<vllm>:8000", "mode": "remote"}'

A publish goes through vLLM's own weight-transfer engines (the server runs
with ``VLLM_SERVER_DEV_MODE=1`` and ``weight_update_receiver``):

* ``remote`` (``--weight-transfer-config '{"backend": "nccl"}'``): owners send
  their ranges to rank 0, which broadcasts each packed bucket, cut exactly as
  vLLM's consumer cuts it, to every vLLM worker. The NCCL group is reused
  across publishes; a restarted vLLM (whose weight version is no longer the
  one the last publish set) gets a new group.
* ``colocated`` (``{"backend": "ipc"}``): the publisher ranks share the vLLM
  GPUs; every rank assembles each chunk on its GPU and hands it to the vLLM
  worker on the same GPU through CUDA IPC.

Other ops: ``health`` (per-rank residency) and ``shutdown``. The control API
is unauthenticated: keep its port on a trusted network. A remote publish that
fails mid-transfer leaves the vLLM workers waiting in it: restart the server.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import mmap
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests
import torch
import torch.distributed as dist

from receiver_patches import SAFETENSORS_DTYPES, read_safetensors_header

GiB = 1 << 30
AHEAD = 2  # remote: chunks in flight to rank 0, the one being sent included
PIECE = GiB  # remote: owner -> rank 0 transfers are cut to this size


@dataclass(frozen=True)
class Entry:
    name: str
    file: str
    offset: int  # absolute byte offset of the data in the file
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


def load_manifest(checkpoint: str) -> list[Entry]:
    """Every tensor of a safetensors checkpoint, in file and offset order."""
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))
    entries = []
    for file in files:
        path = os.path.join(checkpoint, file)
        header, data_start = read_safetensors_header(path)
        for name, e in header.items():
            if e["dtype"] not in SAFETENSORS_DTYPES:
                raise ValueError(f"unsupported dtype {e['dtype']} of {name}")
            start, end = e["data_offsets"]
            dtype, shape = SAFETENSORS_DTYPES[e["dtype"]], tuple(e["shape"])
            entries.append(
                Entry(name, path, data_start + start, end - start, dtype, shape)
            )
    names = [e.name for e in entries]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"no tensors, or duplicate tensor names, in {checkpoint}")
    return sorted(entries, key=lambda e: (e.file, e.offset))


def bucket_plan(entries: list[Entry], bucket_bytes: int) -> list[list[Entry]]:
    """Chunks cut where vLLM's packed consumer cuts its buckets: a bucket
    closes once its byte count exceeds the buffer size, the crossing tensor
    included, so each chunk is byte for byte the packed tensor vLLM expects.

    A tensor larger than a bucket goes after its module's other tensors:
    layerwise reload keeps a loaded view (and so its bucket) until the module
    is complete, so a huge tensor arriving first would pin its bucket."""
    last = {e.name.rpartition(".")[0]: i for i, e in enumerate(entries)}

    def order(item):
        i, e = item
        return (
            last[e.name.rpartition(".")[0]] + 0.5 if e.nbytes > bucket_bytes else i,
            i,
        )

    chunks, cur, size = [], [], 0
    for _, e in sorted(enumerate(entries), key=order):
        cur.append(e)
        size += e.nbytes
        if size > bucket_bytes:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def layout(entries: list[Entry], world: int):
    """Each tensor's owner rank and offset in that rank's resident buffer,
    and the buffer sizes: ``world`` contiguous ranges of ~equal bytes (a
    tensor belongs to the range its first byte falls in)."""
    total = sum(e.nbytes for e in entries) or 1
    owner, offset, size, acc = {}, {}, [0] * world, 0
    for e in entries:
        r = min(world - 1, acc * world // total)
        owner[e.name], offset[e.name] = r, size[r]
        size[r] += e.nbytes
        acc += e.nbytes
    return owner, offset, size


@dataclass
class Run:
    owner: int
    src: int  # offset in the owner's resident buffer
    dst: int  # offset in the chunk buffer
    nbytes: int


def chunk_runs(owner, offset, chunk) -> tuple[dict[str, int], list[Run], int]:
    """Where each tensor of ``chunk`` lands in a chunk buffer, and the
    contiguous owner ranges that fill it (one transfer each)."""
    pos, runs, p = {}, [], 0
    for e in chunk:
        r, src = owner[e.name], offset[e.name]
        pos[e.name] = p
        last = runs[-1] if runs else None
        if (
            last
            and last.owner == r
            and last.src + last.nbytes == src
            and last.dst + last.nbytes == p
        ):
            last.nbytes += e.nbytes
        else:
            runs.append(Run(r, src, p, e.nbytes))
        p += e.nbytes
    return pos, runs, p


def pieces(run: Run):
    """``(src, dst, nbytes)`` of a run's transfers: rank 0 can then always
    drain an announced chunk into a small scratch buffer, e.g. after a failure."""
    for p in range(0, run.nbytes, PIECE):
        yield run.src + p, run.dst + p, min(PIECE, run.nbytes - p)


def pinned_host_buffer(nbytes: int) -> torch.Tensor:
    """Page-locked host memory on transparent huge pages, faulted in before
    CUDA registers it (pinning and copies walk few pages). For buffers kept
    for the process lifetime: it is never unregistered."""
    mapping = mmap.mmap(-1, max(nbytes, 8), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        mapping.madvise(mmap.MADV_HUGEPAGE)
    except (AttributeError, OSError):
        pass  # base pages: correct, only slower to pin
    owner = np.frombuffer(mapping, dtype=np.uint8)
    owner[:: mmap.PAGESIZE] = 0
    tensor = torch.from_numpy(owner)
    result = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel(), 0)
    if result.value != 0 or not tensor.is_pinned():
        raise RuntimeError(f"cudaHostRegister of {nbytes} bytes failed: {result}")
    return tensor


def read_into(buf: torch.Tensor, ranges, threads: int = 32, piece: int = 64 << 20):
    """Read the ``(file, file offset, buffer offset, nbytes)`` ranges into the
    host tensor ``buf``, ``threads`` reads in flight (shared storage serves
    parallel reads much faster than one stream), dropping the page cache
    behind them."""
    view = (ctypes.c_char * buf.numel()).from_address(buf.data_ptr())
    mv = memoryview(view).cast("B")
    fds = {f: os.open(f, os.O_RDONLY) for f in {r[0] for r in ranges}}
    jobs = [
        (fds[file], src + p, dst + p, min(piece, n - p))
        for file, src, dst, n in ranges
        for p in range(0, n, piece)
    ]

    def read(job):
        fd, src, dst, n = job
        got = 0
        while got < n:
            r = os.preadv(fd, [mv[dst + got : dst + n]], src + got)
            if r <= 0:
                raise OSError(f"short read at offset {src + got}")
            got += r
        os.posix_fadvise(fd, src, n, os.POSIX_FADV_DONTNEED)

    try:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(read, jobs))
    finally:
        for fd in fds.values():
            os.close(fd)


def _get(base_url: str, path: str) -> dict:
    response = requests.get(f"{base_url}/{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _env(*names: str, default: str | None = None) -> str:
    for name in names:
        if name in os.environ:
            return os.environ[name]
    if default is None:
        raise KeyError(f"none of {names} is set")
    return default


class Publisher:
    def __init__(self, checkpoint: str, residency: str = "gpu", bucket_mb: int = 4096):
        self.residency = residency
        self.bucket = bucket_mb << 20
        self.rank = int(_env("RANK", "SLURM_PROCID"))
        self.world = int(_env("WORLD_SIZE", "SLURM_NTASKS"))
        local = int(_env("LOCAL_RANK", "SLURM_LOCALID", default="0"))
        self.device = torch.device("cuda", local)
        torch.cuda.set_device(self.device)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{_env('MASTER_ADDR')}:{_env('MASTER_PORT')}",
            rank=self.rank,
            world_size=self.world,
            timeout=datetime.timedelta(minutes=30),
            device_id=self.device,
        )
        # commands wait on this group between publishes
        self.cpu = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=365))
        self._warm_p2p()
        self.entries = load_manifest(checkpoint)
        self.owner, self.offset, self.size = layout(self.entries, self.world)
        self._plans: dict[int, list] = {}
        self._remote = None  # (key, engine, weight version it set)
        self._receiver_failed = None
        self._load()

    def _warm_p2p(self) -> None:
        """Open every owner -> rank 0 connection now. Opened lazily, the first
        receive from a new peer runs during vLLM's broadcast and fails."""
        t = torch.ones(1 << 20, dtype=torch.uint8, device=self.device)
        for peer in range(1, self.world):
            if self.rank == 0:
                dist.recv(t, src=peer)
            elif self.rank == peer:
                dist.send(t, dst=0)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu)

    def _load(self) -> None:
        t = time.perf_counter()
        ranges = [
            (e.file, e.offset, self.offset[e.name], e.nbytes)
            for e in self.entries
            if self.owner[e.name] == self.rank
        ]
        nbytes = self.size[self.rank]
        if self.residency == "host":
            self.buf = pinned_host_buffer(nbytes)
            read_into(self.buf, ranges)
        else:  # through a pinned staging buffer, 1 GiB of the slice at a time
            self.buf = torch.empty(
                max(nbytes, 1), dtype=torch.uint8, device=self.device
            )
            stage = torch.empty(min(nbytes, GiB), dtype=torch.uint8, pin_memory=True)
            for start in range(0, nbytes, GiB):
                end = min(start + GiB, nbytes)
                window = [
                    (f, src + lo - dst, lo - start, hi - lo)
                    for f, src, dst, n in ranges
                    if (lo := max(dst, start)) < (hi := min(dst + n, end))
                ]
                read_into(stage, window)
                self.buf[start:end].copy_(stage[: end - start])
            torch.cuda.synchronize(self.device)
        self.load_s = time.perf_counter() - t

    def health(self) -> dict:
        free, _ = torch.cuda.mem_get_info(self.device)
        return {
            "rank": self.rank,
            "ok": True,
            "residency": self.residency,
            "resident_gib": round(self.size[self.rank] / GiB, 3),
            "load_s": round(self.load_s, 3),
            "gpu_free_gib": round(free / GiB, 2),
        }

    def _plan(self, bucket_bytes: int) -> list:
        if bucket_bytes not in self._plans:
            self._plans[bucket_bytes] = [
                (*chunk_runs(self.owner, self.offset, c), c)
                for c in bucket_plan(self.entries, bucket_bytes)
            ]
        return self._plans[bucket_bytes]

    def _wait(self, event) -> None:
        """``event.synchronize()``, raising once the receiving vLLM failed."""
        while not event.query():
            if self._receiver_failed is not None and self._receiver_failed():
                raise RuntimeError("the vLLM server stopped receiving")
            time.sleep(50e-6)

    # remote: owners -> rank 0. Rank 0 announces the number of chunks once;
    # owners then queue their sends for all of them in plan order. On failure
    # rank 0 still receives every announced chunk, so no owner is left blocked.
    def _announce(self, n: int = 0) -> int:
        t = torch.tensor([n], dtype=torch.int64)
        dist.broadcast(t, src=0, group=self.cpu)
        return int(t.item())

    def _serve_rank0(self, plan) -> None:
        # NCCL sends from the GPU: host-resident runs go through one buffer,
        # reused safely because dist.send's wait orders the next copy after it
        stage = None
        if not self.buf.is_cuda:
            stage = torch.empty(PIECE, dtype=torch.uint8, device=self.device)
        for k in range(self._announce()):
            for run in plan[k][1]:
                if run.owner == self.rank:
                    for src, _, n in pieces(run):
                        data = self.buf[src : src + n]
                        if stage is not None:
                            data = stage[:n].copy_(data, non_blocking=True)
                        dist.send(data, dst=0)
        torch.cuda.synchronize(self.device)

    def _gather(self, plan):
        """Yield each chunk gathered on rank 0, with up to ``AHEAD`` chunks'
        receives in flight (bounded by consumption events). A chunk grown
        around a huge tensor is gathered alone, as vLLM's patched consumer
        receives it."""
        self._announced = self._announce(len(plan))

        def post():
            _, runs, nbytes, _ = plan[self._posted]
            buf, reqs = (
                torch.empty(max(nbytes, 1), dtype=torch.uint8, device=self.device),
                [],
            )
            for run in runs:
                if run.owner == 0:  # pinned or device: never blocks the host
                    dst = buf[run.dst : run.dst + run.nbytes]
                    dst.copy_(
                        self.buf[run.src : run.src + run.nbytes], non_blocking=True
                    )
                else:
                    for _, dst, n in pieces(run):
                        reqs.append(dist.irecv(buf[dst : dst + n], src=run.owner))
            self._posted += 1
            self._pending += reqs
            return buf, reqs

        def ahead() -> bool:
            if not posted:
                return True
            sizes = [b.numel() for b, _ in posted] + [plan[self._posted][2]]
            return len(posted) < AHEAD and max(sizes) <= 2 * self.bucket

        posted, consumed = deque(), deque()
        for k in range(len(plan)):
            while self._posted < len(plan) and ahead():
                if len(consumed) >= AHEAD:
                    self._wait(consumed.popleft())
                posted.append(post())
            buf, reqs = posted.popleft()
            for r in reqs:
                r.wait()
            self._pending = [r for _, rs in posted for r in rs]
            yield k, buf
            consumed.append(torch.cuda.Event())
            consumed[-1].record()
            del buf, reqs  # freed (in stream order) before the next is posted

    def _finish_gather(self, plan) -> None:
        """Receive whatever is still announced: the owners send all of it."""
        if self._announced is None:
            self._announced = self._announce(0)
        reqs, self._pending = self._pending, []
        # not on the current stream, which may hold a broadcast to a failed
        # receiver (completed only once the group is aborted)
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            if self._posted < self._announced:
                scratch = torch.empty(PIECE, dtype=torch.uint8, device=self.device)
            while self._posted < self._announced:
                for run in plan[self._posted][1]:
                    if run.owner != 0:
                        for _, _, n in pieces(run):
                            reqs.append(dist.irecv(scratch[:n], src=run.owner))
                self._posted += 1
        while not all(r.is_completed() for r in reqs):
            time.sleep(1e-3)

    def _send_packed(self, engine, plan, metas, version: str) -> None:
        """Drive vLLM's update handshake (the requests ``send_weights`` makes)
        and broadcast every gathered chunk as one packed bucket."""
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferUpdateInfo,
        )

        info = NCCLWeightTransferUpdateInfo(
            names=[m.name for m in metas],
            dtype_names=[str(m.dtype).split(".")[-1] for m in metas],
            shapes=[list(m.shape) for m in metas],
        )
        client, group = engine.client, engine.model_update_group
        client.start_weight_update()
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = exe.submit(client.update_weights, asdict(info))
            # a worker that fails stops receiving and its broadcast never
            # completes: every wait also watches the update request
            self._receiver_failed = lambda: (
                future.done() and future.exception() is not None
            )
            stream = torch.cuda.current_stream()
            chunks = self._gather(plan)
            try:
                for _, buf in chunks:
                    group.broadcast(buf, src=0, stream=stream)
                    del buf
            finally:
                chunks.close()  # its buffers, before _finish_gather allocates
            done = torch.cuda.Event()
            done.record(stream)
            self._wait(done)
            future.result()
        finally:
            exe.shutdown(wait=False)  # never join: a failed broadcast hangs
            self._receiver_failed = None
        client.finish_weight_update(weight_version=version)

    def _remote_engine(self, base_url: str, bucket: int, source):
        """The cached transfer engine, if it still reaches this server (a
        restarted vLLM keeps its URL but resets the weight version), else a
        new one."""
        from vllm.distributed.weight_transfer import (
            HTTPVLLMWeightSyncClient,
            WeightTransferTrainerFactory,
        )
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
        from vllm.utils.network_utils import get_ip, get_open_port

        if self._remote is not None:
            key, engine, version = self._remote
            if (
                key == (base_url, bucket)
                and _get(base_url, "weight_info").get("weight_version") == version
            ):
                return engine
            self._drop_remote()
        world = _get(base_url, "get_world_size")
        init = NCCLTrainerInitInfo(
            rank=0,
            master_address=get_ip(),
            master_port=get_open_port(),
            world_size=world["world_size"] + 1,
            packed=True,
            packed_buffer_size_bytes=bucket,
            # with one buffer an unpatched vLLM consumer loads on one stream
            packed_num_buffers=1,
        )
        client = HTTPVLLMWeightSyncClient(base_url, timeout=3600)
        return WeightTransferTrainerFactory.trainer_init(
            init, client=client, source=source
        )

    def _drop_remote(self) -> None:
        """Abort the cached transfer group (its peers may be gone)."""
        cached, self._remote = self._remote, None
        group = getattr(cached[1], "model_update_group", None) if cached else None
        if group is not None:
            try:
                group.destroy()  # ncclCommAbort, bounded by a timeout
            except Exception as exc:  # nothing left to release
                print(f"[rank {self.rank}] group abort: {exc!r}", file=sys.stderr)
            cached[1].model_update_group = None

    def _assemble(self, plan):
        """Colocated: every rank assembles every chunk (owner broadcasts)."""
        for pos, runs, nbytes, chunk in plan:
            torch.cuda.synchronize(self.device)  # the previous chunk is consumed
            out = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=self.device)
            for run in runs:
                dst = out[run.dst : run.dst + run.nbytes]
                if run.owner == self.rank:
                    dst.copy_(self.buf[run.src : run.src + run.nbytes])
                dist.broadcast(dst, src=run.owner)
            torch.cuda.synchronize(self.device)
            for e in chunk:
                raw = out[pos[e.name] : pos[e.name] + e.nbytes]
                if raw.data_ptr() % e.dtype.itemsize:  # after an odd-sized tensor
                    raw = raw.clone()
                yield e.name, raw.view(e.dtype).view(e.shape)

    def _publish_remote(self, cmd, bucket, plan, metas, source) -> None:
        base_url = cmd["base_url"].rstrip("/")
        version = cmd.get("weight_version") or f"publisher-{uuid.uuid4().hex[:12]}"
        engine = None
        try:
            engine = self._remote_engine(base_url, bucket, source)
            self._send_packed(engine, plan, metas, version)
            self._remote = ((base_url, bucket), engine, version)
        except Exception:
            # a broadcast may be left in flight: abort the group
            if self._remote is None and engine is not None:
                self._remote = (None, engine, None)
            self._drop_remote()
            raise

    def _publish_colocated(self, cmd, bucket, source) -> None:
        from vllm.distributed.weight_transfer import (
            HTTPVLLMWeightSyncClient,
            WeightTransferTrainerFactory,
        )
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

        init = IPCTrainerInitInfo(
            rank=self.rank, packed=True, packed_buffer_size_bytes=bucket
        )
        client = HTTPVLLMWeightSyncClient(cmd["base_url"].rstrip("/"), timeout=3600)
        engine = WeightTransferTrainerFactory.trainer_init(
            init, client=client, source=source
        )
        engine.send_weights()
        engine.shutdown()

    def publish(self, cmd: dict) -> dict:
        from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

        mode = cmd.get("mode", "remote")
        if mode not in ("remote", "colocated"):
            raise ValueError(f"unknown publish mode {mode!r}")
        bucket = self.bucket
        if mode == "colocated":  # the IPC engine keeps every tensor in a bucket
            bucket = max(bucket, -(-max(e.nbytes for e in self.entries) >> 20) << 20)
        plan = self._plan(bucket)
        metas = [
            ParamMeta(name=e.name, dtype=e.dtype, shape=e.shape)
            for *_, c in plan
            for e in c
        ]
        pub = self

        class Source(WeightSource):
            # remote publishes send the gathered chunks themselves: the engine
            # only iterates its source in colocated mode
            def metadata(self):
                return metas

            def __iter__(self):
                return pub._assemble(plan)

        t0 = time.perf_counter()
        try:
            if mode == "colocated":
                self._publish_colocated(cmd, bucket, Source())
            elif self.rank != 0:
                self._serve_rank0(plan)
            else:
                self._announced, self._posted, self._pending = None, 0, []
                try:
                    self._publish_remote(cmd, bucket, plan, metas, Source())
                finally:
                    self._finish_gather(plan)  # the owners wait for the announcement
            torch.cuda.synchronize(self.device)
        except Exception:
            print(
                f"[rank {self.rank}] publish failed:\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            raise
        seconds = time.perf_counter() - t0
        gib = sum(e.nbytes for e in self.entries) / GiB
        return {
            "rank": self.rank,
            "ok": True,
            "mode": mode,
            "send_s": round(seconds, 3),
            "gib": round(gib, 3),
            "gib_per_s": round(gib / seconds, 2),
        }

    def run(self, control_dir: str) -> None:
        inbox: queue.Queue = queue.Queue()
        if self.rank == 0:
            url = self._serve_http(inbox)
            os.makedirs(control_dir, exist_ok=True)
            tmp = os.path.join(control_dir, "url.tmp")
            with open(tmp, "w") as f:
                f.write(url)
            os.replace(tmp, os.path.join(control_dir, "url"))
        while True:
            reply = None
            box = [None]
            if self.rank == 0:
                box[0], reply = inbox.get()
            dist.broadcast_object_list(box, src=0, group=self.cpu)
            op = box[0].get("op")
            try:
                if op == "publish":
                    res = self.publish(box[0])
                elif op == "health":
                    res = self.health()
                elif op == "shutdown":
                    res = {"rank": self.rank, "ok": True}
                else:
                    raise ValueError(f"unknown op {op!r}")
            except Exception as exc:  # reported; the publisher stays up
                res = {"rank": self.rank, "ok": False, "error": repr(exc)}
            gathered = [None] * self.world if self.rank == 0 else None
            dist.gather_object(res, gathered, dst=0, group=self.cpu)
            if self.rank == 0:
                ok = all(r.get("ok") for r in gathered)
                reply.put({"op": op, "ok": ok, "ranks": gathered})
            if op == "shutdown":
                break
        dist.barrier(group=self.cpu)
        dist.destroy_process_group()

    def _serve_http(self, inbox: queue.Queue) -> str:
        lock = threading.Lock()  # one command at a time

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                try:
                    cmd = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    cmd = None
                if isinstance(cmd, dict):
                    reply: queue.Queue = queue.Queue()
                    with lock:
                        inbox.put((cmd, reply))
                        res = reply.get()
                    code = 200 if res["ok"] else 500
                else:
                    res, code = {"ok": False, "error": "expected a JSON object"}, 400
                data = json.dumps(res).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://{socket.gethostname()}:{server.server_address[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="local safetensors checkpoint directory")
    parser.add_argument("control_dir", help="rank 0 writes its control URL here")
    parser.add_argument("--residency", choices=("gpu", "host"), default="gpu")
    parser.add_argument(
        "--bucket-mb",
        type=int,
        default=4096,
        help="packed bucket size; larger buckets mean fewer load_weights calls "
        "in vLLM (each has a fixed cost), at that much receive memory per "
        "bucket in flight (vLLM's default is 1024)",
    )
    args = parser.parse_args()
    publisher = Publisher(args.checkpoint, args.residency, args.bucket_mb)
    if publisher.rank == 0:
        print(json.dumps(publisher.health()), flush=True)
    publisher.run(args.control_dir)


if __name__ == "__main__":
    main()
