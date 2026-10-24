from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

sys.path.insert(
    0, str(Path(__file__).parents[1] / "scripts" / "vllm_weight_update_client")
)

import receiver_patches  # noqa: E402
import resident_publisher  # noqa: E402


def _checkpoint(tmp_path: Path) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    shards = [
        {
            "model.layers.0.mlp.w": torch.randn(8, 32, generator=g).bfloat16(),
            "model.layers.0.mlp.big": torch.randn(4, 256, generator=g),
            "model.layers.0.norm": torch.ones(32),
        },
        {
            "model.layers.1.mlp.w": torch.randn(16, 16, generator=g).to(
                torch.float8_e4m3fn
            ),
            "model.layers.1.ids": torch.arange(7, dtype=torch.int64),
            "model.scale": torch.tensor(1.5),
        },
    ]
    for i, shard in enumerate(shards):
        safetensors_torch.save_file(shard, tmp_path / f"model-{i}.safetensors")
    return {k: v for shard in shards for k, v in shard.items()}


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    )


def test_mapped_safetensors_are_views_of_the_file(tmp_path):
    want = _checkpoint(tmp_path)
    got = {}
    for path in sorted(tmp_path.glob("*.safetensors")):
        got.update(receiver_patches.mapped_safetensors(str(path)))
    assert set(got) == set(want)
    assert all(_same(got[k], want[k]) for k in want)
    assert receiver_patches._file_mapped(got["model.scale"])
    assert not receiver_patches._file_mapped(want["model.scale"])


@pytest.mark.parametrize("world", [1, 3])
def test_publisher_buckets_reassemble_the_checkpoint(tmp_path, world):
    want = _checkpoint(tmp_path)
    entries = resident_publisher.load_manifest(str(tmp_path))
    owner, offset, size = resident_publisher.layout(entries, world)
    # each rank's resident buffer: its contiguous range of the checkpoint
    buffers = [bytearray(n) for n in size]
    for e in entries:
        with open(e.file, "rb") as f:
            f.seek(e.offset)
            buffers[owner[e.name]][offset[e.name] : offset[e.name] + e.nbytes] = f.read(
                e.nbytes
            )
    bucket = 2048
    chunks = resident_publisher.bucket_plan(entries, bucket)
    order = [e.name for c in chunks for e in c]
    assert sorted(order) == sorted(want)
    # a tensor larger than a bucket follows its module's other tensors
    assert order.index("model.layers.0.mlp.big") > order.index("model.layers.0.mlp.w")
    for chunk in chunks[:-1]:  # vLLM's cut: closed by the crossing tensor
        sizes = [e.nbytes for e in chunk]
        assert sum(sizes) > bucket >= sum(sizes[:-1])
    for chunk in chunks:
        pos, runs, nbytes = resident_publisher.chunk_runs(owner, offset, chunk)
        out = bytearray(nbytes)
        for run in runs:
            src = buffers[run.owner][run.src : run.src + run.nbytes]
            out[run.dst : run.dst + run.nbytes] = src
        for e in chunk:
            raw = torch.frombuffer(out, dtype=torch.uint8)[pos[e.name] :]
            tensor = raw[: e.nbytes].view(e.dtype).view(e.shape)
            assert _same(tensor, want[e.name])
