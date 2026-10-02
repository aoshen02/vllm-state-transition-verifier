---
title: GB200 Weight Update Benchmark
---

# GB200 weight-update benchmark

Measured on GB200 for the standalone `scripts/vllm_weight_update_client`
publisher. The full-update numbers below
time `start_weight_update`, every bucket, and `finish_weight_update` on an
already running vLLM receiver. Checkpoint staging, receiver startup, NCCL
initialization, and inference checks are reported separately. Each full-model
number is one observation, not a latency distribution.

## Setup

| Item | DeepSeek-V4.1-Flash | GLM-5.3-Flash |
| --- | --- | --- |
| Safetensors shards / tensors | 48 / 96,085 | 62 / 76,108 |
| Transferred payload | 510,286,023,000 bytes | 328,326,771,576 bytes |
| Validated receiver | TP2+EP2, 120 GiB CPU offload per rank | TP2+EP2, no CPU offload |

The publisher and TP2 receiver used separate GB200 GPUs. Checkpoints were
SHA-verified copies on local SSD. Earlier storage trials are shown as context,
not paired comparisons with the final setup.
The verified receiver used `--weight-transfer-config '{"backend":"nccl"}'`;
the publisher supplied packed transfer settings during NCCL initialization.

## Storage and staging

| Measurement | Result | Boundary |
| --- | ---: | --- |
| Nonlocal NVMe direct read | ~2.49 GiB/s | Separate O_DIRECT probe |
| Local SSD direct read | 25.677 GiB/s | 64 GiB, 64 readers, 4 MiB blocks; physical counters confirmed reads |
| Local SSD direct write | 21.203 GiB/s | 64 GiB, 16 writers, fsync |
| DeepSeek source → local SSD | 178.164 s copy + 111.796 s SHA check | 510.31 GB, four-worker complete stage |
| GLM source → local SSD | 81.288 s copy + 50.077 s SHA check | 328.37 GB complete stage |

The 25.677 GiB/s result is an isolated device probe. Its ideal read-only
lower bound for the 510.29 GB DeepSeek payload is about 18.5 seconds, but it
does not predict an update that also copies to GPU, sends NCCL data and loads
receiver weights. The full staging copy and the isolated nonlocal NVMe probe
used different sources. Temporary SSD copies were removed after validation.

## Complete weight-update time

| Model and source | Configuration | Time | Correctness |
| --- | --- | ---: | --- |
| DeepSeek, nonlocal storage | Natural-order early baseline | 921.855 s | Full transfer and fixed-output oracle passed |
| DeepSeek, nonlocal storage | Physical expert order, direct-file H2D, 8×256 MiB readers | 335.093 s | Full transfer and oracle passed |
| DeepSeek, nonlocal storage | Same candidate, 16 readers | 285.403 s | Full transfer and oracle passed |
| DeepSeek, local SSD, earlier trial | No prefetch | 75.508 s | 68/68 buckets, finish, fixed-output equality |
| DeepSeek, local SSD, earlier trial | General direct-file candidate | 74.611 s | 69/69 buckets, finish, fixed-output equality |
| DeepSeek, local SSD, paired trial | Baseline, no prefetch | **70.7775 s** | 69/69 buckets and fixed-output equality |
| DeepSeek, local SSD, paired trial | 8 GiB bounded prefetch, direct-file reads, 16×256 MiB readers; A→B | **48.3697 s** | 69/69 buckets, finish, changed oracle |
| DeepSeek, local SSD, paired trial | Same candidate; fresh B receiver, B→A | 48.9322 s | Original A oracle restored |
| GLM, local SSD | Physical nonexpert order, A→B | 33.20 s | **Invalid:** NaN output despite successful transfer |
| GLM, local SSD | Physical experts, natural nonexperts; A→A | 32.3466 s | All runtime hashes and oracle equal |
| GLM, local SSD | Same safe layout; A→B2 | **32.8507 s** | Only target norm changed; fresh-B2 oracle equal |
| GLM, local SSD | Same safe layout; B2→A | 32.3792 s | All runtime hashes and original A oracle restored |

The meaningful paired DeepSeek comparison is **70.7775 →
48.3697 seconds**, a 22.4078-second reduction (31.7%, 1.46× throughput) in
these single samples. It covers a full 510.29 GB transfer and excludes
receiver cold start. The candidate combines several options; the measurement
does not isolate each option's individual effect. GLM has **no valid
pre-optimization baseline** under the same TP2 configuration: the 33.20 s
physical-nonexpert transfer produced wrong weights and cannot support a
speedup claim. GLM's correct A→B2 update carried 328.33 GB in 32.8507 s,
about 10.0 GB/s effective payload rate. This is end-to-end payload rate,
not physical SSD utilization.

The DeepSeek paired baseline spent 15.412 s materializing nonexpert
weights, 2.243 s in their RPC/NCCL work, 23.119 s materializing experts,
and 28.704 s in expert RPC/NCCL work. The candidate prefetch overlapped the
expert stages: expert materialization summed to 21.623 s, but source wait
on the critical path was only 0.784 s and expert bucket wall time was
32.452 s. The candidate's nonexpert bucket wall time was 15.670 s. For GLM
A→B2, 30 nonexpert buckets took 3.848 s on the critical path and 43 expert
buckets took 28.303 s; materialization and RPC/NCCL overlap and must not be
added as wall time. GLM's publisher process took 36.201 s including 1.420 s
manifest preparation and 1.914 s NCCL initialization.

## Correctness and GPU residency

DeepSeek A→B changed deterministic token IDs and logprobs; a fresh B receiver
then accepted B→A and reproduced the original A token IDs, text and logprobs.
All transactions sent the same 96,085-tensor, 510,286,023,000-byte manifest
and completed every bucket plus finish. The immediate next A→B on that same
receiver failed at bucket 22/69 with GPU OOM; there is no same-worker
DeepSeek repeat-success claim for the tested CPU-offloaded layout.

GLM's full A→A, A→B2 and B2→A sequence ran in one receiver without CPU
offload, with about 160,547 MiB of model memory on each receiver GPU.
SHA-256 of all 1,675 live parameters and buffers per rank matched fresh A
after safe A→A. B2 scaled only the 8,192-byte BF16 final norm by 0.5; after
A→B2, that was the only changed runtime tensor per rank and its digest
matched the B2 checkpoint. Updated B2 token IDs, text, prompt logprobs and
completion logprobs exactly matched a separate fresh disk-loaded B2 receiver.
B2→A restored all 1,675 hashes per rank and the original oracle.

The incorrect GLM run exposed a real ordering constraint. `Glm5Next`
buffers FP8 attention weights and scales only inside each `model.load_weights`
call. Physical nonexpert order split 30 such pairs across 512 MiB buckets;
the transfer returned success but changed two fused projection tensors per
rank and produced NaN logprobs. Natural nonexpert order split none. The client
now rejects split GLM FP8 pairs before NCCL setup. Expert tensors can still
use physical order, direct-file copies and bounded prefetch.

The separate `ipc_cache` daemon retained about 80 GiB of GLM weights per GPU
after an engine stopped, and a restarted engine mapped them
quickly. Its initial A output nevertheless differed from a normal disk load
and produced NaN logprobs on one path. This does **not** establish a correct
cross-process GPU-resident update. DeepSeek MXFP8 is rejected by this pinned
runtime's IPC cache gate. GPU residency is validated only for GLM updates
within one running worker.

## Measurement limits

The update durations exclude model startup and staging. One DeepSeek cold
receiver reported 6,975.30 s model loading; GLM fresh B2 TP2 reported
93.43–93.47 s model loading per rank. Temporary SSD copies were removed
after the tests. The later NCCL trainer API compatibility merge has not been
remeasured with a full model, so these timings describe the validated
pre-merge publisher.
