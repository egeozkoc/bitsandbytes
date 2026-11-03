# MPS blockwise quantization benchmark

Run the public operator against separate baseline and patched source checkouts.
Both checkouts must be importable, including their native bitsandbytes library.
Use the same Python environment and machine, with MPS and `torch.mps.compile_shader` available.

```sh
git worktree add ../bitsandbytes-baseline a616886cacf86701514f0cb1d6be760f967481c0
python benchmarking/mps/quantize_blockwise.py --repo ../bitsandbytes-baseline --mode baseline --output baseline.jsonl
python benchmarking/mps/quantize_blockwise.py --repo . --mode metal --output metal.jsonl
```

The baseline requires successful `torch.compile(..., dynamic=True, fullgraph=True)`
compilation of the existing MPS computation. Metal mode fails if it falls back.
Input casting, layout copies, output allocations and dispatch are included.
Each case uses 10 warmups and 7 batches of 20 calls, synchronizing after each batch.
The JSONL output includes all batch samples, first-call cost and the median of
15 separately synchronized calls. Inputs are seeded, inference mode is enabled,
and CPU threads are limited to one. Run the processes sequentially to avoid GPU contention.

## Measurements

2026-11-03: Apple M5 (10 GPU cores, 24 GiB), macOS 26.0, Python 3.12.13, AC power.
Median batch milliseconds per call for 16,777,216 elements:

| Input / block size | Torch 2.12.1 baseline → Metal (ms) | Speedup | Torch 2.15.0 baseline → Metal (ms) | Speedup |
|---|---:|---:|---:|---:|
| float32 / 64 | 7.777 → 2.122 | 3.66× | 7.685 → 2.123 | 3.62× |
| float32 / 256 | 6.997 → 1.848 | 3.79× | 7.010 → 1.860 | 3.77× |
| float32 / 4096 | 5.635 → 1.284 | 4.39× | 5.614 → 1.287 | 4.36× |
| float16 / 256 | 7.758 → 2.590 | 2.99× | 7.765 → 2.610 | 2.97× |
| bfloat16 / 256 | 7.754 → 2.597 | 2.99× | 7.764 → 2.618 | 2.97× |

These are operator measurements, not training speedups. The complete sweep has
23 configurations per Torch version: three dtypes, sizes 256 through 16M,
block sizes 64/256/4096, partial tails and transposed inputs. Small synchronized
calls are noisy and can regress (FP32/65,536 was about 8% slower on Torch 2.15
in this run); these data do not establish a reliable dispatch size threshold.
Only this M5/macOS combination was tested. First-call timings include warm
system caches and should not be interpreted as a cold-device compilation cost.

## Regression tests

```sh
BNB_TEST_DEVICE=mps python -m pytest tests/test_mps.py
```

The shader tests forbid fallback and cover NaN propagation, full versus partial
absmax statistics, dtype/layout parity, and code-boundary rounding. Additional
tests simulate missing compiler APIs and compiler/launch failures. Oversized
index guards use meta tensors so they also run when CPU is selected.
