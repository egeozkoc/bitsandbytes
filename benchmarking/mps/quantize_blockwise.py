"""Benchmark the actual public operator in separate upstream/prototype processes."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--mode", choices=["baseline", "metal"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import torch

    import bitsandbytes as bnb
    from bitsandbytes.backends.mps import ops

    torch.set_num_threads(1)
    assert torch.backends.mps.is_available()
    out = args.output.open("w")

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    emit(
        {
            "environment": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "macos": platform.mac_ver()[0],
                "repo": bnb.__file__,
                "mode": args.mode,
                "cpu_threads": 1,
                "warmup": 10,
                "rounds": 7,
                "calls_per_round": 20,
            }
        }
    )
    if args.mode == "metal":
        from bitsandbytes.backends.mps import shaders

        t = time.perf_counter()
        assert shaders.get_library() is not None
        emit({"shader_compile_ms": (time.perf_counter() - t) * 1000})

        def no_fallback(*a, **kw):
            raise AssertionError("Shader unexpectedly fell back")

        ops._quantize_blockwise_compute = no_fallback
    else:
        # Same function/options as upstream, but surface any compilation failure
        # instead of allowing its exception wrapper to silently time eager execution.
        raw = ops._quantize_blockwise_compute.__wrapped__

    code = bnb.functional.create_dynamic_map().to("mps")
    cases = [
        (str(dtype).split(".")[-1], n, 256, "contiguous")
        for dtype in (torch.float32, torch.float16, torch.bfloat16)
        for n in (256, 4096, 65536, 1024**2, 4096**2)
    ]
    cases += [("float32", 4096**2, b, "contiguous") for b in (64, 4096)]
    cases += [(dtype, 1024**2 + 17, 256, "partial") for dtype in ("float32", "float16", "bfloat16")]
    cases += [(dtype, 1024**2, 256, "transpose") for dtype in ("float32", "float16", "bfloat16")]
    with torch.inference_mode():
        for dtype_name, n, blocksize, layout in cases:
            torch.manual_seed(2011)
            a = torch.randn(n, dtype=getattr(torch, dtype_name)).to("mps")
            if layout == "transpose":
                a = a.reshape(1024, 1024).T
            if args.mode == "baseline":
                torch._dynamo.reset()
                ops._quantize_blockwise_compute = torch.compile(raw, dynamic=True, fullgraph=True)

            def call(a=a):
                return torch.ops.bitsandbytes.quantize_blockwise.default(a, code, blocksize)

            torch.mps.synchronize()
            t = time.perf_counter()
            result = call()
            torch.mps.synchronize()
            first = (time.perf_counter() - t) * 1000
            for _ in range(10):
                result = call()
            torch.mps.synchronize()
            batches = []
            for _ in range(7):
                t = time.perf_counter()
                for _ in range(20):
                    result = call()
                torch.mps.synchronize()
                batches.append((time.perf_counter() - t) * 1000 / 20)
            latency = []
            for _ in range(15):
                t = time.perf_counter()
                result = call()
                torch.mps.synchronize()
                latency.append((time.perf_counter() - t) * 1000)
            emit(
                {
                    "dtype": dtype_name,
                    "n": n,
                    "blocksize": blocksize,
                    "layout": layout,
                    "first_call_ms": first,
                    "batch_median_ms": statistics.median(batches),
                    "batch_min_ms": min(batches),
                    "batch_max_ms": max(batches),
                    "sync_latency_median_ms": statistics.median(latency),
                    "samples_ms": batches,
                }
            )
            del result, a
            torch.mps.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
