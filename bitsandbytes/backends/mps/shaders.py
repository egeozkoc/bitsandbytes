"""JIT-compiled Metal shaders for the MPS backend.

Shaders are compiled lazily through `torch.mps.compile_shader` on first use and
cached for the lifetime of the process. `get_library()` returns `None` when the
running PyTorch does not expose `compile_shader` or when compilation fails, so
callers must always keep a pure PyTorch fallback.
"""

import torch

_library = None
_compile_failed = not hasattr(torch.mps, "compile_shader")

# Threadgroup size used by the quantize kernels. Blocks smaller than this are
# dispatched with a correspondingly smaller threadgroup.
MAX_THREADGROUP_SIZE = 256

_SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Per-threadgroup maximum of |A[i]| over one quantization block.
// Returns the block absmax to every thread in the threadgroup.
inline float block_absmax(
    device const float* A,
    threadgroup float* scratch,
    uint base,
    uint block_end,
    uint tid,
    uint tg_size,
    uint simd_lane,
    uint simd_id
) {
    float m = 0.0f;
    for (uint i = base + tid; i < block_end; i += tg_size) {
        m = max(m, fabs(A[i]));
    }

    m = simd_max(m);
    if (simd_lane == 0) {
        scratch[simd_id] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        uint n_simds = (tg_size + 31u) / 32u;
        // absmax is non-negative, so -1 is a safe identity for inactive lanes.
        float v = (simd_lane < n_simds) ? scratch[simd_lane] : -1.0f;
        v = simd_max(v);
        if (simd_lane == 0) {
            scratch[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return scratch[0];
}

// Blockwise 8-bit quantization against an arbitrary sorted 256-entry code book.
//
// Each threadgroup owns one block. Values are normalized by the block absmax and
// mapped to the nearest code index via binary search over the midpoints between
// adjacent code entries, matching torch.bucketize(scaled, (code[:-1]+code[1:])/2).
kernel void quantize_blockwise(
    device const float* A         [[buffer(0)]],
    device const float* code      [[buffer(1)]],
    device uchar*       out       [[buffer(2)]],
    device float*       absmax    [[buffer(3)]],
    constant uint&      n         [[buffer(4)]],
    constant uint&      blocksize [[buffer(5)]],
    uint tid       [[thread_index_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tg_size   [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float scratch[MAX_SIMDGROUPS];
    threadgroup float lcode[256];

    for (uint i = tid; i < 256u; i += tg_size) {
        lcode[i] = code[i];
    }

    uint base = gid * blocksize;
    uint block_end = min(base + blocksize, n);

    float m = block_absmax(A, scratch, base, block_end, tid, tg_size, simd_lane, simd_id);
    if (tid == 0) {
        absmax[gid] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = 1.0f / max(m, 1e-38f);

    for (uint i = base + tid; i < block_end; i += tg_size) {
        float x = clamp(A[i] * inv, -1.0f, 1.0f);
        // Binary search for the first midpoint >= x over the 255 midpoints.
        uint lo = 0u, hi = 255u;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            float bound = 0.5f * (lcode[mid] + lcode[mid + 1u]);
            if (bound < x) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        out[i] = (uchar)lo;
    }
}
"""


def get_library():
    """Return the compiled shader library, or None if unavailable."""
    global _library, _compile_failed

    if _compile_failed:
        return None
    if _library is not None:
        return _library

    max_simdgroups = MAX_THREADGROUP_SIZE // 32
    source = f"#define MAX_SIMDGROUPS {max_simdgroups}\n" + _SHADER_SOURCE

    try:
        _library = torch.mps.compile_shader(source)
    except Exception:
        _compile_failed = True
        return None

    return _library
