#!/usr/bin/env python3
"""Apply the runtime-switchable intervention patch to a pristine llama.cpp b10434 tree.

    python probes/apply_probe_patch.py third_party/llama.cpp-b10434

Every intervention is gated on the environment variable GGML_PROBE_FLAGS, read once
per device when the CUDA backend is created. With the variable unset (or 0) every
patched function takes its original upstream code path, so one build serves the
baseline and all interventions.

    bit 0 (1)  I1  MMQ D4 activation scale rounded through FP16 before storing
                   (matches MMVQ's FP16 scale; layout unchanged)        -> tests M1
    bit 1 (2)  I2  MMQ activation codes computed with MMVQ's formula
                   d = amax/127, q = roundf(x/d)                         -> tests M2
    bit 2 (4)  I4  MMVQ applies the offset/min correction the way MMQ does on the
                   same build: Q4_K/Q5_K min term from the FP32 activation
                   sum; Q5_0 (all builds) and Q4_0 (Turing-MMA builds) offset from
                   integer codes                                         -> tests M4
    bit 3 (8)  I5  MMQ sums each 32-value activation block in MMVQ's association
                   before the FP16 rounding of the sum                   -> tests M4b
    bit 5 (32) I7  (Turing-MMA builds) MMVQ Q4_K/Q5_K reproduce MMQ-MMA's FP16 sub-block
                   scale products d*sc, dmin*m, plus I4's min term   -> tests M6
    bit 7 (128) I9  MMVQ Q4_1/Q5_1 form the scale products in FP32 instead of FP16 __hmul2
                    (as MMQ does)                                      -> tests M7
    bit 8 (256) I10 global code sum: MMVQ's Q8_1 sum = d*sum(q), and MMQ's DS4 sum = d*sum(q)
                    for every DS4 format                               -> consistent formulation
    bit 9 (512) I11 BF16 precision (8 mantissa bits) for the stored activation scale and sum
                    in MMVQ's Q8_1 ds and MMQ's DS4 ds; D4/D2S6 untouched -> emulates the
                    precision (not the range) of llama.cpp PR #22571
    bit 6 (64) I8  MMQ stores the activation CODE sum d*sum(q) instead of the FP32 sum
                   for Q4_K/Q5_K weights (MMVQ's own formulation)    -> tests M8 (gate gain)
    bit 4 (16) I6  dispatch: never use MMVQ (dense mul_mat), so a decoded token and
                   a batched token run the same MMQ kernel              -> tests M5 and
                   is the path-invariance mitigation (latency cost measured)

The script refuses to run twice and asserts every edit matched exactly once, so a
silent partial patch is impossible. README.md lists what each switch removes.
"""

from __future__ import annotations

import sys
from pathlib import Path


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path.name}: expected exactly 1 match, found {n}:\n{old[:200]}")
    path.write_text(text.replace(old, new))


def main(root: Path) -> None:
    cu = root / "ggml" / "src" / "ggml-cuda"
    if "GGML_PROBE_FLAGS" in (cu / "quantize.cu").read_text():
        raise SystemExit("tree is already patched")

    # ------------------------------------------------------------------ quantize.cu
    q = cu / "quantize.cu"
    edit(q, '#include "quantize.cuh"\n',
         '#include "quantize.cuh"\n'
         '#include <cstdlib>\n\n'
         '// [cs6886 probe] intervention flags; 0 = upstream behaviour (docs/PROJECT.md)\n'
         'static __constant__ int ggml_probe_quant_flags;\n\n'
         'void ggml_cuda_probe_init_quant(void) {\n'
         '    const char * e = getenv("GGML_PROBE_FLAGS");\n'
         '    const int f = e ? atoi(e) : 0;\n'
         '    CUDA_CHECK(cudaMemcpyToSymbol(ggml_probe_quant_flags, &f, sizeof(int)));\n'
         '}\n')
    edit(q,
         '    const float d_inv = 127.0f / amax;\n'
         '    char4 q;\n'
         '    q.x = roundf(xi.x*d_inv);\n'
         '    q.y = roundf(xi.y*d_inv);\n'
         '    q.z = roundf(xi.z*d_inv);\n'
         '    q.w = roundf(xi.w*d_inv);\n'
         '    const float d = 1.0f / d_inv;\n',
         '    char4 q;\n'
         '    float d;\n'
         '    if (ggml_probe_quant_flags & 2) { // I2: MMVQ formula\n'
         '        d = amax / 127.0f;\n'
         '        q.x = amax == 0.0f ? 0 : roundf(xi.x / d);\n'
         '        q.y = amax == 0.0f ? 0 : roundf(xi.y / d);\n'
         '        q.z = amax == 0.0f ? 0 : roundf(xi.z / d);\n'
         '        q.w = amax == 0.0f ? 0 : roundf(xi.w / d);\n'
         '    } else {\n'
         '    const float d_inv = 127.0f / amax;\n'
         '    q.x = roundf(xi.x*d_inv);\n'
         '    q.y = roundf(xi.y*d_inv);\n'
         '    q.z = roundf(xi.z*d_inv);\n'
         '    q.w = roundf(xi.w*d_inv);\n'
         '    d = 1.0f / d_inv;\n'
         '    }\n')
    # I5: MMQ computes each 32-value activation sum in exactly the association that
    # MMVQ's warp_reduce_sum<32> uses for lane 0 (xor 16, 8, 4, 2, 1). MMQ thread t holds
    # values 4t..4t+3, so lane xor 16/8/4 are thread xor 4/2/1 and lane xor 2/1 are
    # within-thread pairs: sum = (c0 + c2) + (c1 + c3).
    edit(q,
         '        sum = xi.x + xi.y + xi.z + xi.w;\n',
         '        if (vals_per_sum == 32 && (ggml_probe_quant_flags & 8)) { // I5\n'
         '            float c0 = xi.x, c1 = xi.y, c2 = xi.z, c3 = xi.w;\n'
         '#pragma unroll\n'
         '            for (int offset = 4; offset > 0; offset >>= 1) {\n'
         '                c0 += __shfl_xor_sync(0xFFFFFFFF, c0, offset, WARP_SIZE);\n'
         '                c1 += __shfl_xor_sync(0xFFFFFFFF, c1, offset, WARP_SIZE);\n'
         '                c2 += __shfl_xor_sync(0xFFFFFFFF, c2, offset, WARP_SIZE);\n'
         '                c3 += __shfl_xor_sync(0xFFFFFFFF, c3, offset, WARP_SIZE);\n'
         '            }\n'
         '            sum = (c0 + c2) + (c1 + c3);\n'
         '        } else {\n'
         '        sum = xi.x + xi.y + xi.z + xi.w;\n')
    edit(q,
         '        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {\n'
         '            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);\n'
         '        }\n',
         '        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {\n'
         '            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);\n'
         '        }\n'
         '        }\n')
    edit(q,
         '                y[ib].d4[iqs/32]  = d;\n',
         '                y[ib].d4[iqs/32]  = (ggml_probe_quant_flags & 1) ? __half2float(__float2half(d)) : d; // I1\n')

    # I8: host copy of the flags, a code-sum argument on the MMQ quantizer, and the switch in
    # the (non-scatter) launcher, for Q4_K/Q5_K weights only.
    edit(q, 'static __constant__ int ggml_probe_quant_flags;\n',
         'static __constant__ int ggml_probe_quant_flags;\n'
         'static int ggml_probe_quant_host_flags = 0;\n')
    edit(q, '    CUDA_CHECK(cudaMemcpyToSymbol(ggml_probe_quant_flags, &f, sizeof(int)));\n',
         '    ggml_probe_quant_host_flags = f;\n'
         '    CUDA_CHECK(cudaMemcpyToSymbol(ggml_probe_quant_flags, &f, sizeof(int)));\n')
    edit(q, '        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {\n',
         '        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used, const int code_sum = 0) {\n')
    edit(q, '    d = 1.0f / d_inv;\n    }\n',
         '    d = 1.0f / d_inv;\n    }\n'
         '    if (code_sum && ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) { // I8: sum of codes, as MMVQ uses\n'
         '        int isum = (int) q.x + (int) q.y + (int) q.z + (int) q.w;\n'
         '#pragma unroll\n'
         '        for (int offset = 4; offset > 0; offset >>= 1) {\n'
         '            isum += __shfl_xor_sync(0xFFFFFFFF, isum, offset, WARP_SIZE);\n'
         '        }\n'
         '        sum = __half2float(__float2half(d)) * (float) isum;\n'
         '    }\n')
    text = q.read_text()
    launch = '                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);\n'
    assert text.count(launch) == 3, text.count(launch)
    q.write_text(text.replace(launch, launch.replace('/*n_expert_used=*/0);', '/*n_expert_used=*/0, code_sum);')))
    edit(q, '    switch (mmq_get_q8_1_ds_layout(type_src0)) {\n        case MMQ_Q8_1_DS_LAYOUT_D4:\n            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>\n',
         '    const int code_sum = ((ggml_probe_quant_host_flags & 64) && (type_src0 == GGML_TYPE_Q4_K || type_src0 == GGML_TYPE_Q5_K))'
         ' || (ggml_probe_quant_host_flags & 256); // I8, I10\n'
         '    switch (mmq_get_q8_1_ds_layout(type_src0)) {\n        case MMQ_Q8_1_DS_LAYOUT_D4:\n            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>\n')

    edit(q, '    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);\n',
         '    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);\n'
         '    if (ggml_probe_quant_flags & 256) { // I10: code sum, the consistent formulation\n'
         '        sum = warp_reduce_sum<QK8_1>((float) q) * __half2float(__float2half(d));\n'
         '    }\n')

    # I11: BF16 precision for the stored scale and sum (llama.cpp PR #22571 swaps these half2 fields
    # for bfloat16). Rounded through BF16, then stored in the usual FP16 field: precision, not range.
    edit(q, '    y[ib].ds = make_half2(d, sum);\n',
         '    if (ggml_probe_quant_flags & 512) { // I11\n'
         '        y[ib].ds = make_half2(__bfloat162float(__float2bfloat16(d)), __bfloat162float(__float2bfloat16(sum)));\n'
         '        return;\n'
         '    }\n'
         '    y[ib].ds = make_half2(d, sum);\n')
    edit(q, '                y[ib].ds4[iqs/32] = make_half2(d, sum);\n',
         '                y[ib].ds4[iqs/32] = (ggml_probe_quant_flags & 512) // I11\n'
         '                    ? make_half2(__bfloat162float(__float2bfloat16(d)), __bfloat162float(__float2bfloat16(sum)))\n'
         '                    : make_half2(d, sum);\n')

    edit(cu / "quantize.cuh", '#include "mmq.cuh"\n',
         '#include "mmq.cuh"\n\nvoid ggml_cuda_probe_init_quant(void); // [cs6886 probe]\n')

    # ------------------------------------------------------------------ vecdotq.cuh
    v = cu / "vecdotq.cuh"
    text = v.read_text()
    anchor = "static __device__ __forceinline__ float vec_dot_q4_0_q8_1(\n"
    assert text.count(anchor) == 1
    helpers = (
        '// [cs6886 probe] I4 helpers: offset/min correction applied the way MMQ applies it.\n'
        'template <int vdr> static __device__ __forceinline__ float probe_q4_0_codesum(\n'
        '    const int * v, const int * u, const float & d4, const half2 & ds8) {\n'
        '    int sumi = 0;\n'
        '#pragma unroll\n'
        '    for (int i = 0; i < vdr; ++i) {\n'
        '        const int vi0 = __vsubss4((v[i] >> 0) & 0x0F0F0F0F, 0x08080808);\n'
        '        const int vi1 = __vsubss4((v[i] >> 4) & 0x0F0F0F0F, 0x08080808);\n'
        '        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);\n'
        '        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);\n'
        '    }\n'
        '    return d4 * (sumi * __low2float(ds8));\n'
        '}\n\n'
        'template <int vdr> static __device__ __forceinline__ float probe_q5_0_codesum(\n'
        '    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {\n'
        '    int sumi = 0;\n'
        '#pragma unroll\n'
        '    for (int i = 0; i < vdr; ++i) {\n'
        '        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;\n'
        '        vi0    |= (vh[i] <<  4) & 0x00000010;\n'
        '        vi0    |= (vh[i] << 11) & 0x00001000;\n'
        '        vi0    |= (vh[i] << 18) & 0x00100000;\n'
        '        vi0    |= (vh[i] << 25) & 0x10000000;\n'
        '        sumi = ggml_cuda_dp4a(__vsubss4(vi0, 0x10101010), u[2*i+0], sumi);\n'
        '        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;\n'
        '        vi1    |= (vh[i] >> 12) & 0x00000010;\n'
        '        vi1    |= (vh[i] >>  5) & 0x00001000;\n'
        '        vi1    |= (vh[i] <<  2) & 0x00100000;\n'
        '        vi1    |= (vh[i] <<  9) & 0x10000000;\n'
        '        sumi = ggml_cuda_dp4a(__vsubss4(vi1, 0x10101010), u[2*i+1], sumi);\n'
        '    }\n'
        '    return d5 * (sumi * __low2float(ds8));\n'
        '}\n\n'
        '// [cs6886 probe] I7: reproduce MMQ-MMA exactly for Q4_K/Q5_K: the sub-block scale\n'
        '// products d*sc and dmin*m are formed in FP16 (mmq-load-tiles: dm*make_half2(sc, m))\n'
        '// and the min term uses the FP32 activation sum (as I4).\n'
        'static __device__ __forceinline__ float probe_q4_K_m6(\n'
        '    const int * v, const int * u, const uint8_t * sc, const uint8_t * m, const half2 & dm4,\n'
        '    const float * d8, const float * s8) {\n'
        '    float sumf = 0.0f;\n'
        '#pragma unroll\n'
        '    for (int i = 0; i < QR4_K; ++i) {\n'
        '        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;\n'
        '        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;\n'
        '        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));\n'
        '        const float2 dmi = __half22float2(dm4 * make_half2(sc[i], m[i]));\n'
        '        sumf += dmi.x * d8[i] * dot1 - dmi.y * (0.25f*s8[i]);\n'
        '    }\n'
        '    return sumf;\n'
        '}\n\n'
        'static __device__ __forceinline__ float probe_q5_K_m6(\n'
        '    const int * vl, const int * vh, const int * u, const uint8_t * sc, const uint8_t * m, const half2 & dm5,\n'
        '    const float * d8, const float * s8) {\n'
        '    float sumf = 0.0f;\n'
        '#pragma unroll\n'
        '    for (int i = 0; i < QR5_K; ++i) {\n'
        '        const int v0i = ((vl[0] >> (4*i)) & 0x0F0F0F0F) | (((vh[0] >> i) << 4) & 0x10101010);\n'
        '        const int v1i = ((vl[1] >> (4*i)) & 0x0F0F0F0F) | (((vh[1] >> i) << 4) & 0x10101010);\n'
        '        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0));\n'
        '        const float2 dmi = __half22float2(dm5 * make_half2(sc[i], m[i]));\n'
        '        sumf += dmi.x * d8[i] * dot1 - dmi.y * (0.25f*s8[i]);\n'
        '    }\n'
        '    return sumf;\n'
        '}\n\n'
    )
    text = text.replace(anchor, helpers + anchor)
    v.write_text(text)
    edit(v, '#include "common.cuh"\n',
         '#include "common.cuh"\n\n'
         '// [cs6886 probe] decode-kernel intervention flags (set in mmvq.cu; zero in every other TU)\n'
         'static __constant__ int ggml_probe_mmvq_flags;\n')
    for dm, dd, ms in (("dm4", "d4d8", "m4s8"), ("dm5", "d5d8", "m5s8")):  # I9: Q4_1 and Q5_1
        edit(v,
             '#ifdef FAST_FP16_AVAILABLE\n'
             f'    const float2 tmp = __half22float2(__hmul2({dm}, ds8));\n'
             f'    const float {dd} = tmp.x;\n'
             f'    const float {ms} = tmp.y;\n'
             '#else\n',
             '#ifdef FAST_FP16_AVAILABLE\n'
             f'    float {dd}, {ms};\n'
             '    if (ggml_probe_mmvq_flags & 128) { // I9: FP32 scale products, as MMQ\n'
             f'        const float2 {dm}f_ = __half22float2({dm});\n'
             '        const float2 ds8f_ = __half22float2(ds8);\n'
             f'        {dd} = {dm}f_.x * ds8f_.x;\n'
             f'        {ms} = {dm}f_.y * ds8f_.y;\n'
             '    } else {\n'
             f'    const float2 tmp = __half22float2(__hmul2({dm}, ds8));\n'
             f'    {dd} = tmp.x;\n'
             f'    {ms} = tmp.y;\n'
             '    }\n'
             '#else\n')

    # Q4_0 MMVQ entry: integer offset only on Turing-MMA builds, where MMQ does the same.
    edit(v,
         '    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);\n',
         '#if defined(TURING_MMA_AVAILABLE)\n'
         '    if (ggml_probe_mmvq_flags & 4) { // I4\n'
         '        return probe_q4_0_codesum<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);\n'
         '    }\n'
         '#endif\n'
         '    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);\n')
    # Q5_0 MMVQ entry: MMQ subtracts 16 from integer codes on every build.
    edit(v,
         '    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);\n',
         '    if (ggml_probe_mmvq_flags & 4) { // I4\n'
         '        return probe_q5_0_codesum<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);\n'
         '    }\n'
         '    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);\n')

    # Q4_K / Q5_K MMVQ: min term from the FP32 activation sum (ds.y), like MMQ. Each
    # q8_1 block is covered by 4 MMVQ calls (8 of its 32 values each), so each call
    # contributes a quarter of the block sum.
    for kind, impl_tail, entry_tail in (
        ("4",
         '        sumf_d += d8[i] * (dot1 * sc[i]);\n'
         '        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values\n',
         '    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);\n'),
        ("5",
         '        sumf_d += d8[i] * (dot1 * sc[i]);\n'
         '        sumf_m += d8[i] * (dot2 * m[i]);\n\n',
         '    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);\n'),
    ):
        edit(v, impl_tail,
             '        sumf_d += d8[i] * (dot1 * sc[i]);\n'
             '        if (s8 != nullptr) { // I4\n'
             '            sumf_m += (0.25f*s8[i]) * m[i];\n'
             '        } else {\n'
             '        sumf_m += d8[i] * (dot2 * m[i]);\n'
             '        }\n')
        sig_old = (f'    const uint8_t * __restrict__ m, const half2 & dm{kind}, const float * __restrict__ d8) {{\n')
        text = v.read_text()
        # the _vmmq signature is the first occurrence of this parameter line
        i = text.index(f'float vec_dot_q{kind}_K_q8_1_impl_vmmq(')
        j = text.index(sig_old, i)
        text = text[:j] + (f'    const uint8_t * __restrict__ m, const half2 & dm{kind}, const float * __restrict__ d8,\n'
                           f'    const float * __restrict__ s8 = nullptr) {{\n') + text[j + len(sig_old):]
        v.write_text(text)
        args7 = 'v, u, sc, m, bq4_K->dm, d8, s8' if kind == '4' else 'vl, vh, u, sc, m, bq5_K->dm, d8, s8'
        edit(v, entry_tail,
             '#if defined(TURING_MMA_AVAILABLE)\n'
             '    if (ggml_probe_mmvq_flags & 32) { // I7\n'
             '        float s8[2];\n'
             '        for (int i = 0; i < 2; ++i) { s8[i] = __high2float((bq8_1 + bq8_offset + i)->ds); }\n'
             f'        return probe_q{kind}_K_m6({args7});\n'
             '    }\n'
             '#endif\n'
             '    if (ggml_probe_mmvq_flags & 4) { // I4\n'
             '        float s8[2];\n'
             f'        for (int i = 0; i < 2; ++i) {{ s8[i] = __high2float((bq8_1 + bq8_offset + i)->ds); }}\n'
             + entry_tail.replace('d8);', 'd8, s8);').replace('    return', '        return') +
             '    }\n' + entry_tail)

    # ------------------------------------------------------------------ mmvq.cu / .cuh
    edit(cu / "mmvq.cuh", '#define MMVQ_MAX_BATCH_SIZE 8',
         'void ggml_cuda_probe_init_mmvq(void); // [cs6886 probe]\n\n#define MMVQ_MAX_BATCH_SIZE 8')
    m = cu / "mmvq.cu"
    edit(m, 'bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {\n',
         '#include <cstdlib>\n'
         'static int ggml_probe_host_flags = 0; // [cs6886 probe] host copy, for dispatch interventions\n'
         'void ggml_cuda_probe_init_mmvq(void) {\n'
         '    const char * e = getenv("GGML_PROBE_FLAGS");\n'
         '    const int f = e ? atoi(e) : 0;\n'
         '    ggml_probe_host_flags = f;\n'
         '    CUDA_CHECK(cudaMemcpyToSymbol(ggml_probe_mmvq_flags, &f, sizeof(int)));\n'
         '}\n\n'
         'bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {\n'
         '    if (ggml_probe_host_flags & 16) { // I6: never MMVQ, so decode and prompt share MMQ\n'
         '        return false;\n'
         '    }\n')

    # ------------------------------------------------------------------ backend init
    edit(cu / "ggml-cuda.cu",
         '    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);\n'
         '    if (ctx == nullptr) {\n'
         '        GGML_LOG_ERROR("%s: failed to allocate context\\n", __func__);\n'
         '        return nullptr;\n'
         '    }\n',
         '    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);\n'
         '    if (ctx == nullptr) {\n'
         '        GGML_LOG_ERROR("%s: failed to allocate context\\n", __func__);\n'
         '        return nullptr;\n'
         '    }\n'
         '    { // [cs6886 probe] set intervention flags before any graph capture\n'
         '        ggml_cuda_set_device(device);\n'
         '        ggml_cuda_probe_init_quant();\n'
         '        ggml_cuda_probe_init_mmvq();\n'
         '        const char * e = getenv("GGML_PROBE_FLAGS");\n'
         '        GGML_LOG_INFO("[cs6886 probe] GGML_PROBE_FLAGS=%s on device %d\\n", e ? e : "0", device);\n'
         '    }\n')
    print("patched OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
