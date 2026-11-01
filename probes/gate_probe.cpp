// Transfer function of a matmul path for tiny input perturbations (mechanism M5).
//
// For activations X and a perturbed copy X' = X * (1 + eps * u), u uniform in {-1, +1},
// both are pushed through the SAME kernel path and the outputs compared:
//   path "alone": ne11 = 1   (MMVQ for quantized weights, MMVF for F16)
//   path "batch": ne11 = 64  (MMQ for quantized weights on sm_61/Turing+)
// A smooth path gives an output change proportional to eps and never bit-identical
// outputs. A path that quantizes its activations (Q8_1) is expected to give mostly
// bit-identical outputs with rare jumps of about one quantization step: a gate.
// Reported per (type, path, eps): fraction of tokens with bit-identical output, and the
// median and max relative output change over tokens.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> run_mm(ggml_backend_t be, ggml_type type, const std::vector<uint8_t> & wq,
                                 int K, int N, const float * x, int nb) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, nb);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(W, wq.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(X, x, 0, ggml_nbytes(X));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> y((size_t) N*nb);
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

int main(int argc, char ** argv) {
    std::vector<std::string> types = {"f16", "q8_0", "q4_0", "q4_K", "q6_K"};
    int K = 1536, N = 1536, B = 64;
    std::string out = "-";
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--K") K = atoi(argv[++i]);
        else if (k == "--N") N = atoi(argv[++i]);
        else if (k == "--out") out = argv[++i];
        else if (k == "--types") {
            types.clear(); std::string s = argv[++i]; size_t p = 0;
            while (p != std::string::npos) { size_t q = s.find(',', p); types.push_back(s.substr(p, q == std::string::npos ? q : q - p)); p = q == std::string::npos ? q : q + 1; }
        }
    }
    FILE * fo = out == "-" ? stdout : fopen(out.c_str(), "a");
    ggml_backend_t be = ggml_backend_cuda_init(0);
    char desc[256]; ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    const char * flags = getenv("GGML_PROBE_FLAGS");

    std::mt19937 rng(6886);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<float> X((size_t) K*B), W((size_t) N*K);
    std::vector<int> outlier(K);
    for (int k = 0; k < K; ++k) outlier[k] = u01(rng) < 0.01f;
    for (int j = 0; j < B; ++j) { const float s = 0.5f + 1.5f*u01(rng); for (int k = 0; k < K; ++k) X[(size_t) j*K + k] = s*g(rng)*(outlier[k] ? 20.0f : 1.0f); }
    for (auto & w : W) w = 0.02f*g(rng);

    for (const std::string & ts : types) {
        ggml_type type = GGML_TYPE_COUNT;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) if (ggml_type_name((ggml_type) t) && ts == ggml_type_name((ggml_type) t)) type = (ggml_type) t;
        if (type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", ts.c_str()); continue; }
        std::vector<uint8_t> wq(ggml_row_size(type, K)*N);
        if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(W.data(), (ggml_fp16_t *) wq.data(), (int64_t) N*K);
        else ggml_quantize_chunk(type, W.data(), wq.data(), 0, N, K, nullptr);

        for (double eps : {1e-7, 1e-6, 1e-5, 1e-4}) {
            std::vector<float> Xp(X);
            std::mt19937 r2(1234);
            for (auto & v : Xp) v *= (float) (1.0 + eps*((r2() & 1) ? 1.0 : -1.0));
            for (int alone = 0; alone < 2; ++alone) {
                std::vector<double> d; int same = 0;
                const int ntok = alone ? 16 : B;
                std::vector<float> y0, y1;
                if (!alone) { y0 = run_mm(be, type, wq, K, N, X.data(), B); y1 = run_mm(be, type, wq, K, N, Xp.data(), B); }
                for (int j = 0; j < ntok; ++j) {
                    const float * a; const float * b; std::vector<float> ya, yb;
                    if (alone) { ya = run_mm(be, type, wq, K, N, X.data() + (size_t) j*K, 1); yb = run_mm(be, type, wq, K, N, Xp.data() + (size_t) j*K, 1); a = ya.data(); b = yb.data(); }
                    else { a = y0.data() + (size_t) j*N; b = y1.data() + (size_t) j*N; }
                    double num = 0, den = 0; bool eq = true;
                    for (int n = 0; n < N; ++n) { const double t = (double) a[n] - b[n]; num += t*t; den += (double) a[n]*a[n]; eq = eq && a[n] == b[n]; }
                    d.push_back(std::sqrt(num/std::max(den, 1e-300))); same += eq;
                }
                std::sort(d.begin(), d.end());
                fprintf(fo, "{\"probe\":\"gate\",\"llama_cpp\":\"b10434\",\"device\":\"%s\",\"cc\":%d%d,\"probe_flags\":\"%s\",\"type\":\"%s\","
                            "\"path\":\"%s\",\"eps\":%.0e,\"tokens\":%d,\"bitwise_same_frac\":%.4f,\"rel_change_median\":%.3e,\"rel_change_max\":%.3e}\n",
                        desc, prop.major, prop.minor, flags ? flags : "0", ts.c_str(), alone ? "alone" : "batch", eps, ntok,
                        (double) same/ntok, d[d.size()/2], d.back());
                fflush(fo);
            }
        }
    }
    ggml_backend_free(be);
    if (fo != stdout) fclose(fo);
    return 0;
}
