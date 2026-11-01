// Real-data replay of every quantized matmul (experiment 4c: P1-P3).
//
// 1. A batch forward of T real tokens runs with llama.cpp's eval callback, which captures each
//    quantized MUL_MAT's weight (raw blocks) and its F32 input activations [K x T].
// 2. Each captured matmul is replayed on a separate CUDA backend:
//       batch  : all T tokens at once           (prompt kernel, MMQ)
//       decode : sampled tokens one at a time   (decode kernel, MMVQ)
//    and both are compared with an FP64 reference built from the dequantized weights, on a
//    fixed random sample of output rows.
// 3. Outlier statistics of the input: the crest factor amax/rms of each 32-value block.
// 4. Counterfactual (--clip): every block is clipped to |x| <= c * rms(block) before the replay,
//    and the reference is recomputed for the clipped input. It asks: with a lighter outlier
//    structure, does the decode path still lose more accuracy than the prompt path?
//
// GGML_PROBE_FLAGS selects the intervention for both backends (read at backend init).
// Output: one JSON line per (matmul, clip level).

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Captured {
    std::string name;
    ggml_type type;
    int64_t K, N, T;
    std::vector<uint8_t> w;   // raw quantized rows
    std::vector<float> x;     // [T][K]
};

struct Cap { bool active = false; int64_t T = 0; std::vector<Captured> mm; std::map<std::string, bool> seen; };

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    Cap * c = (Cap *) ud;
    if (!c->active || t->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * w = t->src[0], * x = t->src[1];
    if (!ggml_is_quantized(w->type) || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(w)) return false;
    if (x->ne[1] != c->T || x->ne[2] != 1 || x->ne[3] != 1 || w->ne[2] != 1 || w->ne[3] != 1) return false;
    if (ask) return true;
    std::string name = w->name;
    if (c->seen.count(name)) return true;
    c->seen[name] = true;
    Captured m; m.name = name; m.type = w->type; m.K = w->ne[0]; m.N = w->ne[1]; m.T = x->ne[1];
    m.w.resize(ggml_nbytes(w)); ggml_backend_tensor_get(w, m.w.data(), 0, m.w.size());
    m.x.resize((size_t) m.K*m.T); ggml_backend_tensor_get(x, m.x.data(), 0, m.x.size()*sizeof(float));
    c->mm.push_back(std::move(m));
    return true;
}

// One weight uploaded once; two graphs (batch of T, single column) reused for every input.
struct Replayer {
    ggml_backend_t be; ggml_context * ctx; ggml_backend_buffer_t buf;
    ggml_tensor *W, *Xb, *X1, *Yb, *Y1; ggml_cgraph *gb, *g1; int64_t K, N, T;
    Replayer(ggml_backend_t be_, const Captured & m) : be(be_), K(m.K), N(m.N), T(m.T) {
        ggml_init_params ip = { ggml_tensor_overhead()*16 + 2*ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        W = ggml_new_tensor_2d(ctx, m.type, K, N);
        Xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        X1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        Yb = ggml_mul_mat(ctx, W, Xb); Y1 = ggml_mul_mat(ctx, W, X1);
        gb = ggml_new_graph(ctx); ggml_build_forward_expand(gb, Yb);
        g1 = ggml_new_graph(ctx); ggml_build_forward_expand(g1, Y1);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(W, m.w.data(), 0, ggml_nbytes(W));
    }
    ~Replayer() { ggml_backend_buffer_free(buf); ggml_free(ctx); }
    std::vector<float> batch(const std::vector<float> & x) {
        ggml_backend_tensor_set(Xb, x.data(), 0, ggml_nbytes(Xb)); ggml_backend_graph_compute(be, gb);
        std::vector<float> y((size_t) N*T); ggml_backend_tensor_get(Yb, y.data(), 0, ggml_nbytes(Yb)); return y;
    }
    std::vector<float> one(const float * x) {
        ggml_backend_tensor_set(X1, x, 0, ggml_nbytes(X1)); ggml_backend_graph_compute(be, g1);
        std::vector<float> y((size_t) N); ggml_backend_tensor_get(Y1, y.data(), 0, ggml_nbytes(Y1)); return y;
    }
};

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out; std::stringstream ss(s); std::string it;
    while (std::getline(ss, it, ',')) if (!it.empty()) out.push_back(it);
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, text_path, out = "-", tag, clips = "0";
    int T = 64, nrows = 256, ntok = 4, offset = 20000;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--model") model_path = argv[++i];
        else if (k == "--text") text_path = argv[++i];
        else if (k == "--out") out = argv[++i];
        else if (k == "--tag") tag = argv[++i];
        else if (k == "--clip") clips = argv[++i];
        else if (k == "--tokens") T = atoi(argv[++i]);
        else if (k == "--rows") nrows = atoi(argv[++i]);
        else if (k == "--tok-sample") ntok = atoi(argv[++i]);
        else if (k == "--offset") offset = atoi(argv[++i]);
    }
    std::ifstream f(text_path); std::stringstream ss; ss << f.rdbuf(); std::string text = ss.str();
    text = text.substr(std::min<size_t>(offset, text.size()), 4000);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    toks.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false));
    if ((int) toks.size() < T) { fprintf(stderr, "text too short\n"); return 2; }
    toks.resize(T);

    // 1. capture
    Cap cap; cap.T = T;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.no_perf = true;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(T, 0, 1);
    for (int i = 0; i < T; ++i) { b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == T - 1; }
    b.n_tokens = T;
    cap.active = true;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 3; }
    cap.active = false;
    llama_batch_free(b);
    llama_free(ctx);
    fprintf(stderr, "captured %zu quantized matmuls\n", cap.mm.size());

    // 2-4. replay
    ggml_backend_t be = ggml_backend_cuda_init(0);
    const char * flags = getenv("GGML_PROBE_FLAGS");
    FILE * fo = out == "-" ? stdout : fopen(out.c_str(), "a");
    std::vector<int> tok_idx;
    for (int s = 0; s < ntok; ++s) tok_idx.push_back((int) ((s + 1) * (int64_t) T / (ntok + 1)));

    for (const Captured & m : cap.mm) {
        Replayer rp(be, m);
        std::mt19937 rng(6886);
        std::vector<int64_t> rows(m.N); for (int64_t r = 0; r < m.N; ++r) rows[r] = r;
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(std::min<int64_t>(nrows, m.N));
        // dequantized sampled rows
        const size_t rs = ggml_row_size(m.type, m.K);
        std::vector<float> wd((size_t) rows.size()*m.K);
        for (size_t r = 0; r < rows.size(); ++r)
            ggml_get_type_traits(m.type)->to_float(m.w.data() + rows[r]*rs, wd.data() + r*m.K, m.K);

        for (const std::string & cs : split(clips)) {
            const double c = std::stod(cs);
            std::vector<float> x = m.x;
            std::vector<double> crest;
            int over4 = 0, nb = 0;
            for (int64_t t = 0; t < m.T; ++t)
                for (int64_t k0 = 0; k0 + 32 <= m.K; k0 += 32) {
                    float * blk = x.data() + t*m.K + k0;
                    double ss2 = 0; for (int j = 0; j < 32; ++j) ss2 += (double) blk[j]*blk[j];
                    const double rms = std::sqrt(ss2/32);
                    if (c > 0 && rms > 0) for (int j = 0; j < 32; ++j) blk[j] = (float) std::max(-c*rms, std::min(c*rms, (double) blk[j]));
                    double am = 0, s2 = 0; for (int j = 0; j < 32; ++j) { am = std::max(am, (double) std::fabs(blk[j])); s2 += (double) blk[j]*blk[j]; }
                    const double r2 = std::sqrt(s2/32);
                    if (r2 > 0) { const double cr = am/r2; crest.push_back(cr); over4 += cr > 4; }
                    ++nb;
                }
            std::sort(crest.begin(), crest.end());
            const double crest_med = crest.empty() ? 0 : crest[crest.size()/2];
            const std::vector<float> yb = rp.batch(x);
            double nd = 0, nbt = 0, nref = 0, ndelta = 0, nyb = 0;
            for (int t : tok_idx) {
                const std::vector<float> y1 = rp.one(x.data() + (size_t) t*m.K);
                const float * xt = x.data() + (size_t) t*m.K;
                for (size_t r = 0; r < rows.size(); ++r) {
                    double ref = 0; const float * w = wd.data() + r*m.K;
                    for (int64_t k = 0; k < m.K; ++k) ref += (double) w[k]*xt[k];
                    const double vb = yb[(size_t) t*m.N + rows[r]], v1 = y1[rows[r]];
                    nd += (v1 - ref)*(v1 - ref); nbt += (vb - ref)*(vb - ref); nref += ref*ref;
                    ndelta += (v1 - vb)*(v1 - vb); nyb += vb*vb;
                }
            }
            const double err_dec = std::sqrt(nd/nref), err_bat = std::sqrt(nbt/nref);
            fprintf(fo, "{\"probe\":\"real_replay\",\"tag\":\"%s\",\"probe_flags\":\"%s\",\"matmul\":\"%s\",\"type\":\"%s\",\"K\":%lld,\"N\":%lld,"
                        "\"T\":%lld,\"clip\":%g,\"crest_median\":%.4f,\"crest_frac_gt4\":%.5f,\"err_decode\":%.5e,\"err_batch\":%.5e,"
                        "\"excess\":%.5e,\"delta\":%.5e,\"rows\":%zu,\"tokens_sampled\":%zu}\n",
                    tag.c_str(), flags ? flags : "0", m.name.c_str(), ggml_type_name(m.type), (long long) m.K, (long long) m.N,
                    (long long) m.T, c, crest_med, nb ? (double) over4/nb : 0.0, err_dec, err_bat, err_dec - err_bat,
                    std::sqrt(ndelta/std::max(nyb, 1e-300)), rows.size(), tok_idx.size());
        }
        fflush(fo);
    }
    ggml_backend_free(be);
    if (fo != stdout) fclose(fo);
    llama_model_free(model);
    return 0;
}
