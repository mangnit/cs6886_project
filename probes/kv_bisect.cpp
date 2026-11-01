// Op-by-op bisection of the decode path against the recompute path for ONE token.
//
//   A  prompt P (np tokens) as one batch, then the next token t decoded ALONE (decode path)
//   B  P + t as one batch (prompt path)
// Every earlier token's KV comes from a prompt batch in both runs (MMQ is batch-invariant,
// measured), so the only difference is how token t itself is computed. Every named F32
// activation produced for token t is captured with llama.cpp's eval callback and compared,
// node by node in A's execution order: relative L2 difference and bit-equality.
//
// Observation can perturb what it observes (forced graph splits can disable kernel fusion).
// Each run is therefore also done WITHOUT the callback, and the final logits of the observed
// and unobserved runs are compared bitwise; `observer_changed_logits` must be false for the
// bisection to describe the real execution.
//
//   kv_bisect --model M.gguf [--filter l_out] [--prompt 0] [--out file]
// --filter: only capture nodes whose name contains this substring ("" = all nodes).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const char * PROMPTS[] = {
    "The Industrial Revolution began in Great Britain in the late eighteenth century and spread to Western Europe and North America. "
    "It replaced hand production with machines, introduced new chemical and iron production processes, and increased the use of steam power.",
    "def merge_sort(xs):\n    if len(xs) <= 1:\n        return xs\n    mid = len(xs) // 2\n    left = merge_sort(xs[:mid])\n    right = merge_sort(xs[mid:])\n",
    "A patient presents with a three-day history of fever, productive cough and pleuritic chest pain. On examination the respiratory rate is 24 per minute.",
};

struct Capture {
    bool active = false;
    std::string filter;
    std::vector<std::string> order;
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, std::vector<int64_t>> ne;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    Capture * c = (Capture *) ud;
    if (!c->active || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return false;
    if (!c->filter.empty() && strstr(t->name, c->filter.c_str()) == nullptr) return false;
    if (ask) return true;
    std::string name = t->name;
    if (name.empty() || c->data.count(name)) return true;  // keep the first node of a given name
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    c->data[name] = std::move(v);
    c->ne[name] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    c->order.push_back(name);
    return true;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & toks, int p0) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) { b.token[i] = toks[i]; b.pos[i] = p0 + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i + 1 == toks.size(); }
    b.n_tokens = (int) toks.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(4); }
    llama_batch_free(b);
}

// Slice token index `ti` out of a tensor captured from a batch of T tokens. The token
// dimension is the one that is T in B and 1 in A, with all other dimensions equal.
static bool slice(const std::vector<float> & b, const std::vector<int64_t> & neb, const std::vector<int64_t> & nea,
                  int64_t T, int64_t ti, std::vector<float> & out) {
    int dim = -1;
    for (int d = 0; d < 4; ++d) {
        if (neb[d] == T && nea[d] == 1) {
            bool rest = true;
            for (int e = 0; e < 4; ++e) if (e != d && neb[e] != nea[e]) rest = false;
            if (rest) { dim = d; break; }
        }
    }
    if (dim < 0) return false;
    int64_t inner = 1; for (int d = 0; d < dim; ++d) inner *= neb[d];
    int64_t outer = 1; for (int d = dim + 1; d < 4; ++d) outer *= neb[d];
    out.resize((size_t) (inner*outer));
    for (int64_t o = 0; o < outer; ++o)
        memcpy(out.data() + o*inner, b.data() + (o*T + ti)*inner, (size_t) inner*sizeof(float));
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, out = "-", filter = "l_out", tag;
    int pi = 0, np_max = 96;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--model") model_path = argv[++i];
        else if (k == "--out") out = argv[++i];
        else if (k == "--filter") filter = argv[++i];
        else if (k == "--prompt") pi = atoi(argv[++i]);
        else if (k == "--tag") tag = argv[++i];
        else if (k == "--P") np_max = atoi(argv[++i]);
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(512);
    toks.resize(llama_tokenize(vocab, PROMPTS[pi], (int) strlen(PROMPTS[pi]), toks.data(), (int) toks.size(), true, false));
    if ((int) toks.size() > np_max + 1) toks.resize(np_max + 1);
    const std::vector<llama_token> prompt(toks.begin(), toks.end() - 1);
    const llama_token last = toks.back();
    const int np = (int) prompt.size(), T = np + 1;

    Capture cap; cap.filter = filter;
    auto make_ctx = [&](bool observe) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.no_perf = true;
        if (observe) { cp.cb_eval = cb_eval; cp.cb_eval_user_data = &cap; }
        return llama_init_from_model(model, cp);
    };
    auto logits = [&](llama_context * ctx) { const float * l = llama_get_logits_ith(ctx, -1); return std::vector<float>(l, l + nv); };

    // Unobserved references.
    llama_context * c0 = make_ctx(false);
    decode(c0, prompt, 0); decode(c0, {last}, np); const auto la0 = logits(c0);
    llama_memory_clear(llama_get_memory(c0), true);
    decode(c0, toks, 0); const auto lb0 = logits(c0);
    llama_free(c0);

    // Observed runs: capture only while token `last` is being computed.
    llama_context * c1 = make_ctx(true);
    decode(c1, prompt, 0);
    cap.active = true; decode(c1, {last}, np); cap.active = false;
    const auto la1 = logits(c1);
    Capture A = cap; cap = Capture(); cap.filter = filter;
    llama_memory_clear(llama_get_memory(c1), true);
    cap.active = true; decode(c1, toks, 0); cap.active = false;
    const auto lb1 = logits(c1);
    Capture B = cap;
    llama_free(c1);

    const bool heis_a = memcmp(la0.data(), la1.data(), nv*sizeof(float)) != 0;
    const bool heis_b = memcmp(lb0.data(), lb1.data(), nv*sizeof(float)) != 0;
    double lnum = 0, lden = 0, lmax = 0;
    for (int i = 0; i < nv; ++i) { const double d = (double) la0[i] - lb0[i]; lnum += d*d; lden += (double) lb0[i]*lb0[i]; lmax = std::max(lmax, std::fabs(d)); }

    const char * flags = getenv("GGML_PROBE_FLAGS");
    FILE * fo = out == "-" ? stdout : fopen(out.c_str(), "a");
    int compared = 0;
    for (const std::string & name : A.order) {
        if (!B.data.count(name)) continue;
        std::vector<float> bs;
        if (!slice(B.data[name], B.ne[name], A.ne[name], T, T - 1, bs) || bs.size() != A.data[name].size()) continue;
        const std::vector<float> & a = A.data[name];
        double num = 0, den = 0; bool eq = true;
        for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - bs[i]; num += d*d; den += (double) bs[i]*bs[i]; eq = eq && a[i] == bs[i]; }
        fprintf(fo, "{\"probe\":\"kv_bisect\",\"tag\":\"%s\",\"probe_flags\":\"%s\",\"prompt\":%d,\"node\":\"%s\",\"order\":%d,\"n\":%zu,"
                    "\"rel_l2\":%.4e,\"bitwise_equal\":%s}\n",
                tag.c_str(), flags ? flags : "0", pi, name.c_str(), compared, a.size(), std::sqrt(num/std::max(den, 1e-300)), eq ? "true" : "false");
        ++compared;
    }
    fprintf(fo, "{\"probe\":\"kv_bisect_summary\",\"tag\":\"%s\",\"probe_flags\":\"%s\",\"prompt\":%d,\"np\":%d,\"filter\":\"%s\","
                "\"nodes_compared\":%d,\"observer_changed_logits_A\":%s,\"observer_changed_logits_B\":%s,"
                "\"final_logits_rel_l2\":%.4e,\"final_logits_maxabs\":%.4e}\n",
            tag.c_str(), flags ? flags : "0", pi, np, filter.c_str(), compared, heis_a ? "true" : "false", heis_b ? "true" : "false",
            std::sqrt(lnum/std::max(lden, 1e-300)), lmax);
    if (fo != stdout) fclose(fo);
    llama_model_free(model);
    return 0;
}
