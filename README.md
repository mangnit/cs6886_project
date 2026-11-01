# Why quantized llama.cpp diverges between cache-hit and recompute paths

A server with prefix caching (llama-server's `cache_prompt`) reuses the stored keys and values of earlier
tokens instead of recomputing them. That should not change the answer, but it does. And it changes it far
more often when the weights are quantized.

**The prior result.** Patodiya (arXiv 2609.04748) measured this on llama.cpp tag **b10434**:
- with F16 weights, 37.4% of agentic episodes diverged; with Q4_K_M weights, 82.5% did;
- on the same four histories, routing the 4-bit weights through 16-bit kernels gave 33.4% (F16: 37.5%),
  placing the amplification mainly in the quantized execution path;
- the authors left open which part of that path is responsible.

**This repository explains why 4- and 5-bit formats diverge more than 8-bit ones, and removes that excess.**
- **The two kernels.** llama.cpp's CUDA backend runs a one-token kernel (MMVQ) for ≤ 8 tokens, and a batch
  kernel (MMQ) above that. So reused cache entries come from the first, and recomputed ones from the second.
- **The mismatch.** For every format with a block offset or minimum (Q4_0, Q5_0, Q4_1, Q5_1, Q4_K, Q5_K), the
  two kernels compute the correction term with **different formulas**. One forms it from the sum of the 8-bit
  activation codes; the other from the FP32 sum of the original activations.
- **Why it amplifies.** The FP32-sum formula is the less accurate one, and it passes tiny upstream
  differences through the 8-bit activation rounding more strongly. It has a larger gain per code flip, and
  its FP16-rounded sum can jump even when no code flips. So the kernels must agree on the code-sum formula;
  merely agreeing is not enough.
- **The fix.** One runtime switch (`GGML_PROBE_FLAGS=384`, called **G**) makes both kernels use the code sum
  (I10). It also makes MMVQ form the Q4_1/Q5_1 scale products in FP32 (I9).
  - **Results:** it brings every 4- and 5-bit format down to the 8-bit (Q8_0) divergence level, with no
    decode cost above a few percent. This holds on three models on a Turing T4, and on a second GPU
    generation (Pascal sm_61, one model).
  - **Not our invention:** the code sum is the Q8_1 format's own definition. Concurrent public PRs adopted
    it for accuracy, one of them in both kernels (see the last section). What is ours is the link to cache
    divergence, the M8 finding, and the measurements.

## Results at a glance

Every number below is printed by the scripts in step 2, from the raw data in `results/`. All are on a Kaggle
Tesla T4 unless marked otherwise.

| What | Upstream b10434 | With the fix G |
|---|---|---|
| End-to-end divergence (median max\|Δlogit\|, decode-built vs recomputed cache), 7 quantizations × 3 models, ratio to Q8_0 | 1.51–2.69× | **0.97–1.08×** |
| Same, Pascal sm_61 (MX330): Q4_0, Q4_1, Q5_1, whose two kernels have no systematic mismatch there (median operator mismatch 2e-7) | 2.36–3.14× (floor: Q6_K) | **0.98–1.02×** |
| Perplexity, decode path − batch path, Qwen2.5-1.5B Q4_0 | +1.16% (z 4.42) | −0.06% |
| Decode time per token, Q4_0 / Q5_0 / Q4_1 (5 ABBA repeats) | — | +0.9% / −1.2% / +1.1%; interquartile ranges overlap upstream's |
| Decode time, for comparison: aligning the other way (switch I4) | — | +3.8% to +8.6% |
| llama-server episodes, cache on vs off (80 MT-Bench episodes × 4 turns; Qwen2.5-7B, Qwen2.5-3B, Llama-3.2-3B): episodes that diverged only upstream vs only under G | — | 39 vs 9, exact McNemar p = 1.5e-5; per question, G better on 24 and worse on 3 (sign test p = 4.9e-5) |

**What the fix does not remove:** the larger step from 16-bit to 8-bit. Even Q8_0 diverges, because 8-bit
activation rounding amplifies a difference between the attention kernels, which no matmul change touches.

## Repository layout

| Path | Contents |
|---|---|
| `probes/apply_probe_patch.py` | Patches llama.cpp b10434 with the runtime switches I1–I11 (`GGML_PROBE_FLAGS`; 0 = upstream behaviour) |
| `probes/probe_interventions_b10434.patch` | The same change as a unified diff, for reading |
| `probes/CMakeLists.txt`, `probes/*.cpp` | Builds the patched llama.cpp and seven probe programs (below) |
| `probes/kaggle_job.py` | Every experiment plan; runs on Kaggle or locally (`--local`) |
| `probes/score_*.py`, `probes/summarize.py` | Score the raw data of each experiment against its pass/fail thresholds |
| `probes/make_report.py` | Regenerates `docs/figures/` and `docs/tables_generated.md` from `results/` |
| `runners/kaggle_v2.py` | Uploads the binaries to Kaggle, pushes a job, fetches its output into `results/` |
| `runners/pascal_sm61.sh` | The Pascal (sm_61) run, end to end |
| `third_party/README.md` | How to download llama.cpp b10434 (not committed) |
| `results/` | Raw output of every run (JSONL, environment record `env.json`, logs). Never edited by hand. |
| `docs/` | Generated figures and tables |

**The probes:**

| Program | Measures |
|---|---|
| `matmul_paths` | One weight matrix: each token alone (→ MMVQ) vs inside a batch of 64 (→ MMQ). δ is the relative difference; each path's error is also taken against an FP64 reference. |
| `gate_probe` | How each kernel responds to a tiny input perturbation |
| `kv_paths` | A real model: logits after a decode-built cache vs a full recompute (max\|Δlogit\|, argmax flips) |
| `ppl_paths` | wikitext-2 perplexity through the batch path vs the decode path |
| `kv_bisect` | Every intermediate tensor of one token, decode vs recompute |
| `real_replay` | Real matmuls captured from a forward pass, replayed through both kernels vs FP64 |
| `quantize_pure` | Quantizes an F16 model to one format for every tensor (or the stock Q4_K_M mix) |

## Requirements

| To | You need |
|---|---|
| Reproduce every result from the raw data (step 2) | Python 3 (tested with 3.12) and matplotlib. No GPU. |
| Build and run the probes (steps 3–4) | Linux, an NVIDIA GPU of compute capability ≥ 6.1 (on sm_60 llama.cpp takes a different, cuBLAS path), CUDA toolkit 12.6 (`nvcc`), CMake ≥ 3.18, a C++17 compiler, `curl`, `wget`. Tested with nvcc 12.6.20, CMake 4.3.1 and GCC 9.4 on Ubuntu 20.04 (WSL2). |
| Re-run the experiments on Kaggle | A Kaggle account with GPU access and the `kaggle` CLI, authenticated (tested with 2.2.4) |

## 1. Check out

```bash
git clone https://github.com/mangnit/cs6886_project.git
cd cs6886_project
python -m pip install -r requirements.txt
```

## 2. Reproduce every result from the raw data (no GPU, under a minute)

Each scorer reads only the raw JSONL in `results/`, applies the experiment's thresholds and prints a verdict.

```bash
python probes/summarize.py  results/kaggle-night1/results results/kaggle-night3/results results/kaggle-i8/results
python probes/score_4c.py   results/kaggle-explain-q3b/results results/kaggle-explain-l3b/results
python probes/score_4d.py   results/kaggle-final/results
python probes/score_4f.py   results/kaggle-close-q15/results results/kaggle-close-q3b/results results/kaggle-close-l3b/results
python probes/score_4g.py   results/kaggle-ep-q7b/results results/kaggle-ep-q3b/results results/kaggle-ep-l3b/results results/kaggle-costbf/results
python probes/score_4i.py   results/mx330-sm61-20261029
python probes/make_report.py   # rewrites docs/figures/*.pdf|png and docs/tables_generated.md
```

What each one should print:

| Script | Experiment | Expected verdicts |
|---|---|---|
| `summarize.py` | The operator decomposition, the M6 test and I8 (switches I1–I8) on Qwen2.5-1.5B | Tables: per-format δ and error vs FP64 per switch, end-to-end divergence, perplexity |
| `score_4c.py` | Do activation outliers explain why the perplexity gap appears on Qwen2.5-3B but not on Llama-3.2-3B? | All four criteria fail: hypothesis rejected, so the model dependence is reported as unexplained |
| `score_4d.py` | I9 (FP32 scale products) and I10 (global code sum); the M8 test | C1–C4 pass. Q4_1 end to end: 0.814 upstream → 0.788 with I9 alone (systematic mismatch gone) → 0.349 with I9 + I10. |
| `score_4f.py` | The fix G on Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-3B | F1, F2, F4 pass. F3 fails on 1 of 6 entries: Qwen2.5-3B Q4_1's perplexity gap reverses sign under G (z −2.27). |
| `score_4g.py` | llama-server episodes; the cost of G (ABBA timing); BF16 precision (I11) | H1 partial, H2, H3, H5 pass; K1–K3 pass; BF1 passes, BF2 fails |
| `score_4i.py` | Pascal sm_61, numerics only | P1–P4 pass, P5 partial |
| `make_report.py` | Figures and tables (T4 data) | Ends with `cross-run mismatches: 0`, i.e. runs that measured the same thing agree bit for bit |

## 3. Build the patched llama.cpp and the probes (NVIDIA GPU)

Download b10434 and apply the switches (see `third_party/README.md`):

```bash
cd third_party
curl -L -o b10434.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10434.tar.gz
tar xzf b10434.tar.gz
python ../probes/apply_probe_patch.py llama.cpp-b10434
cd ..
```

Build for your GPU's architecture: 75 = Turing T4, 61 = Pascal GTX 10xx / MX330, 86 = Ampere RTX 30xx, 89 = Ada.
The sm_61 build took 48 minutes on a 4-core laptop.

```bash
cmake -S probes -B build/probes-sm75 -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build/probes-sm75 --target matmul_paths gate_probe kv_paths ppl_paths kv_bisect real_replay quantize_pure -j4
```

A one-minute check, the same matmul with the fix off and on:

```bash
mkdir -p work
for f in 0 384; do GGML_PROBE_FLAGS=$f build/probes-sm75/matmul_paths --K 1536 --N 1536 --B 64 --tokens 16 --out work/op.jsonl; done
python -c "import json; [print(r['probe_flags'], r['type'], '%.2e' % r['delta_median'], '%.3e' % r['err_alone_vs_fp64_median'], '%.3e' % r['err_batch_vs_fp64_median']) for r in map(json.loads, open('work/op.jsonl')) if r['type'] in ('q8_0', 'q4_0', 'q4_K')]"
```

On a T4 you should see:
- **Q4_0:** δ 2.0e-2 at flags 0, with errors vs FP64 (decode / batch) 2.280e-2 / 1.089e-2. Under 384, δ is
  4.87e-4 with errors 1.087e-2 / 1.089e-2.
- **Q4_K:** δ 1.14e-2 at flags 0; 7.65e-4 under 384.
- **Q8_0:** δ 2.1e-4 either way. It has no correction term, so the fix leaves it alone.

The raw rows to compare against are in `results/kaggle-final/results/matmul_paths.jsonl` (T4) and
`results/mx330-sm61-20261029/operator/matmul_paths.jsonl` (sm_61).

## 4. Re-run the experiments

### The switches

`GGML_PROBE_FLAGS` is a bitmask; 0 means upstream behaviour.

| Value | Name | What changes | Removes |
|---|---|---|---|
| 1 | I1 | MMQ rounds its FP32 activation scale through FP16, as MMVQ does | M1, the scale-precision split (Q8_0, Q5_0, Q3_K, Q6_K) |
| 2 | I2 | MMQ quantizes activations with MMVQ's rounding formula | M2, rare one-step code differences |
| 4 | I4 | MMVQ applies the offset/min correction the way MMQ does (for K-quants that is the *worse* FP32-sum formula) | M4, the formula mismatch |
| 8 | I5 | MMQ sums each activation block in MMVQ's order | M4b, occasional FP16 rounding straddles of the sum |
| 16 | I6 | Dense matmuls never use MMVQ | the dispatch split itself |
| 32 | I7 | MMVQ reproduces MMQ's FP16 K-quant sub-scale products (tensor-core GPUs) | M6, FP16 sub-scale products in tensor-core MMQ |
| 64 | I8 | MMQ uses the activation code sum for Q4_K/Q5_K | M4 for K-quants, toward the accurate formula |
| 128 | I9 | MMVQ forms Q4_1/Q5_1 scale products in FP32 instead of FP16 | M7, FP16 scale products in MMVQ (GPUs with fast FP16; not sm_61) |
| 256 | I10 | Both kernels use the activation code sum for every offset/min format | M4 everywhere, toward the accurate formula |
| 512 | I11 | Rounds the stored activation scale and sum through BF16 (a what-if) | — |
| **384** | **G = I9 + I10** | **the recommended fix** | |

### The experiments

Each Kaggle job is `probes/kaggle_job.py` with the arguments shown.
- **Model:** without `--src-url` it downloads Qwen2.5-1.5B-Instruct F16. `--src-url` takes the official
  F16 GGUF(s), comma-separated if split. We used:
  - `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-fp16-00001-of-00002.gguf,https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-fp16-00002-of-00002.gguf`
    (Qwen2.5-3B, tag prefix `Q3B_`);
  - `https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf`
    (Llama-3.2-3B, `L3B_`).
- **Text and questions:** the job downloads wikitext-2 for perplexity, and the MT-Bench questions for the
  episodes.

| Result folder | Job arguments | What it tests | Read by |
|---|---|---|---|
| `kaggle-night1` | `--plan night1` | Operator decomposition (I1–I6), gate, end to end, perplexity | `summarize.py`, `make_report.py` |
| `kaggle-night3` | `--plan night2` | The M6 test (I7) | `summarize.py`, `make_report.py` |
| `kaggle-bisect` | `--plan bisect` | Per-tensor bisection of one token (finds where divergence enters: attention) | raw data only |
| `kaggle-i8` | `--plan i8` | I8: aligning K-quants on the code sum | `summarize.py`, `make_report.py` |
| `kaggle-gen-q3b`, `kaggle-gen-l3b` | `--plan gen --src-url <3B> --tag-prefix Q3B_` (or `L3B_`) | The perplexity gap on two 3B models | `make_report.py` |
| `kaggle-explain-q3b`, `-l3b` | `--plan explain --src-url <3B> --tag-prefix Q3B_` (or `L3B_`) | Activation-outlier hypothesis | `score_4c.py` |
| `kaggle-final` | `--plan final` | I9, I10 and the M8 test, prefill cost | `score_4d.py`, `make_report.py` |
| `kaggle-close-q15`, `-q3b`, `-l3b` | `--plan close --tag-prefix Q15_` (3B: `--src-url <3B> --tag-prefix Q3B_` / `L3B_`) | The fix G on three models | `score_4f.py`, `make_report.py` |
| `kaggle-ep-q7b` | `--plan episodes7b --tag-prefix Q7B_` | llama-server cache on vs off, Qwen2.5-7B (official Q8_0, Q4_K_M and Q4_0 GGUFs) | `score_4g.py`, `make_report.py` |
| `kaggle-ep-q3b`, `-l3b` | `--plan episodes --src-url <3B> --tag-prefix Q3B_ --ep-repeat` (Llama: `L3B_`, no `--ep-repeat`) | The same on the 3B models, plus a repeat of one pass | `score_4g.py`, `make_report.py` |
| `kaggle-costbf` | `--plan costbf` | Decode/prefill cost of G vs I4 (ABBA), and I11 | `score_4g.py` |
| `kaggle-p100` | `--plan p100` | Requested a P100, but Kaggle ran it on a T4; used only as a cross-run determinism check | raw data only |
| `mx330-sm61-20261029` | `runners/pascal_sm61.sh` | Pascal sm_61 (GeForce MX330, 2 GB), numerics only | `score_4i.py` |

**Locally,** with a build from step 3:

```bash
python probes/kaggle_job.py --local --plan close --tag-prefix Q15_ --bin build/probes-sm75 --out work/close-q15
python probes/score_4f.py work/close-q15
```

On a 2 GB GPU the F16 model does not fit. Add `--no-f16` to use it only as the quantization source, as
`runners/pascal_sm61.sh` does.

**On Kaggle (T4).** Build one bundle for sm_60 and sm_75 with shared libraries, upload it as a private
dataset, then push jobs against it:

```bash
cmake -S probes -B build/probes-kaggle -DCMAKE_CUDA_ARCHITECTURES="60;75" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/probes-kaggle --target matmul_paths gate_probe kv_paths ppl_paths kv_bisect real_replay quantize_pure llama-server -j4
python runners/kaggle_v2.py stage  --build build/probes-kaggle --dataset cs6886-probes-1
python runners/kaggle_v2.py push   --name close-q15 --dataset cs6886-probes-1 --job-args "--plan close --tag-prefix Q15_"
python runners/kaggle_v2.py status --name close-q15
python runners/kaggle_v2.py fetch  --name close-q15        # -> results/kaggle-close-q15/
```

Use a new `--dataset` name for every new bundle: a re-versioned Kaggle dataset can mount the old files.
Each job records the SHA-256 of the binaries it actually ran in `env.json`, so this is checkable.

**On Pascal (sm_61):**

```bash
WORK=/path/with/space bash runners/pascal_sm61.sh
python probes/score_4i.py /path/with/space/runs/pascal
```

## Notes on reproducibility

- **Determinism.** Every probe is deterministic: repeated runs gave bit-identical results across Kaggle
  accounts, sessions and physical T4s.
- **The patch at flags 0 runs the original code by construction:** every behavioural edit sits behind its
  flag bit. Results at flags 0 are bit-identical across our four successive patch versions. We did not
  compare against an unpatched build.
- **The 8-token switch.** From 9 tokens up, outputs are bit-identical to the 64-token batch. Batches of 1–8
  differ from each other only by FP32 rounding, because MMVQ's warp count varies with the batch size.
- **A different build.** With another CUDA version or driver, expect the last bits to change. The ratios and
  verdicts above are the claims, not the bits.
- **Pinned libraries on Kaggle.** Jobs pip-install cuBLAS 12.6.0.22 and CUDA runtime 12.6.37, matching the
  nvcc 12.6 build. Local runs use the toolkit's own libraries.
- **Timing comes only from the T4.** The MX330 is a power-limited laptop GPU, so its timings are not used.
- **The Pascal run was done in stages,** because of interruptions. `runners/pascal_sm61.sh` runs the same
  steps with the same settings in one go. It also runs Q5_0's perplexity under G, which the published run
  lost to a restart.

## What is new here, and what is not

Based on our search of the literature, and of the issues and pull requests of llama.cpp and related
projects, up to 1 November 2026.

**New, to our knowledge:**
1. **The specific link to caching.** In MMVQ and MMQ, the correction term is computed with different
   formulas, per format and GPU (M4). We trace what this does to cache-hit vs recompute divergence: at the
   operator, on real data, end to end, in perplexity and in episodes. None of the sources below explains
   decode-vs-prompt divergence through the correction formula.
2. **M8.** The formula sets how strongly 8-bit activation rounding amplifies small differences, so the
   kernels must agree on the *accurate* formula. Removing the mismatch alone barely helps, and aligning the
   obvious way can make things worse.
3. **The measurements.** The effect of the code sum plus I9 on that divergence, on two GPU generations; the
   decode-vs-batch perplexity gap and its model dependence; and the cost. I9 removes the last mismatch (M7);
   we did not test whether it matters once the code sum is in.

**Not ours:**
- **The cache-hit vs recompute divergence itself:** Patodiya, arXiv 2609.04748.
- **The code sum itself.** It is the Q8_1 format's own definition (`ggml-common.h`), and it was adopted
  concurrently for accuracy by:
  - llama.cpp [PR #31247](https://github.com/ggml-org/llama.cpp/pull/31247) (decode quantizer only; open);
  - sglang [PR #46428](https://github.com/sgl-project/sglang/pull/46428);
  - Brontir/NInferno [PR #91](https://github.com/Brontir/NInferno/pull/91), which covers both kernels and was
    merged on 24 October 2026.
- **FP32 scale products for Q4_1/Q5_1 (I9's change):** llama.cpp
  [PR #31019](https://github.com/ggml-org/llama.cpp/pull/31019) (closed unmerged) and NInferno
  [PR #218](https://github.com/Brontir/NInferno/pull/218) (merged 29 October 2026).
- **M7 itself,** already public as llama.cpp issue #30950 (as an overflow).
- **The likely cause of the attention-kernel difference** (FP32 vs FP16 query), found by a llama.cpp
  contributor in [issue #25618](https://github.com/ggml-org/llama.cpp/issues/25618#issuecomment-6441719728).
  We did not test that fix. The same thread notes, in general, that batch non-invariance can flip tokens
  under prefix caching.
- **The overflow of the FP16 activation-sum field** (llama.cpp #21652, #22571).

**Limitations:**
- **Untested:** the prior work's RTX 4090; the control for M8 with identical quantizers and formula B; a
  comparison against an unpatched build.
- **A different protocol:** our episodes (MT-Bench, 4 requests) differ from the prior work's (BFCL, about 10).
- **Unexplained:** on the T4, splitting a prompt into two batches already changes logits (Q8_0: 0.35).
