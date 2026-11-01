#!/usr/bin/env python3
"""Build the figures and tables from raw results (results/kaggle-*/results/*.jsonl).

    python probes/make_report.py
    -> docs/figures/fig_*.pdf + .png, docs/tables_generated.md

Nothing is typed in by hand: every plotted or tabulated number is read from a JSONL line, and each
table names its source files and bundle hash (env.json). Where two runs measured the same thing
(same type, flags and shape), the script checks that they agree bit for bit and reports any
mismatch. Figures and tables that need the closing run (experiment 4f) are skipped until
results/kaggle-close-* exist. All data are Tesla T4 (sm_75); laptop runs are never read.
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, OUT = ROOT / "results", ROOT / "docs" / "figures", ROOT / "docs" / "tables_generated.md"

UP, FIX, ALT, FLOOR = "#d95f02", "#1b9e77", "#7570b3", "#333333"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200, "savefig.bbox": "tight"})

OP_ORDER = ["f16", "q8_0", "q6_K", "q3_K", "q4_1", "q5_1", "q4_K", "q5_K", "q4_0", "q5_0", "q2_K"]
SEVEN = ["Q4_0", "Q5_0", "Q4_1", "Q5_1", "Q4_K", "Q5_K", "Q4_K_M"]
FLAG_NAMES = {"0": "upstream", "1": "I1", "2": "I2", "4": "I4", "7": "I1+I2+I4", "8": "I5", "15": "I1+I2+I4+I5",
              "16": "I6", "32": "I7", "47": "15+I7", "64": "I8", "128": "I9", "256": "I10", "384": "G=I9+I10"}
MODELS = {"Q15": "Qwen2.5-1.5B", "Q3B": "Qwen2.5-3B", "L3B": "Llama-3.2-3B"}
CLOSE = {"Q15": "close-q15", "Q3B": "close-q3b", "L3B": "close-l3b"}

out: list[str] = []
notes: list[str] = []
checks = {"kv": 0, "ppl": 0}


def jl(run: str, name: str) -> list[dict]:
    p = RES / f"kaggle-{run}" / "results" / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def bundle(run: str) -> str:
    p = RES / f"kaggle-{run}" / "results" / "env.json"
    if not p.exists():
        return "n/a"
    shas = json.loads(p.read_text()).get("bin_sha256", {})
    h = shas.get("libggml-cuda.so.0") or shas.get("libggml-cuda.so")
    return h[:8] + "…" if h else "not recorded (pre-night-3 run)"


def src(*runs: str, files: str) -> str:
    return "Source: " + "; ".join(f"`results/kaggle-{r}/results/{files}` (bundle {bundle(r)})" for r in runs)


def table(head: list[str], rows: list[list[str]]) -> None:
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    out.append("")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    out.append(f"Figure: `docs/figures/{name}.pdf`\n")


# ---------------------------------------------------------------- operator level
def operator() -> dict:
    runs = ["night1", "night3", "i8", "final"]
    mm: dict = {}
    for run in runs:
        for r in jl(run, "matmul_paths.jsonl"):
            k = (r["type"], r["probe_flags"], r["K"])
            vals = (r["delta_median"], r["err_alone_vs_fp64_median"], r["err_batch_vs_fp64_median"])
            if k in mm and mm[k][1] != vals:
                notes.append(f"operator {k}: {mm[k][0]} {mm[k][1]} vs {run} {vals}")
            mm.setdefault(k, (run, vals, r))
    out.append("## Operator level (T4, synthetic weights/activations, K = N = 1536)\n")
    out.append("δ = median over tokens of ‖y_alone − y_batch64‖ / ‖y_batch64‖; err = relative error vs FP64.\n")
    rows = []
    for t in OP_ORDER:
        a, g = mm.get((t, "0", 1536)), mm.get((t, "384", 1536))
        if a:
            rows.append([t.upper().replace("_K", "_K"), f"{a[1][0]:.2e}", f"{a[1][1]:.3e} / {a[1][2]:.3e}",
                         f"{g[1][0]:.2e}" if g else "—", f"{g[1][1]:.3e} / {g[1][2]:.3e}" if g else "—"])
    table(["Format", "δ upstream", "err decode / prompt, upstream", "δ under G", "err decode / prompt, under G"], rows)
    out.append(src("final", files="matmul_paths.jsonl") + "\n")

    flags = ["0", "1", "2", "4", "8", "15", "16", "32", "47", "64", "128", "256", "384"]
    out.append("### δ median by intervention (all runs merged; cells agree bit for bit where runs overlap)\n")
    rows = []
    for t in OP_ORDER:
        rows.append([t] + [f"{mm[(t, f, 1536)][1][0]:.1e}" if (t, f, 1536) in mm else "" for f in flags])
    table(["type"] + [f"{f} ({FLAG_NAMES[f]})" for f in flags], rows)
    out.append(src(*runs, files="matmul_paths.jsonl") + "\n")
    k8960 = [(t, f) for (t, f, K) in mm if K == 8960]
    if k8960:
        rows = [[t, f, f"{mm[(t, f, 8960)][1][0]:.2e}", f"{mm[(t, f, 8960)][1][1]:.3e} / {mm[(t, f, 8960)][1][2]:.3e}"]
                for t, f in sorted(k8960, key=lambda x: (OP_ORDER.index(x[0]) if x[0] in OP_ORDER else 99, int(x[1])))]
        out.append("### K = 8960 (ffn_down shape)\n")
        table(["type", "flags", "δ", "err decode / prompt"], rows)

    # figure: delta and error, upstream vs G
    ts = [t for t in OP_ORDER if (t, "0", 1536) in mm and (t, "384", 1536) in mm]
    x = range(len(ts))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    fig.subplots_adjust(wspace=0.35)
    a1.bar([i - 0.2 for i in x], [mm[(t, "0", 1536)][1][0] for t in ts], 0.4, color=UP, label="upstream")
    a1.bar([i + 0.2 for i in x], [mm[(t, "384", 1536)][1][0] for t in ts], 0.4, color=FIX, label="G (I9+I10)")
    a1.set_yscale("log"); a1.set_ylabel("δ, decode vs prompt kernel"); a1.set_xticks(list(x), [t.upper() for t in ts], rotation=60)
    a1.legend(frameon=False); a1.set_title("(a) operator mismatch")
    for i, t in enumerate(ts):
        u, g = mm[(t, "0", 1536)][1], mm[(t, "384", 1536)][1]
        a2.plot([i - 0.15, i - 0.15], [u[1], u[2]], color=UP, lw=0.8)
        a2.plot(i - 0.15, u[1], "o", mfc="white", mec=UP, ms=4); a2.plot(i - 0.15, u[2], "s", mfc="white", mec=UP, ms=4)
        a2.plot(i + 0.15, g[1], "o", color=FIX, ms=4); a2.plot(i + 0.15, g[2], "s", color=FIX, ms=4)
    q8 = mm[("q8_0", "0", 1536)][1][2]
    a2.axhline(q8, color=FLOOR, ls="--", lw=0.7)
    a2.text(1.5, q8 * 0.93, "Q8_0 level", ha="left", va="top", fontsize=6)
    a2.set_yscale("log"); a2.set_ylabel("relative error vs FP64"); a2.set_xticks(list(x), [t.upper() for t in ts], rotation=60)
    a2.plot([], [], "o", mfc="white", mec="k", ms=4, label="decode kernel (MMVQ)")
    a2.plot([], [], "s", mfc="white", mec="k", ms=4, label="prompt kernel (MMQ)")
    a2.set_ylim(2.5e-4, 4e-2)
    a2.legend(frameon=False, loc="lower right"); a2.set_title("(b) error vs FP64 (open: upstream, filled: G)")
    save(fig, "fig_operator")
    return mm


# ---------------------------------------------------------------- gate (M5/M8)
# Correction formulation of each (type, kernel) on T4 at flags 0 (README.md). D4 prompt kernels keep an
# unrounded FP32 activation scale, so their output moves smoothly (~1e-7) with any input change: a third class.
CODE = {("q8_0", "alone"), ("q6_K", "alone"), ("q4_K", "alone"), ("q5_K", "alone"), ("q4_0", "batch")}
FP32SUM = {("q4_K", "batch"), ("q5_K", "batch"), ("q4_1", "alone"), ("q4_1", "batch"), ("q4_0", "alone"), ("q5_0", "alone")}
D4 = {("q8_0", "batch"), ("q5_0", "batch"), ("q6_K", "batch")}
D4C = "#4a7ab5"


def gclass(t: str, p: str) -> str:
    return "F16" if t == "f16" else "code" if (t, p) in CODE else "FP32 sum" if (t, p) in FP32SUM else "D4 FP32 scale"


def gate() -> None:
    g = [r for r in jl("night1", "gate.jsonl") if r["probe_flags"] == "0"]
    out.append("## Gate probe (M5/M8): response of each kernel to an input perturbation ε\n")
    out.append("Correction formulation on T4: code/integer or none = Q8_0, Q6_K, Q4_K/Q5_K decode, "
               "Q4_0 prompt; FP32 sum = Q4_K/Q5_K prompt, Q4_1 both, Q4_0/Q5_0 decode; D4 FP32 scale = Q8_0/Q5_0/Q6_K "
               "prompt (unrounded FP32 activation scale: never bit-identical, smooth ~1e-7 changes).\n")
    eps = sorted({r["eps"] for r in g})
    rows = []
    for t in ["f16", "q8_0", "q6_K", "q4_K", "q5_K", "q4_0", "q5_0", "q4_1"]:
        for p in ("alone", "batch"):
            rs = {r["eps"]: r for r in g if r["type"] == t and r["path"] == p}
            if rs:
                cls = gclass(t, p)
                rows.append([t, "decode" if p == "alone" else "prompt", cls] +
                            [f"{rs[e]['bitwise_same_frac']:.2f} / {rs[e]['rel_change_max']:.1e}" if e in rs else "" for e in eps])
    table(["type", "kernel", "correction"] + [f"ε={e:g}: identical / max" for e in eps], rows)
    out.append(src("night1", files="gate.jsonl") + "\n")
    i8 = [r for r in jl("i8", "gate.jsonl")]
    if i8:
        rows = [[r["type"], "decode" if r["path"] == "alone" else "prompt", r["probe_flags"], f"{r['eps']:g}",
                 f"{r['bitwise_same_frac']:.2f}", f"{r['rel_change_max']:.1e}"]
                for r in i8 if r["type"] in ("q4_K", "q5_K") and r["eps"] == 1e-6]
        out.append("I8 moves the Q4_K/Q5_K prompt kernel to the code sum (ε = 1e-6):\n")
        table(["type", "kernel", "flags", "ε", "identical", "max change"], rows)
        out.append(src("i8", files="gate.jsonl") + "\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for (t, p) in sorted({(r["type"], r["path"]) for r in g}):
        rs = sorted((r for r in g if r["type"] == t and r["path"] == p), key=lambda r: r["eps"])
        c = {"F16": "#999999", "code": FIX, "FP32 sum": UP, "D4 FP32 scale": D4C}[gclass(t, p)]
        a1.plot([r["eps"] for r in rs], [r["bitwise_same_frac"] for r in rs], "-o", color=c, ms=2.5, lw=0.8, alpha=0.8)
        nz = [r for r in rs if r["rel_change_max"] > 0]  # 0 (all outputs identical) cannot be drawn on a log axis
        a2.plot([r["eps"] for r in nz], [r["rel_change_max"] for r in nz], "-o", color=c, ms=2.5, lw=0.8, alpha=0.8)
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.set_xlabel("input perturbation ε (relative)")
    for c, lab in ((FIX, "code sum / integer offset"), (UP, "FP32 sum"), (D4C, "FP32 activation scale (D4 prompt)"),
                   ("#999999", "F16")):
        a2.plot([], [], color=c, label=lab)
    a1.set_ylabel("fraction of outputs bit-identical"); a1.set_title("(a) the quantizer usually erases ε")
    a2.legend(frameon=False, loc="lower right", fontsize=6)
    a2.set_yscale("log"); a2.set_ylabel("max relative output change\n(points with no change omitted)")
    a2.set_title("(b) ...and occasionally amplifies it")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "fig_gate")


# ---------------------------------------------------------------- end to end (existing runs, Qwen2.5-1.5B)
def kv_runs(runs: list[str]) -> dict:
    kv: dict = {}
    for run in runs:
        for r in jl(run, "kv_paths.jsonl"):
            if r["cmp"] != "gen":
                continue
            k = (r["tag"], r["probe_flags"])
            v = (r["maxabs_median"], r["argmax_flips"])
            if k in kv and kv[k][1] != v:
                notes.append(f"kv {k}: {kv[k][0]} {kv[k][1]} vs {run} {v}")
            kv.setdefault(k, (run, v))
    return kv


def m8() -> None:
    kv = kv_runs(["night1", "night3", "i8", "final"])
    out.append("## M8: the direction of alignment (Qwen2.5-1.5B, median max|Δlogit| on generated positions, (argmax flips))\n")
    q8 = kv[("Q8_0", "0")][1]
    rows = [[f, *(f"{kv[(f, fl)][1][0]:.3f} ({kv[(f, fl)][1][1]})" if (f, fl) in kv else "—" for fl in ("0", "128", "384"))]
            for f in ("Q4_1", "Q5_1")]
    table(["format", "upstream", "I9 alone (mismatch removed)", "G = I9+I10 (both code sum)"], rows)
    rows = [[f, *(f"{kv[(f, fl)][1][0]:.3f} ({kv[(f, fl)][1][1]})" if (f, fl) in kv else "—" for fl in ("0", "4", "64"))]
            for f in ("Q4_K", "Q5_K", "Q4_K_M")]
    table(["format", "upstream", "I4 (decode aligned to MMQ's FP32 sum)", "I8 (prompt aligned to MMVQ's code sum)"], rows)
    out.append(f"Q8_0 floor {q8[0]:.3f} ({q8[1]}); F16 {kv[('F16', '0')][1][0]:.3f} ({kv[('F16', '0')][1][1]}); "
               f"F16 with FP32 cuBLAS {kv[('F16_cublas_f32', '0')][1][0]:.3f} ({kv[('F16_cublas_f32', '0')][1][1]}).\n")
    out.append(src("night1", "i8", "final", files="kv_paths.jsonl") + "\n")

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, fmts, fls, labels, cols, title in (
            (axs[0], ("Q4_1", "Q5_1"), ("0", "128", "384"), ("upstream", "I9: mismatch removed", "I9+I10: both code sum"),
             (UP, ALT, FIX), "(a) agreeing is not enough (Q4_1/Q5_1)"),
            (axs[1], ("Q4_K", "Q5_K"), ("0", "4", "64"), ("upstream", "I4: aligned to FP32 sum", "I8: aligned to code sum"),
             (UP, ALT, FIX), "(b) direction of alignment (K-quants)")):
        for j, (fl, lab, c) in enumerate(zip(fls, labels, cols)):
            xs = [i + (j - 1) * 0.27 for i in range(len(fmts))]
            ys = [kv[(f, fl)][1][0] for f in fmts]
            ax.bar(xs, ys, 0.27, color=c, label=lab)
            for xx, f in zip(xs, fmts):
                ax.text(xx, kv[(f, fl)][1][0] + 0.01, str(kv[(f, fl)][1][1]), ha="center", va="bottom", fontsize=6)
        ax.axhline(q8[0], color=FLOOR, ls="--", lw=0.7); ax.text(-0.5, q8[0] - 0.01, "Q8_0", ha="left", va="top", fontsize=6,
                                                                 bbox=dict(fc="white", ec="none", pad=0.5))
        ax.axhline(kv[("F16", "0")][1][0], color=FLOOR, ls=":", lw=0.7)
        ax.text(-0.5, kv[("F16", "0")][1][0] - 0.01, "F16", ha="left", va="top", fontsize=6, bbox=dict(fc="white", ec="none", pad=0.5))
        ax.set_xticks(range(len(fmts)), fmts); ax.legend(frameon=False, loc="upper right", fontsize=6); ax.set_title(title)
        ax.set_ylim(0, 1.15)
    axs[0].set_ylabel("median max |Δlogit|, decode vs recompute\n(numbers: argmax flips / 384)")
    save(fig, "fig_m8")


# ---------------------------------------------------------------- latency
def latency() -> None:
    lat = [r for r in jl("final", "latency.jsonl") if r["cmp"] == "gen"]
    log = (RES / "kaggle-final" / "results" / "job.log").read_text() if (RES / "kaggle-final" / "results" / "job.log").exists() else ""
    clk = [(int(m.group(1)), int(m.group(2)), m.group(3)) for m in
           re.finditer(r"latency \S+ rep=\d+ flags=\d+ rc=0 clocks\[(\d+) MHz, (\d+), [\d.]+ W, (0x[0-9a-f]+)\]", log)]
    out.append("## Cost of the fixes (T4; 5 ABBA-interleaved repeats per condition; each repeat = median over 8 prompts)\n")
    rows, data = [], []
    for fmt, alt, what, name in (("Q4_K", "64", "prefill", "I8"), ("Q4_1", "256", "prefill", "I10"),
                                 ("Q4_0", "4", "decode", "I4"), ("Q5_0", "4", "decode", "I4")):
        key = f"{what}_ms_median"
        b = [r[key] for r in lat if r["tag"].startswith(f"LAT_{fmt}_") and r["probe_flags"] == "0"]
        t = [r[key] for r in lat if r["tag"].startswith(f"LAT_{fmt}_") and r["probe_flags"] == alt]
        if len(b) < 2 or len(t) < 2:
            continue
        qb, qt = st.quantiles(b, n=4, method="inclusive"), st.quantiles(t, n=4, method="inclusive")
        rows.append([f"{name} on {fmt}", what, f"{st.median(b):.3f} [{qb[0]:.3f}, {qb[2]:.3f}]",
                     f"{st.median(t):.3f} [{qt[0]:.3f}, {qt[2]:.3f}]", f"{100 * (st.median(t) / st.median(b) - 1):+.1f}%"])
        data.append((f"{name} on {fmt}\n({what})", b, t))
    table(["change", "phase", "baseline ms, median [IQR]", "with fix ms, median [IQR]", "Δ median"], rows)
    if clk:
        out.append(f"SM clock {min(c[0] for c in clk)}–{max(c[0] for c in clk)} MHz (max boost 1590; clock_ratio ≥ "
                   f"{min(c[0] for c in clk) / 1590:.3f}), {min(c[1] for c in clk)}–{max(c[1] for c in clk)} °C, "
                   f"throttle reasons {sorted({c[2] for c in clk})} over {len(clk)} runs.\n")
    out.append(src("final", files="latency.jsonl, job.log") + "\n")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, (lab, b, t) in enumerate(data):
        m = st.median(b)
        ax.plot([i - 0.15] * len(b), [100 * (v / m - 1) for v in b], "o", mfc="white", mec=UP, ms=3)
        ax.plot([i + 0.15] * len(t), [100 * (v / m - 1) for v in t], "o", color=FIX, ms=3)
    ax.axhline(0, color=FLOOR, lw=0.5)
    ax.set_xticks(range(len(data)), [d[0] for d in data], fontsize=6)
    ax.set_ylabel("time vs baseline median (%)")
    ax.plot([], [], "o", mfc="white", mec=UP, ms=3, label="upstream"); ax.plot([], [], "o", color=FIX, ms=3, label="with fix")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_latency")


# ---------------------------------------------------------------- closing run (4f): three models
def close() -> None:
    have = {m: run for m, run in CLOSE.items() if jl(run, "ppl_paths.jsonl")}
    out.append("## Closing run (§4f): upstream vs the global fix G on three models\n")
    if not have:
        out.append("PENDING: `results/kaggle-close-*` not present yet.\n")
        return
    kv, ppl = {}, {}
    for m, run in have.items():
        for r in jl(run, "kv_paths.jsonl"):
            if r["cmp"] == "gen":
                kv[(m, r["tag"].split("_", 1)[1], r["probe_flags"])] = r
        for r in jl(run, "ppl_paths.jsonl"):
            ppl[(m, r["tag"].split("_", 1)[1], r["probe_flags"])] = r
    # Cross-check against earlier runs of the same configuration (other bundles): end to end against the
    # 1.5B night/i8/final runs (untagged) and the 3B gen runs; PPL only where the chunk count matches (gen, 24).
    earlier = {"Q15": (["night1", "night3", "i8", "final"], ""), "Q3B": (["gen-q3b"], "Q3B_"), "L3B": (["gen-l3b"], "L3B_")}
    for m in have:
        runs, pre = earlier[m]
        for run in runs:
            for r in jl(run, "kv_paths.jsonl"):
                k = (m, r["tag"][len(pre):], r["probe_flags"])
                if r["cmp"] == "gen" and k in kv:
                    checks["kv"] += 1
                    if (r["maxabs_median"], r["argmax_flips"]) != (kv[k]["maxabs_median"], kv[k]["argmax_flips"]):
                        notes.append(f"close kv {k} vs {run}")
            for r in jl(run, "ppl_paths.jsonl"):
                k = (m, r["tag"][len(pre):], r["probe_flags"])
                if k in ppl and r.get("chunks") == ppl[k].get("chunks"):
                    checks["ppl"] += 1
                    if (r["ppl_batch"], r["ppl_decode"], r["argmax_flips"]) != (ppl[k]["ppl_batch"], ppl[k]["ppl_decode"], ppl[k]["argmax_flips"]):
                        notes.append(f"close ppl {k} vs {run}")
    for m in have:
        q8, f16 = kv.get((m, "Q8_0", "0")), kv.get((m, "F16", "0"))
        out.append(f"### {MODELS[m]}: end to end, median max|Δlogit| (flips); PPL decode − batch (z = ΔNLL/SE_chunk, argmax disagreements)\n")
        rows = []
        for f in ["Q8_0"] + SEVEN:
            cells = [f]
            for fl in ("0", "128", "384"):
                k, p = kv.get((m, f, fl)), ppl.get((m, f, fl))
                cells.append(f"{k['maxabs_median']:.3f} ({k['argmax_flips']})" if k else "—")
                cells.append(f"{100 * (p['ppl_decode'] / p['ppl_batch'] - 1):+.2f}% (z {p['dnll'] / p['dnll_se_chunk']:+.1f}, {p['argmax_flips']})"
                             if p else "—")
            rows.append(cells)
        table(["format", "e2e upstream", "PPL upstream", "e2e I9", "PPL I9", "e2e G", "PPL G"], rows)
        if q8 and f16:
            out.append(f"F16 e2e {f16['maxabs_median']:.3f} ({f16['argmax_flips']}).\n")
        out.append(src(have[m], files="kv_paths.jsonl, ppl_paths.jsonl") + "\n")

    fig, axs = plt.subplots(1, len(have), figsize=(7.0, 2.3), sharey=False, squeeze=False)
    for ax, m in zip(axs[0], have):
        for j, (fl, c, lab) in enumerate((("0", UP, "upstream"), ("384", FIX, "G"))):
            xs = [i + (j - 0.5) * 0.38 for i in range(len(SEVEN))]
            ax.bar(xs, [kv[(m, f, fl)]["maxabs_median"] if (m, f, fl) in kv else 0 for f in SEVEN], 0.38, color=c, label=lab)
        for f, ls in (("Q8_0", "--"), ("F16", ":")):
            if (m, f, "0") in kv:
                ax.axhline(kv[(m, f, "0")]["maxabs_median"], color=FLOOR, ls=ls, lw=0.7)
                ax.text(len(SEVEN) - 0.5, kv[(m, f, "0")]["maxabs_median"] + 0.01, f, ha="right", fontsize=6)
        ax.set_xticks(range(len(SEVEN)), SEVEN, rotation=60); ax.set_title(MODELS[m])
    axs[0][0].set_ylabel("median max |Δlogit|"); axs[0][0].legend(frameon=False, fontsize=6)
    save(fig, "fig_e2e_models")

    fig, axs = plt.subplots(1, len(have), figsize=(7.0, 2.3), sharey=True, squeeze=False)
    fmts = ["Q8_0"] + SEVEN  # Q8_0: control without an offset/min term, untouched by G
    for ax, m in zip(axs[0], have):
        for j, (fl, c, lab) in enumerate((("0", UP, "upstream"), ("384", FIX, "G"))):
            xs = [i + (j - 0.5) * 0.38 for i in range(len(fmts))]
            ax.bar(xs, [ppl[(m, f, fl)]["dnll"] / ppl[(m, f, fl)]["dnll_se_chunk"] if (m, f, fl) in ppl else 0 for f in fmts],
                   0.38, color=c, label=lab)
        ax.axvspan(-0.5, 0.5, color="#eeeeee", zorder=0)
        for y in (-2, 2):
            ax.axhline(y, color=FLOOR, ls=":", lw=0.6)
        ax.axhline(0, color=FLOOR, lw=0.5)
        ax.set_xticks(range(len(fmts)), fmts, rotation=60); ax.set_title(MODELS[m])
    axs[0][0].set_ylabel("decode − batch ΔNLL, z (chunk SE)"); axs[0][0].legend(frameon=False, fontsize=6)
    save(fig, "fig_ppl_models")


# ---------------------------------------------------------------- episodes (4g): the prior work's metric
EP_RUNS = {"Q7B": "ep-q7b", "Q3B": "ep-q3b", "L3B": "ep-l3b"}
EP_MODELS = {"Q7B": "Qwen2.5-7B", **MODELS}


def episode_rates() -> None:
    import math
    have = {m: run for m, run in EP_RUNS.items() if jl(run, "episodes.jsonl")}
    out.append("## Episode divergence (§4g): llama-server, cache_prompt on vs off, 80 MT-Bench episodes × 4 requests\n")
    if not have:
        out.append("PENDING: `results/kaggle-ep-*` not present yet.\n")
        return
    rows, fig_data = [], {}
    for m, run in have.items():
        runs: dict = {}
        for r in jl(run, "episodes.jsonl"):
            runs.setdefault((r["tag"].split("_", 1)[1], r["probe_flags"], r["arm"]), {}).setdefault(r["episode"], {})[r["request"]] = r["tokens"]
        for (fmt, fl, arm) in sorted(runs):
            if arm != "cache" or fmt.endswith("_rep") or (fmt, fl, "recompute") not in runs:
                continue
            a, b = runs[(fmt, fl, "cache")], runs[(fmt, fl, "recompute")]
            eps = sorted(set(a) & set(b))
            k = sum(any(a[e].get(i) != b[e].get(i) for i in set(a[e]) | set(b[e])) for e in eps)
            n = len(eps); p = k / n; z = 1.96
            c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append([EP_MODELS[m], fmt, FLAG_NAMES.get(fl, fl), f"{k}/{n}", f"{100 * p:.1f}% [{100 * (c - h):.1f}, {100 * (c + h):.1f}]"])
            fig_data.setdefault(m, {})[(fmt, fl)] = (100 * p, 100 * (c - h), 100 * (c + h))
        out.append(src(run, files="episodes.jsonl") + "\n")
    table(["model", "format", "flags", "divergent episodes", "rate [Wilson 95% CI]"], rows)
    fig, axs = plt.subplots(1, len(fig_data), figsize=(7.0, 2.3), sharey=True, squeeze=False)
    order = [("F16", "0"), ("Q8_0", "0"), ("Q4_K_M", "0"), ("Q4_K_M", "384"), ("Q4_0", "0"), ("Q4_0", "384")]
    for ax, (m, d) in zip(axs[0], fig_data.items()):
        keys = [k for k in order if k in d]
        cols = ["#999999" if f == "F16" else FLOOR if f == "Q8_0" else (UP if fl == "0" else FIX) for f, fl in keys]
        ax.bar(range(len(keys)), [d[k][0] for k in keys], color=cols,
               yerr=[[d[k][0] - d[k][1] for k in keys], [d[k][2] - d[k][0] for k in keys]], capsize=2, error_kw={"lw": 0.6})
        ax.set_xticks(range(len(keys)), [f + ("\nG" if fl == "384" else "") for f, fl in keys], fontsize=6)
        ax.set_title(EP_MODELS[m]); ax.set_ylim(0, 100)
    axs[0][0].set_ylabel("episodes diverging, cache vs recompute (%)")
    save(fig, "fig_episodes")


def main() -> None:
    out.append("# Generated tables (do not edit; run `probes/make_report.py`)\n")
    out.append("All runs: Tesla T4 (sm_75), llama.cpp b10434 + runtime-flag patch, CUDA 12.6, cuBLAS 12.6.0.22.\n")
    operator()
    gate()
    m8()
    latency()
    close()
    episode_rates()
    out.append("## Cross-run consistency\n")
    out.append(f"Closing-run rows re-measured from earlier bundles: {checks['kv']} end-to-end and {checks['ppl']} PPL comparisons.\n")
    out.append("Every overlapping (type, flags, shape) operator result and (tag, flags) end-to-end result agrees bit for bit "
               "across runs.\n" if not notes else "**Mismatches between runs:**\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)} and figures in {FIG.relative_to(ROOT)}; closing-run cross-checks {checks}; "
          f"cross-run mismatches: {len(notes)}")
    for n in notes:
        print("  ", n)


if __name__ == "__main__":
    main()
