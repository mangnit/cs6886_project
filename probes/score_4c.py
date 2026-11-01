#!/usr/bin/env python3
"""Score the explanatory probe against its pass/fail criteria (experiment 4c).

    python probes/score_4c.py results/kaggle-explain-q3b/results results/kaggle-explain-l3b/results

Reads raw real_replay.jsonl and kv_bisect.jsonl only, applies P1-P4 and the decision rule exactly
as written here, and prints the verdict. Nothing here is tuned after seeing the data.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def load(d: str, name: str) -> list[dict]:
    p = Path(d) / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):  # average ranks over ties
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    qdir, ldir = sys.argv[1], sys.argv[2]
    rows = {"Qwen": load(qdir, "real_replay.jsonl"), "Llama": load(ldir, "real_replay.jsonl")}
    bis = {"Qwen": load(qdir, "kv_bisect.jsonl"), "Llama": load(ldir, "kv_bisect.jsonl")}

    def sel(model: str, fmt: str, flags: str, clip: float) -> list[dict]:
        return [r for r in rows[model] if r["tag"].endswith(fmt) and r["probe_flags"] == flags and abs(r["clip"] - clip) < 1e-9]

    print("Context: median per-matmul values, clip 0")
    print(f"{'model':6s} {'fmt':5s} {'flags':5s} {'n':>4s} {'crest_med':>9s} {'frac>4':>7s} {'err_dec':>9s} {'err_bat':>9s} {'excess':>10s}")
    for m in ("Qwen", "Llama"):
        for fmt in ("Q4_0", "Q5_0", "Q8_0"):
            for fl in ("0", "4"):
                s = sel(m, fmt, fl, 0)
                if s:
                    print(f"{m:6s} {fmt:5s} {fl:5s} {len(s):4d} {st.median(r['crest_median'] for r in s):9.3f} "
                          f"{st.median(r['crest_frac_gt4'] for r in s):7.4f} {st.median(r['err_decode'] for r in s):9.3e} "
                          f"{st.median(r['err_batch'] for r in s):9.3e} {st.median(r['excess'] for r in s):+10.3e}")

    q0, l0 = sel("Qwen", "Q4_0", "0", 0), sel("Llama", "Q4_0", "0", 0)
    if not q0 or not l0:
        print("missing data"); return
    eq, el = st.median(r["excess"] for r in q0), st.median(r["excess"] for r in l0)
    p1_ratio = eq / el if el > 0 else float("inf")
    p1 = p1_ratio >= 2
    print(f"\nP1 initial error: median excess Q4_0 f0  Qwen {eq:.3e}  Llama {el:.3e}  ratio {p1_ratio:.2f} (>= 2) -> {'HOLDS' if p1 else 'FAILS'}")

    both = q0 + l0
    rho = spearman([r["crest_median"] for r in both], [r["excess"] for r in both])
    p2 = rho >= 0.5
    print(f"P2 association: Spearman rho(crest_median, excess) over {len(both)} matmuls = {rho:.3f} (>= 0.5) -> {'HOLDS' if p2 else 'FAILS'}")

    llama_crest = st.median(r["crest_median"] for r in l0)
    chosen = None
    for c in (5, 4, 3.5, 3):
        qc = sel("Qwen", "Q4_0", "0", c)
        if qc and st.median(r["crest_median"] for r in qc) <= llama_crest:
            chosen = c
            break
    if chosen is None:
        p3 = False
        print(f"P3 intervention: no c in {{5,4,3.5,3}} brings Qwen's median crest to <= Llama's {llama_crest:.3f} -> FAILS (undefined)")
    else:
        qc, lc = sel("Qwen", "Q4_0", "0", chosen), sel("Llama", "Q4_0", "0", chosen)
        eqc, elc = st.median(r["excess"] for r in qc), st.median(r["excess"] for r in lc)
        q_drop = eq / eqc if eqc > 0 else float("inf")
        l_change = max(el / elc, elc / el) if el > 0 and elc > 0 else float("inf")
        p3 = q_drop >= 2 and l_change < 1.5
        print(f"P3 intervention at c={chosen}: Qwen excess {eq:.3e} -> {eqc:.3e} (x{q_drop:.2f}, need >= 2); "
              f"Llama {el:.3e} -> {elc:.3e} (x{l_change:.2f}, need < 1.5) -> {'HOLDS' if p3 else 'FAILS'}")

    def growth(model: str) -> float | None:
        per_prompt = {}
        for r in bis[model]:
            if r.get("probe") == "kv_bisect" and r["tag"].endswith("Q4_0") and r["probe_flags"] == "0" and r["node"].startswith("l_out-"):
                per_prompt.setdefault(r["prompt"], {})[int(r["node"].split("-")[1])] = r["rel_l2"]
        g = [p[max(p)] / p[min(p)] for p in per_prompt.values() if p and p[min(p)] > 0]
        return st.median(g) if g else None
    gq, gl = growth("Qwen"), growth("Llama")
    if gq is None or gl is None:
        p4 = False
        print("P4 propagation: missing kv_bisect data -> FAILS (undefined)")
    else:
        p4 = gq / gl >= 2
        print(f"P4 propagation: last/first layer divergence  Qwen {gq:.2f}  Llama {gl:.2f}  ratio {gq/gl:.2f} (>= 2) -> {'HOLDS' if p4 else 'FAILS'}")

    print("\nDecision rule:")
    if p1 and p2 and p3:
        print("  H SUPPORTED: outliers explain the model dependence; report with the clipping intervention.")
    elif not p1:
        print("  H REJECTED (P1 fails). " + ("P4 holds: report 'propagation, not initial error'." if p4
                                            else "P4 fails: report the model dependence as unexplained."))
    else:
        print("  P1 holds but P2/P3 fail: report 'initial error differs, outliers do not explain it'.")


if __name__ == "__main__":
    main()
