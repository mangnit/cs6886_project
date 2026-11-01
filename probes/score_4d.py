#!/usr/bin/env python3
"""Score the final run against its pass/fail criteria (experiment 4d, C1-C4).

    python probes/score_4d.py results/kaggle-final/results

Raw JSONL in, verdicts out; thresholds are fixed in this file.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def load(d: Path, name: str) -> list[dict]:
    p = d / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def iqr(v: list[float]) -> tuple[float, float, float]:
    q = st.quantiles(v, n=4, method="inclusive") if len(v) >= 2 else [v[0]] * 3
    return q[0], st.median(v), q[2]


def main() -> None:
    d = Path(sys.argv[1])
    mm = {(r["type"], r["probe_flags"], r["K"]): r for r in load(d, "matmul_paths.jsonl")}

    print("C1  I9 removes M7 (operator, Q4_1/Q5_1 delta median at flags 128 <= 1e-6; falsified if > 1e-5)")
    c1 = []
    for t in ("q4_1", "q5_1"):
        r0, r = mm.get((t, "0", 1536)), mm.get((t, "128", 1536))
        if r:
            ok = r["delta_median"] <= 1e-6
            c1.append("pass" if ok else ("FALSIFIED" if r["delta_median"] > 1e-5 else "partial"))
            print(f"    {t}: {r0['delta_median']:.2e} -> {r['delta_median']:.2e} (max {r['delta_max']:.2e})  {c1[-1]}")

    print("C2  I10 lowers Q4_1/Q5_1 error vs FP64 on both paths to <= 1.2e-2 (falsified if either > 1.35e-2)")
    for t in ("q4_1", "q5_1"):
        r0, r = mm.get((t, "0", 1536)), mm.get((t, "256", 1536))
        if r:
            worst = max(r["err_alone_vs_fp64_median"], r["err_batch_vs_fp64_median"])
            v = "pass" if worst <= 1.2e-2 else ("FALSIFIED" if worst > 1.35e-2 else "partial")
            print(f"    {t}: alone/batch {r0['err_alone_vs_fp64_median']:.3e}/{r0['err_batch_vs_fp64_median']:.3e} -> "
                  f"{r['err_alone_vs_fp64_median']:.3e}/{r['err_batch_vs_fp64_median']:.3e}  {v}")

    kv = {(r["tag"], r["probe_flags"]): r for r in load(d, "kv_paths.jsonl") if r["cmp"] == "gen"}
    print("C3  M8 test, end to end: I9 alone stays >= 0.6 (falsified if <= 0.45), I9+I10 falls to <= 0.45 (falsified if > 0.45)")
    for t in ("Q4_1", "Q5_1"):
        a, b, c = kv.get((t, "0")), kv.get((t, "128")), kv.get((t, "384"))
        if a and b and c:
            x = b["maxabs_median"]
            v1 = "pass" if x >= 0.6 else ("FALSIFIED" if x <= 0.45 else "partial")
            v2 = "pass" if c["maxabs_median"] <= 0.45 else "FALSIFIED"
            print(f"    {t}: flags0 {a['maxabs_median']:.3f} ({a['argmax_flips']} flips) | I9 {x:.3f} "
                  f"({b['argmax_flips']}) {v1} | I9+I10 {c['maxabs_median']:.3f} ({c['argmax_flips']}) {v2}")

    lat = [r for r in load(d, "latency.jsonl") if r["cmp"] == "gen"]
    print("C4  costs: median [IQR] over 5 ABBA-interleaved repeats, each repeat = medians over 8 prompts")
    for fmt, alt, what in (("Q4_K", "64", "prefill"), ("Q4_1", "256", "prefill"), ("Q4_0", "4", "decode"), ("Q5_0", "4", "decode")):
        key = f"{what}_ms_median"
        base = [r[key] for r in lat if r["tag"].startswith(f"LAT_{fmt}_") and r["probe_flags"] == "0"]
        test = [r[key] for r in lat if r["tag"].startswith(f"LAT_{fmt}_") and r["probe_flags"] == alt]
        if len(base) >= 2 and len(test) >= 2:
            b, t_ = iqr(base), iqr(test)
            ch = 100 * (t_[1] / b[1] - 1)
            if what == "prefill":
                v = "pass (<= 3%)" if abs(ch) <= 3 else ("FALSIFIED (> 5%)" if abs(ch) > 5 else "partial")
            else:
                v = "as expected (+4..+10%)" if 4 <= ch <= 10 else "outside expected range"
            print(f"    {fmt} flags {alt} {what}: {b[1]:.3f} [{b[0]:.3f}, {b[2]:.3f}] -> {t_[1]:.3f} [{t_[0]:.3f}, {t_[2]:.3f}] ms  "
                  f"({ch:+.1f}%)  {v}")


if __name__ == "__main__":
    main()
