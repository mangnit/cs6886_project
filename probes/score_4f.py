#!/usr/bin/env python3
"""Score the closing run against its pass/fail criteria (experiment 4f, F1-F4).

    python probes/score_4f.py results/kaggle-close-q15/results results/kaggle-close-q3b/results results/kaggle-close-l3b/results

Raw JSONL in, verdicts out; thresholds are fixed in this file. z = dnll / dnll_se_chunk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEVEN = ("Q4_0", "Q5_0", "Q4_1", "Q5_1", "Q4_K", "Q5_K", "Q4_K_M")
G = "384"
E3_SET = {("Q15", "Q4_0"), ("Q15", "Q5_0"), ("Q3B", "Q4_0"), ("Q3B", "Q5_0"), ("Q3B", "Q4_1"), ("Q3B", "Q5_1")}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    kv, ppl, models = {}, {}, []
    for d in map(Path, sys.argv[1:]):
        for r in load(d / "kv_paths.jsonl"):
            if r["cmp"] == "gen":
                m, fmt = r["tag"].split("_", 1)
                kv[(m, fmt, r["probe_flags"])] = r
                if m not in models:
                    models.append(m)
        for r in load(d / "ppl_paths.jsonl"):
            m, fmt = r["tag"].split("_", 1)
            ppl[(m, fmt, r["probe_flags"])] = r

    def z(r: dict) -> float:
        return r["dnll"] / r["dnll_se_chunk"]

    fails = {k: 0 for k in ("F1", "F2", "F3", "F4")}
    print("e2e median max|dlogit| (argmax flips); ratio to the model's Q8_0 at the same flags in brackets")
    for m in models:
        base = {fl: kv.get((m, "Q8_0", fl)) for fl in ("0", G)}
        if not base[G]:
            print(f"  {m}: no Q8_0 control at flags {G}"); continue
        q8 = base[G]["maxabs_median"]
        f16 = kv.get((m, "F16", "0"))
        print(f"  {m}: F16 {f16['maxabs_median']:.3f}  Q8_0 {base['0']['maxabs_median']:.3f} | G {q8:.3f}" if f16 else f"  {m}: Q8_0 G {q8:.3f}")
        for fmt in SEVEN:
            row = [f"    {fmt:7}"]
            for fl in ("0", "128", G):
                r = kv.get((m, fmt, fl))
                if r:
                    row.append(f"{fl:>3}: {r['maxabs_median']:.3f} ({r['argmax_flips']:2}) [{r['maxabs_median']/q8:.2f}x]")
            e1 = kv.get((m, fmt, G))
            v = ""
            if e1 and e1["maxabs_median"] > 1.25 * q8:
                fails["F1"] += 1; v += "  F1 FAIL"
            e2 = kv.get((m, fmt, "128"))
            if e2:
                ratio = e2["maxabs_median"] / q8
                if ratio <= 1.25:
                    fails["F2"] += 1; v += "  F2 FAIL"
                elif ratio < 1.5:
                    v += "  F2 partial"
            print("  ".join(row) + v)

    print("\nPPL decode vs batch: change, z = dnll/se_chunk, argmax disagreements [ratio to Q8_0 at G]")
    screen = []
    for m in models:
        q8 = ppl.get((m, "Q8_0", G))
        if not q8:
            print(f"  {m}: no Q8_0 PPL at flags {G}"); continue
        print(f"  {m}: Q8_0 G {100*(q8['ppl_decode']/q8['ppl_batch']-1):+.2f}% z {z(q8):+.2f} flips {q8['argmax_flips']}")
        for fmt in SEVEN:
            row = [f"    {fmt:7}"]
            for fl in ("0", "128", G):
                r = ppl.get((m, fmt, fl))
                if r:
                    row.append(f"{fl:>3}: {100*(r['ppl_decode']/r['ppl_batch']-1):+.2f}% z {z(r):+.2f} fl {r['argmax_flips']:3} [{r['argmax_flips']/q8['argmax_flips']:.2f}x]")
            g = ppl.get((m, fmt, G))
            v = ""
            if g:
                if (m, fmt) in E3_SET:
                    if abs(z(g)) >= 2.0:
                        fails["F3"] += 1; v += "  F3 FAIL"
                    elif abs(z(g)) >= 1.5:
                        v += "  F3 partial"
                if g["argmax_flips"] > 1.5 * q8["argmax_flips"]:
                    fails["F4"] += 1; v += "  F4 FAIL"
                elif g["argmax_flips"] > 1.3 * q8["argmax_flips"]:
                    v += "  F4 partial"
                if abs(z(g)) >= 2.5:
                    screen.append(f"{m} {fmt} z {z(g):+.2f}")
            print("  ".join(row) + v)

    print("\nscreen (extra check, not a criterion), |z| >= 2.5 under G:", screen or "none")
    print("verdicts:", {k: ("FALSIFIED (%d)" % n if n else "pass") for k, n in fails.items()})


if __name__ == "__main__":
    main()
