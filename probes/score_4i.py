#!/usr/bin/env python3
"""Score the native sm_61 (MX330) numerics run against its pass/fail criteria (experiment 4i, P1-P5).

    python probes/score_4i.py results/mx330-sm61-20261029

Expects <run>/operator/matmul_paths.jsonl and <run>/close/{kv_paths,ppl_paths}.jsonl. The floor is
Q6_K at flags 0 (Q8_0 does not fit 2 GB). Thresholds are fixed in this file. Numerics only:
no timing field is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def verdict(ok: bool, falsified: bool) -> str:
    return "FALSIFIED" if falsified else "pass" if ok else "partial"


def main() -> None:
    run = Path(sys.argv[1])
    mm = {(r["type"], r["probe_flags"]): r for r in load(run / "operator" / "matmul_paths.jsonl") if r["K"] == 1536}
    if mm:
        dev = next(iter(mm.values()))
        print(f"device {dev['device']} cc {dev['cc']}")
    print("P1 operator, upstream delta")
    v = []
    for t, lo, hi, kind in (("q4_0", None, 1e-5, "consistent"), ("q4_1", None, 1e-5, "consistent"), ("q5_1", None, 1e-5, "consistent"),
                            ("q5_0", 3e-3, None, "M4"), ("q4_K", 3e-3, None, "M4"), ("q5_K", 3e-3, None, "M4"),
                            ("q8_0", 5e-5, 1e-3, "M1"), ("q6_K", 5e-5, 1e-3, "M1"), ("q3_K", 5e-5, 1e-3, "M1")):
        r = mm.get((t, "0"))
        if not r:
            continue
        d = r["delta_median"]
        ok = (lo is None or d >= lo) and (hi is None or d <= hi)
        fals = (kind == "consistent" and d > 1e-4) or (kind == "M4" and d < 1e-3)
        v.append(verdict(ok, fals))
        print(f"  {t:5} {kind:10} delta {d:.2e}  {v[-1]}")
    print("  P1:", "FALSIFIED" if "FALSIFIED" in v else "partial" if "partial" in v else "pass")
    if ("q6_K", "384") in mm:
        ref = max(mm[("q6_K", "384")]["err_alone_vs_fp64_median"], mm[("q6_K", "384")]["err_batch_vs_fp64_median"])
        print(f"P2 operator under G: error vs FP64 <= 1.05 x Q6_K ({ref:.3e}) (falsified if > 1.15 x)")
        worst = 0.0
        for t in ("q4_0", "q5_0", "q4_1", "q5_1", "q4_K", "q5_K"):
            r = mm.get((t, "384"))
            if r:
                e = max(r["err_alone_vs_fp64_median"], r["err_batch_vs_fp64_median"])
                worst = max(worst, e / ref)
                print(f"  {t:5} decode/prompt {r['err_alone_vs_fp64_median']:.3e}/{r['err_batch_vs_fp64_median']:.3e}  x{e / ref:.3f}")
        print("  P2:", verdict(worst <= 1.05, worst > 1.15))

    kv = {(r["tag"].split("_", 1)[1], r["probe_flags"]): r for r in load(run / "close" / "kv_paths.jsonl") if r["cmp"] == "gen"}
    floor = kv.get(("Q6_K", "0"))
    if floor:
        f = floor["maxabs_median"]
        print(f"end to end: floor Q6_K {f:.3f} ({floor['argmax_flips']} flips)"
              + (f"; Q8_0 {kv[('Q8_0', '0')]['maxabs_median']:.3f}" if ("Q8_0", "0") in kv else "; Q8_0 not run (2 GB)"))
        for pid, fmts in (("P3 (consistent, FP32 sum)", ("Q4_0", "Q4_1", "Q5_1")), ("P4 (inconsistent, M4)", ("Q5_0", "Q4_K", "Q5_K"))):
            v = []
            for fm in fmts:
                u, g = kv.get((fm, "0")), kv.get((fm, "384"))
                if not u or not g:
                    print(f"  {fm}: missing"); continue
                ru, rg = u["maxabs_median"] / f, g["maxabs_median"] / f
                v.append(verdict(ru >= 1.5 and rg <= 1.25, ru <= 1.25 or rg > 1.25))
                extra = f" | I9 {kv[(fm, '128')]['maxabs_median'] / f:.2f}x" if (fm, "128") in kv else ""
                print(f"  {fm}: upstream {u['maxabs_median']:.3f} ({ru:.2f}x) -> G {g['maxabs_median']:.3f} ({rg:.2f}x){extra}  {v[-1]}")
            print(f"  {pid}:", "FALSIFIED" if "FALSIFIED" in v else "partial" if "partial" in v else "pass" if v else "n/a")
        m = kv.get(("Q4_K_M", "0")), kv.get(("Q4_K_M", "384"))
        if all(m):
            print(f"  recorded Q4_K_M: {m[0]['maxabs_median'] / f:.2f}x -> {m[1]['maxabs_median'] / f:.2f}x")

    ppl = {(r["tag"].split("_", 1)[1], r["probe_flags"]): r for r in load(run / "close" / "ppl_paths.jsonl")}
    a, b = ppl.get(("Q4_0", "0")), ppl.get(("Q5_0", "0"))
    if a and b:
        za, zb = a["dnll"] / a["dnll_se_chunk"], b["dnll"] / b["dnll_se_chunk"]
        print("P5 PPL decode - batch (z = dnll/se_chunk): Q4_0 |z| < 2, Q5_0 z >= 2 (falsified if Q4_0 z >= 2.5 or Q5_0 z < 1)")
        for fm, r, z in (("Q4_0", a, za), ("Q5_0", b, zb)):
            print(f"  {fm}: {100 * (r['ppl_decode'] / r['ppl_batch'] - 1):+.2f}% z {z:+.2f}")
        print("  P5:", verdict(abs(za) < 2 and zb >= 2, za >= 2.5 or zb < 1))
        for fm in ("Q4_0", "Q5_0", "Q4_1", "Q5_1", "Q4_K", "Q5_K", "Q4_K_M", "Q6_K"):
            u, g = ppl.get((fm, "0")), ppl.get((fm, "384"))
            if u:
                s = f"  recorded {fm}: z {u['dnll'] / u['dnll_se_chunk']:+.2f}"
                if g:
                    s += f" -> G z {g['dnll'] / g['dnll_se_chunk']:+.2f}"
                print(s)


if __name__ == "__main__":
    main()
