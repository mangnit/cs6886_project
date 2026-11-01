#!/usr/bin/env python3
"""Score the episode, cost and BF16 runs against their pass/fail criteria (experiment 4g).

    python probes/score_4g.py results/kaggle-<run>/results [more result dirs ...]

Episodes (H1-H5): an episode diverges if any generated token ID differs between the cache arm
(cache_prompt=true) and the recompute arm in lockstep, or the request counts differ (the prior
work's strict definition). Cost (K1-K3) and BF16 (BF1-BF2) are scored when their files are present.
Raw JSONL in, verdicts out; thresholds are fixed in this file.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def load(d: Path, name: str) -> list[dict]:
    p = d / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def episodes(dirs: list[Path]) -> None:
    runs: dict = defaultdict(lambda: defaultdict(dict))  # (tag, flags, arm) -> episode -> request -> row
    for d in dirs:
        for r in load(d, "episodes.jsonl"):
            runs[(r["tag"], r["probe_flags"], r["arm"])][r["episode"]][r["request"]] = r
    if not runs:
        return

    def compare(a: dict, b: dict) -> dict:
        """Per episode: divergent?, first divergence (request, position, generated tokens before it)."""
        res = {}
        for ep in sorted(set(a) & set(b)):
            ra, rb = a[ep], b[ep]
            div, before, where = False, 0, None
            for k in range(max(len(ra), len(rb))):
                if k not in ra or k not in rb:
                    div, where = True, (k, 0); break
                ta, tb = ra[k]["tokens"], rb[k]["tokens"]
                j = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
                if j is None and len(ta) != len(tb):
                    j = min(len(ta), len(tb))
                if j is not None:
                    div, where = True, (k, j); before += j; break
                before += len(ta)
            res[ep] = {"div": div, "where": where, "before": before}
        return res

    div = {}  # (model, fmt, flags) -> per-episode comparison
    print("Episode divergence, cache arm vs recompute arm (strict, token IDs). Wilson 95% CI.")
    print("Cache check: median fraction of the prompt evaluated on requests 2-4 (cache arm should be well below 1).")
    for (tag, fl, arm) in sorted(runs):
        if arm != "cache" or tag.endswith("_rep") or (tag, fl, "recompute") not in runs:
            continue
        m, fmt = tag.split("_", 1)
        c = compare(runs[(tag, fl, "cache")], runs[(tag, fl, "recompute")])
        div[(m, fmt, fl)] = c
        n, k = len(c), sum(v["div"] for v in c.values())
        lo, hi = wilson(k, n)
        frac = {}
        for a in ("cache", "recompute"):
            fr = [row["prompt_n"] / (row["prompt_n"] + (row.get("cache_n") or 0))
                  for ep in runs[(tag, fl, a)].values() for kk, row in ep.items()
                  if kk > 0 and row.get("prompt_n") is not None and (row["prompt_n"] + (row.get("cache_n") or 0)) > 0]
            frac[a] = st.median(fr) if fr else float("nan")
        bef = [v["before"] for v in c.values() if v["div"]]
        print(f"  {m:4} {fmt:7} flags {fl:>3}: {k:2}/{n} = {100 * k / max(n, 1):5.1f}% [{100 * lo:4.1f}, {100 * hi:4.1f}]"
              f"  median generated tokens before divergence {st.median(bef) if bef else '-':>6}"
              f"  prompt evaluated: cache {frac['cache']:.2f}, recompute {frac['recompute']:.2f}")

    def rate(m: str, fmt: str, fl: str) -> float | None:
        c = div.get((m, fmt, fl))
        return 100 * sum(v["div"] for v in c.values()) / len(c) if c else None

    models = sorted({m for (m, _, _) in div})
    verdict = {}
    print("\nH1 upstream amplification: Q4_K_M and Q4_0 >= Q8_0 + 10 pp (falsified if neither exceeds Q8_0)")
    v = []
    for m in models:
        q8, a, b = rate(m, "Q8_0", "0"), rate(m, "Q4_K_M", "0"), rate(m, "Q4_0", "0")
        if None in (q8, a, b):
            continue
        ok = a >= q8 + 10 and b >= q8 + 10
        fals = a <= q8 and b <= q8
        v.append("FALSIFIED" if fals else "pass" if ok else "partial")
        print(f"  {m}: Q8_0 {q8:.1f}%  Q4_K_M {a:.1f}%  Q4_0 {b:.1f}%  {v[-1]}")
    verdict["H1"] = v
    print("H2 under G: Q4_K_M and Q4_0 <= Q8_0 + 10 pp (falsified if any > Q8_0 + 20 pp)")
    v = []
    for m in models:
        q8 = rate(m, "Q8_0", "0")
        for fmt in ("Q4_K_M", "Q4_0"):
            g = rate(m, fmt, "384")
            if q8 is None or g is None:
                continue
            v.append("pass" if g <= q8 + 10 else "FALSIFIED" if g > q8 + 20 else "partial")
            print(f"  {m} {fmt}: G {g:.1f}% vs Q8_0 {q8:.1f}%  {v[-1]}")
    verdict["H2"] = v
    print("H3 paired: discordant episodes favour G; pooled exact McNemar p < 0.01 (falsified if not favouring G or p >= 0.05)")
    B = C = 0
    for m in models:
        for fmt in ("Q4_K_M", "Q4_0"):
            u, g = div.get((m, fmt, "0")), div.get((m, fmt, "384"))
            if not u or not g:
                continue
            eps = set(u) & set(g)
            b = sum(u[e]["div"] and not g[e]["div"] for e in eps)
            c = sum(g[e]["div"] and not u[e]["div"] for e in eps)
            B, C = B + b, C + c
            print(f"  {m} {fmt}: upstream-only {b}, G-only {c}, p {mcnemar_exact(b, c):.3g}")
    if B + C:
        p = mcnemar_exact(B, C)
        h3 = "pass" if B > C and p < 0.01 else "FALSIFIED" if B <= C or p >= 0.05 else "partial"
        print(f"  pooled: upstream-only {B}, G-only {C}, exact McNemar p {p:.3g}  {h3}")
        verdict["H3"] = [h3]
    print("H4 (recorded): F16 rate < Q8_0 rate on the 3B models")
    for m in models:
        f, q8 = rate(m, "F16", "0"), rate(m, "Q8_0", "0")
        if f is not None and q8 is not None:
            print(f"  {m}: F16 {f:.1f}% vs Q8_0 {q8:.1f}%  {'holds' if f < q8 else 'does not hold'}")
    print("H5 run to run: the repeated Q4_K_M upstream cache pass reproduces every token")
    for (tag, fl, arm) in sorted(runs):
        if tag.endswith("_rep"):
            base = runs.get((tag[:-4], fl, arm))
            if base:
                c = compare(runs[(tag, fl, arm)], base)
                k = sum(v["div"] for v in c.values())
                verdict.setdefault("H5", []).append("pass" if k == 0 else "FALSIFIED")
                print(f"  {tag}: {k}/{len(c)} episodes differ  {verdict['H5'][-1]}")
    print("verdicts:", {h: ("FALSIFIED" if "FALSIFIED" in vs else "partial" if "partial" in vs else "pass") if vs else "n/a"
                        for h, vs in verdict.items()})


def cost_bf16(dirs: list[Path]) -> None:
    lat = [r for d in dirs for r in load(d, "latency.jsonl") if r["cmp"] == "gen"]
    if lat:
        print("\nCost of G (ABBA; median [IQR] over repeats, each = median over 8 prompts)")

        def series(fmt: str, fl: str, key: str) -> list[float]:
            return [r[key] for r in lat if r["tag"].startswith(f"LAT_{fmt}_") and r["probe_flags"] == fl]

        def ch(fmt: str, fl: str, key: str) -> float | None:
            b, t = series(fmt, "0", key), series(fmt, fl, key)
            if len(b) < 2 or len(t) < 2:
                return None
            qb, qt = st.quantiles(b, n=4, method="inclusive"), st.quantiles(t, n=4, method="inclusive")
            c = 100 * (st.median(t) / st.median(b) - 1)
            print(f"  {fmt} flags {fl} {key[:-10]}: {st.median(b):.3f} [{qb[0]:.3f}, {qb[2]:.3f}] -> "
                  f"{st.median(t):.3f} [{qt[0]:.3f}, {qt[2]:.3f}] ms ({c:+.1f}%)")
            return c
        k1 = [ch(f, "384", "decode_ms_median") for f in ("Q4_0", "Q5_0", "Q4_1")]
        i4 = ch("Q4_0", "4", "decode_ms_median")
        k3 = [ch(f, "384", "prefill_ms_median") for f in ("Q4_0", "Q4_K")]
        if None not in k1:
            print("  K1:", "pass" if max(k1) <= 2 else "FALSIFIED" if max(k1) > 4 else "partial")
        if i4 is not None and k1[0] is not None:
            print(f"  K2: I4 {i4:+.1f}% vs G {k1[0]:+.1f}% on Q4_0 ->", "pass" if i4 - k1[0] >= 2 else "FALSIFIED")
        if None not in k3:
            print("  K3:", "pass" if max(abs(x) for x in k3) <= 3 else "FALSIFIED" if max(abs(x) for x in k3) > 5 else "partial")
    mm = {(r["type"], r["probe_flags"]): r for d in dirs for r in load(d, "matmul_paths.jsonl") if r["K"] == 1536}
    if ("q8_0", "512") in mm:
        print("\nBF1 operator: I11 raises M1-format delta >= 4x (falsified if < 2x)")
        rs = []
        for t in ("q8_0", "q6_K", "q3_K"):
            a, b = mm[(t, "0")]["delta_median"], mm[(t, "512")]["delta_median"]
            rs.append(b / a)
            print(f"  {t}: {a:.2e} -> {b:.2e} (x{b / a:.1f})")
        print("  BF1:", "pass" if min(rs) >= 4 else "FALSIFIED" if min(rs) < 2 else "partial")
        print("  recorded: error vs FP64 (decode / prompt) at flags 0 | 512 | 384 | 896")
        for t in ("q4_0", "q5_0", "q4_1", "q5_1", "q4_K", "q5_K", "q8_0"):
            cells = [f"{mm[(t, f)]['err_alone_vs_fp64_median']:.3e}/{mm[(t, f)]['err_batch_vs_fp64_median']:.3e}"
                     for f in ("0", "512", "384", "896") if (t, f) in mm]
            print(f"  {t:5} " + " | ".join(cells))
    kv = {(r["tag"], r["probe_flags"]): r for d in dirs for r in load(d, "kv_paths.jsonl") if r["cmp"] == "gen"}
    if ("BF_Q8_0", "512") in kv:
        print("\nBF2 end to end: Q8_0 and Q6_K >= 1.25x upstream with I11 (falsified if both < 1.1x)")
        rs = []
        for f in ("Q8_0", "Q6_K"):
            a, b = kv[(f"BF_{f}", "0")], kv[(f"BF_{f}", "512")]
            rs.append(b["maxabs_median"] / a["maxabs_median"])
            print(f"  {f}: {a['maxabs_median']:.3f} ({a['argmax_flips']}) -> {b['maxabs_median']:.3f} ({b['argmax_flips']})  x{rs[-1]:.2f}")
        print("  BF2:", "pass" if min(rs) >= 1.25 else "FALSIFIED" if max(rs) < 1.1 else "partial")
        for fl in ("0", "512", "384", "896"):
            r = kv.get(("BF_Q4_0", fl))
            if r:
                print(f"  recorded Q4_0 flags {fl}: {r['maxabs_median']:.3f} ({r['argmax_flips']})")


if __name__ == "__main__":
    ds = [Path(a) for a in sys.argv[1:]]
    episodes(ds)
    cost_bf16(ds)
