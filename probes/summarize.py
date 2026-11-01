#!/usr/bin/env python3
"""Tabulate probe output and score it against the predictions of the first experiments.

    python probes/summarize.py results/<run_dir> [more run dirs...]

Reads matmul_paths.jsonl / kv_paths.jsonl and prints Markdown tables plus a verdict
per prediction ID. It only reads raw data; it never edits results.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

M4_T4 = ["q4_0", "q5_0", "q4_K", "q5_K"]          # correction-term inconsistent on Turing+
CONSISTENT = ["q8_0", "q4_1", "q5_1", "q3_K", "q6_K"]
D4 = ["q8_0", "q5_0", "q3_K", "q6_K"]
DS4 = ["q4_0", "q4_1", "q5_1", "q4_K", "q5_K"]


def load(dirs: list[str], name: str) -> list[dict]:
    rows = []
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); r["_run"] = str(d); rows.append(r)
    return rows


def fmt(x: float) -> str:
    return "—" if x is None else f"{x:.2e}"


def matmul_report(rows: list[dict]) -> None:
    by = defaultdict(dict)  # (device, cc, K, N) -> {(type, flags): row}
    for r in rows:
        by[(r["device"], r["cc"], r["K"], r["N"])][(r["type"], r["probe_flags"])] = r
    for (dev, cc, K, N), cell in sorted(by.items()):
        flags = sorted({f for (_, f) in cell})
        types = [t for t in ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"]
                 if any((t, f) in cell for f in flags)]
        print(f"\n### Operator probe: {dev} (cc {cc}), K={K}, N={N}\n")
        print("δ = median relative L2 difference, token alone (decode path) vs. in a batch of 64 (prompt path).\n")
        print("| type | " + " | ".join(f"δ med/max flags={f}" for f in flags) + " | bitwise-equal (flags=0) | err vs FP64 alone / batch | switch point (δ vs batch by n) |")
        print("|---|" + "---|"*len(flags) + "---|---|---|")
        for t in types:
            r0 = cell.get((t, "0"))
            sw = ""
            if r0:
                sw = ", ".join(f"{k}:{v:.0e}" for k, v in r0["delta_vs_batchB_by_nb"].items())
            print(f"| {t} | " + " | ".join(fmt(cell[(t, f)]["delta_median"]) + " / " + fmt(cell[(t, f)]["delta_max"]) if (t, f) in cell else "—" for f in flags)
                  + f" | {r0['bitwise_equal_frac']:.3f} | {fmt(r0['err_alone_vs_fp64_median'])} / {fmt(r0['err_batch_vs_fp64_median'])} | {sw} |"
                  if r0 else "")
        d = lambda t, f="0": cell.get((t, f), {}).get("delta_median")
        print("\n**Predictions (operator level)**\n")
        pairs = [("q4_0", "q4_1"), ("q5_0", "q5_1"), ("q4_K", "q6_K"), ("q5_K", "q6_K")]
        if cc < 75:  # O1 is a Turing+ prediction; on DP4A builds Q4_0 is consistent (O6)
            pairs = [p for p in pairs if p[0] != "q4_0"]
            if d("q4_0") and d("q4_1"):
                x = d("q4_0") / d("q4_1")
                print(f"- **O6** (DP4A build) δ(q4_0)/δ(q4_1) = {x:.2f} → {'SUPPORTED' if x < 3 else ('FALSIFIED' if x >= 10 else 'PARTIAL')}")
        ratios = [(a, b, d(a) / d(b)) for a, b in pairs if d(a) and d(b)]
        if ratios:
            ok = all(x >= 10 for *_, x in ratios); bad = any(x < 3 for *_, x in ratios)
            print("- **O1** M4 pairs δ(inconsistent)/δ(consistent): " + ", ".join(f"{a}/{b} = {x:.1f}" for a, b, x in ratios)
                  + f" → {'SUPPORTED' if ok else ('FALSIFIED' if bad else 'PARTIAL')}")
        if all(d(t, "4") for t in M4_T4 if d(t)):
            m4 = M4_T4 if cc >= 75 else [t for t in M4_T4 if t != "q4_0"]  # Q4_0 is consistent on DP4A builds
            red = [(t, d(t) / d(t, "4")) for t in m4 if d(t) and d(t, "4")]
            print("- **O2** I4 reduction factor on M4 formats: " + ", ".join(f"{t} ×{x:.1f}" for t, x in red)
                  + f" → {'SUPPORTED' if red and all(x >= 3 for _, x in red) else 'NOT SUPPORTED'}")
        d4 = [d(t) for t in ["q8_0", "q6_K", "q3_K"] if d(t)]; ds4 = [d(t) for t in ["q4_1", "q5_1"] if d(t)]
        if d4 and ds4:
            i1 = [(t, d(t) / d(t, "1")) for t in ["q8_0", "q6_K", "q3_K"] if d(t) and d(t, "1")]
            print(f"- **O3** mean D4 δ / mean DS4 δ (consistent formats) = {sum(d4)/len(d4) / (sum(ds4)/len(ds4)):.2f}; "
                  "I1 reduction on D4: " + ", ".join(f"{t} ×{x:.2f}" for t, x in i1))
        if d("f16"):
            print(f"- **O5** F16 δ = {fmt(d('f16'))} (predicted range [1e-4, 3e-3])")


def kv_report(rows: list[dict]) -> None:
    by = defaultdict(dict)
    for r in rows:
        by[r["_run"]][(r["tag"], r["probe_flags"], r["cmp"])] = r
    for run, cell in by.items():
        tags = []
        for (t, _, _) in cell:
            if t not in tags:
                tags.append(t)
        print(f"\n### End-to-end probe ({run})\n")
        print("Next-token logits, same tokens and positions. maxabs = median over positions of max |Δlogit|; "
              "flips = argmax disagreements / positions.\n")
        print("| model | flags | gen maxabs | gen flips | gen bitwise-equal | suffix after decode-built prefix | suffix after prompt-built prefix | E2 ratio | decode ms/token (median) |")
        print("|---|---|---|---|---|---|---|---|---|")
        for t in tags:
            for f in sorted({f for (tt, f, _) in cell if tt == t}):
                g = cell.get((t, f, "gen")); sd = cell.get((t, f, "suf_dec")); sp = cell.get((t, f, "suf_pmt"))
                if not g:
                    continue
                ratio = sd["maxabs_median"] / sp["maxabs_median"] if sd and sp and sp["maxabs_median"] > 0 else float("inf")
                print(f"| {t} | {f} | {fmt(g['maxabs_median'])} | {g['argmax_flips']}/{g['positions']} | {g['bitwise_equal_positions']}/{g['positions']} | "
                      f"{fmt(sd['maxabs_median'])} ({sd['argmax_flips']}/{sd['positions']}) | "
                      f"{fmt(sp['maxabs_median'])} ({sp['argmax_flips']}/{sp['positions']}) | {ratio:.1f} | {g.get('decode_ms_median', float('nan')):.2f} |")


def gate_report(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["device"], r["cc"])].append(r)
    for (dev, cc), rs in by.items():
        print(f"\n### Gate probe (M5): {dev} (cc {cc})\n")
        print("Inputs perturbed by a relative ε, pushed through ONE kernel path. A smooth path changes its output by ≈ε and is "
              "never bit-identical; a gate is mostly bit-identical with rare large jumps.\n")
        eps = sorted({r["eps"] for r in rs})
        print("| type | path | " + " | ".join(f"ε={e:.0e}: same / med / max" for e in eps) + " |")
        print("|---|---|" + "---|"*len(eps))
        cell = {(r["type"], r["path"], r["eps"]): r for r in rs}
        for t in dict.fromkeys(r["type"] for r in rs):
            for path in ("alone", "batch"):
                print(f"| {t} | {path} | " + " | ".join(
                    (lambda r: f"{r['bitwise_same_frac']:.2f} / {r['rel_change_median']:.1e} / {r['rel_change_max']:.1e}")(cell[(t, path, e)])
                    if (t, path, e) in cell else "—" for e in eps) + " |")


def ppl_report(rows: list[dict]) -> None:
    print("\n### Perplexity through the prompt path vs the decode path (wikitext-2 test)\n")
    print("Same text, same model. 'batch' = how quantization benchmarks score text (MMQ); 'decode' = how tokens are generated "
          "(MMVQ). Δ = decode − batch.\n")
    print("| model | flags | scored tokens | PPL batch | PPL decode | Δ PPL | mean abs Δ log p | argmax flips |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['tag']} | {r['probe_flags']} | {r['scored_tokens']} | {r['ppl_batch']:.4f} | {r['ppl_decode']:.4f} | "
              f"{r['ppl_decode'] - r['ppl_batch']:+.4f} | {r['mean_abs_dlogp']:.2e} | {r['argmax_flips']} |")


def main() -> None:
    dirs = sys.argv[1:]
    mm = load(dirs, "matmul_paths.jsonl")
    kv = load(dirs, "kv_paths.jsonl")
    gt = load(dirs, "gate.jsonl")
    pp = load(dirs, "ppl_paths.jsonl")
    if mm:
        matmul_report(mm)
    if gt:
        gate_report(gt)
    if kv:
        kv_report(kv)
    if pp:
        ppl_report(pp)
    if not mm and not kv and not gt and not pp:
        print("no data in", dirs)


if __name__ == "__main__":
    main()
