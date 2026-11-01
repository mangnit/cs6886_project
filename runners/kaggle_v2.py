#!/usr/bin/env python3
"""Ship the probe bundle to Kaggle, run the night job, fetch raw results.

    python runners/kaggle_v2.py stage --build build/probes-kaggle
    python runners/kaggle_v2.py push  --name night1
    python runners/kaggle_v2.py status --name night1
    python runners/kaggle_v2.py fetch  --name night1

Uses the `kaggle` CLI with your default credentials. The
token is never read, printed, or written anywhere by this script. The accelerator
cannot be pinned through the API, so whichever GPU is granted (T4 or P100) is
recorded by the job itself in env.json.

The bundle is binaries built locally with the project nvcc 12.6 for sm_60 and sm_75,
so the compiler is identical to the paper's CUDA 12.6 and to the laptop build.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGING = REPO / ".kaggle_staging"
DATASET = "cs6886-v2-probes"
KERNEL_PREFIX = "cs6886-v2"
KAGGLE = shutil.which("kaggle") or str(Path.home() / "miniconda3" / "bin" / "kaggle")


def run(cmd: list[str], check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{out}")
    return out


def user(verbose: bool = False) -> str:
    """The account the active credentials belong to.

    `kaggle config view` is not trusted on its own: with a token passed in the environment it can
    still report another account's name. The owner of this account's own kernels/datasets, as
    returned by the API under the active token, is authoritative; config view is a fallback.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for cmd in (["kernels", "list", "--mine", "--csv", "--page-size", "20"], ["datasets", "list", "--mine", "--csv"]):
        out = run([KAGGLE] + cmd, check=False)
        owners = {l.split(",")[0].split("/")[0] for l in out.splitlines()[1:] if "/" in l.split(",")[0]}
        if len(owners) == 1:
            if verbose:
                print(f"[whoami] from API ({cmd[0]} --mine): {next(iter(owners))}")
            return owners.pop()
    for line in run([KAGGLE, "config", "view"]).splitlines():
        if "username" in line:
            name = line.split(":", 1)[1].strip()
            if verbose:
                print(f"[whoami] from config view (unverified: account owns nothing yet): {name}")
            return name
    raise SystemExit("cannot determine the Kaggle username")


def stage(a) -> None:
    owner = user()
    d = STAGING / f"{owner}__{a.dataset}"  # one folder per account: parallel stages must never share one
    shutil.rmtree(d, ignore_errors=True)
    (d / "bin").mkdir(parents=True)
    if a.bundle:  # an extracted bundle/probes-bundle-*.tar.gz (bin/ + patch): no local build needed (cloud)
        for f in (Path(a.bundle) / "bin").iterdir():
            shutil.copy2(f, d / "bin" / f.name)
        stage_finish(a, d, owner)
        return
    build = Path(a.build)
    for exe in ("matmul_paths", "kv_paths", "quantize_pure", "gate_probe", "ppl_paths", "kv_bisect", "real_replay"):
        shutil.copy2(build / exe, d / "bin" / exe)
    if (build / "bin" / "llama-server").exists():  # experiment 4g: the prior work's cache_prompt workload
        shutil.copy2(build / "bin" / "llama-server", d / "bin" / "llama-server")
    # Ship each shared library once, under the soname the executables load (libX.so.N),
    # resolving the symlink; the unversioned and fully-versioned names are not needed.
    import re
    for so in build.rglob("*.so.*"):
        if re.fullmatch(r"lib.+\.so\.\d+", so.name):
            shutil.copy2(so.resolve(), d / "bin" / so.name)
    # Libraries built without a version (llama-server's libllama-server-impl.so) are loaded by that name.
    for so in (build / "bin").glob("lib*.so"):
        if not so.is_symlink():
            shutil.copy2(so, d / "bin" / so.name)
    stage_finish(a, d, owner)


def stage_finish(a, d: Path, owner: str) -> None:
    shutil.copy2(REPO / "probes" / "probe_interventions_b10434.patch", d / "probe_interventions_b10434.patch")
    ref = f"{owner}/{a.dataset}"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": a.dataset, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"[stage] {len(list((d / 'bin').iterdir()))} files, {size/1e6:.0f} MB -> {ref}")
    exists = a.dataset in run([KAGGLE, "datasets", "list", "--mine", "--csv"], check=False)
    if exists:
        print(run([KAGGLE, "datasets", "version", "-p", str(d), "-m", "update", "--dir-mode", "zip"]))
    else:
        print(run([KAGGLE, "datasets", "create", "-p", str(d), "--dir-mode", "zip"]))
    for _ in range(60):  # Kaggle processes uploads asynchronously
        st = run([KAGGLE, "datasets", "status", ref], check=False).strip()
        print(f"[stage] status: {st}")
        if "ready" in st.lower():
            return
        time.sleep(20)
    raise SystemExit("dataset never became ready")


def push(a) -> None:
    u = user()
    slug = f"{KERNEL_PREFIX}-{a.name}"
    d = STAGING / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    job = (REPO / "probes" / "kaggle_job.py").read_text()
    if a.job_args:  # bake the arguments in: a Kaggle script kernel runs with no argv
        job = job.replace('if __name__ == "__main__":\n', 'if __name__ == "__main__":\n    sys.argv += ' + repr(a.job_args.split()) + '\n', 1)
    (d / "run.py").write_text(job)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{u}/{a.dataset}"], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    cmd = [KAGGLE, "kernels", "push", "-p", str(d)]
    if a.accelerator:  # e.g. NvidiaTeslaP100; the job still records which GPU it really got
        cmd += ["--accelerator", a.accelerator]
    print(run(cmd))


def status(a) -> None:
    print(run([KAGGLE, "kernels", "status", f"{user()}/{KERNEL_PREFIX}-{a.name}"], check=False))


INCOMING = Path(os.environ.get("KAGGLE_INCOMING", REPO / ".kaggle_incoming"))  # raw job output, before filtering


def fetch(a) -> None:
    """Download a job's output to a scratch folder, then copy only real result files into results/.

    Older jobs return ~290 MB of binaries, and Kaggle's output endpoint intermittently
    serves an HTML '502 Server Error' page in place of a file. Neither may reach results/,
    which holds raw data only. A file is accepted only if it does not start with '<'.
    """
    stage = INCOMING / a.name
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    print(run([KAGGLE, "kernels", "output", f"{user()}/{KERNEL_PREFIX}-{a.name}", "-p", str(stage)], check=False)[-500:])
    out = REPO / "results" / f"kaggle-{a.name}"
    bad = 0
    for f in stage.rglob("*"):
        if not f.is_file() or "bin" in f.relative_to(stage).parts:
            continue
        with open(f, "rb") as fh:
            if fh.read(1) == b"<":
                print(f"[fetch] REJECTED (server error page): {f.relative_to(stage)}")
                bad += 1
                continue
        dst = out / f.relative_to(stage)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    print(f"[fetch] -> {out}  rejected={bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--build", default=""); s.add_argument("--bundle", default="")
    s.set_defaults(f=stage)
    s.add_argument("--dataset", default=DATASET, help="use a NEW slug per bundle: a re-versioned dataset can mount stale")
    for name, f in (("push", push), ("status", status), ("fetch", fetch)):
        p = sub.add_parser(name); p.add_argument("--name", required=True); p.set_defaults(f=f)
        if name == "push":
            p.add_argument("--job-args", default="")
            p.add_argument("--dataset", default=DATASET)
            p.add_argument("--accelerator", default="")
    w = sub.add_parser("whoami"); w.set_defaults(f=lambda a: print(user(verbose=True)))
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
