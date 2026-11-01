#!/usr/bin/env bash
# Experiment 4i: the Pascal (sm_61) numerics run, on a 2 GB GeForce MX330. Numerics only: a power-limited
# laptop GPU gives no meaningful timings. Writes the layout that probes/score_4i.py reads:
#   $WORK/runs/pascal/operator/matmul_paths.jsonl, $WORK/runs/pascal/close/{kv_paths,ppl_paths}.jsonl
#
#   WORK=/path/with/space bash runners/pascal_sm61.sh
#   python probes/score_4i.py $WORK/runs/pascal
#
# The published run (results/mx330-sm61-20261029) was executed in stages because of interruptions;
# this script runs the same steps with the same settings in one go. On 2 GB the F16 and Q8_0 models do not
# fit, so the F16 file is used only as the quantization source (--no-f16) and Q6_K is the floor. Each
# decode-path perplexity run takes hours on this GPU, so perplexity runs only for Q4_0 and Q5_0.
set -uo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$PWD/work}
RUN=$WORK/runs/pascal
PY=${PY:-python}
BIN=build/probes-sm61
F16=$WORK/models/qwen2.5-1.5b-instruct-fp16.gguf
URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-fp16.gguf
mkdir -p "$RUN" "$(dirname "$F16")"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] build sm_61 (native: DP4A MMQ, no tensor-core MMA, no FAST_FP16)"
cmake -S probes -B "$BIN" -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release > "$RUN/build-cfg.log" 2>&1 &&
    cmake --build "$BIN" --target matmul_paths kv_paths ppl_paths quantize_pure -j3 > "$RUN/build.log" 2>&1
rc=$?; echo "[$(stamp)] build rc=$rc"; [ $rc -eq 0 ] || exit 1

if [ ! -s "$F16" ]; then
    echo "[$(stamp)] download Qwen2.5-1.5B-Instruct F16 (3.6 GB)"
    wget -q -O "$F16.part" "$URL" && mv "$F16.part" "$F16" || { echo "download failed"; exit 1; }
fi

echo "[$(stamp)] operator probe (P1, P2): upstream, I4, I9, I10, G"
mkdir -p "$RUN/operator"
for fl in 0 4 128 256 384; do
    GGML_PROBE_FLAGS=$fl "$BIN/matmul_paths" --K 1536 --N 1536 --B 64 --tokens 16 --out "$RUN/operator/matmul_paths.jsonl" \
        > /dev/null 2>> "$RUN/operator/stderr.log"
    echo "[$(stamp)] matmul_paths flags=$fl rc=$?"
done

echo "[$(stamp)] end to end (P3, P4) for every format; perplexity (P5) for Q4_0 and Q5_0"
$PY probes/kaggle_job.py --local --plan close --no-f16 --ppl-formats Q4_0,Q5_0 --bin "$BIN" --model "$F16" \
    --out "$RUN/close" --formats Q6_K,Q4_0,Q5_0,Q4_1,Q5_1,Q4_K,Q5_K --mixes Q4_K --tag-prefix P15_ \
    > "$RUN/close.stdout" 2>&1
echo "[$(stamp)] close plan rc=$?"
rm -f "$RUN"/close/models/model-*.gguf
echo "[$(stamp)] done; score with: $PY probes/score_4i.py $RUN"
