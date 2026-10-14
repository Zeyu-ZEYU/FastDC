#!/usr/bin/env bash
# Runs every component once at a small scale on one GPU (about 15 minutes with a warm cache):
# unit tests, offline stage, reference engine, F_p sweep, kernels, vLLM integration, serving harness and
# prefill-decode disaggregation. Usage: scripts/functional_check.sh [GPU index] [output directory]
set -euo pipefail
GPU=${1:-0}
OUT=${2:-results/functional_check}
MODEL=${FASTDC_CHECK_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
PORT=${FASTDC_CHECK_PORT:-8000}  # load balancer / proxy port; instances use the next hundreds
export CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
step() { printf '\n=== %s\n' "$1"; }

step "unit tests (core, reference engine, kernels, vLLM integration, serving harness)"
python -m pytest -q "$HERE/tests"

step "offline stage: calibration and plans"
fastdc calibrate --model "$MODEL" --out "$OUT/calib.safetensors" --seq-len 2048
fastdc plan --calibration "$OUT/calib.safetensors" --method fastdc --r 0.01 --out "$OUT/fastdc.safetensors"
fastdc plan --calibration "$OUT/calib.safetensors" --method palu --p 0.2 --out "$OUT/palu.safetensors"
fastdc plan --calibration "$OUT/calib.safetensors" --method full --out "$OUT/full.safetensors"

step "reference engine: generation and evaluation"
fastdc generate --model "$MODEL" --plan "$OUT/fastdc.safetensors" --prompt "Name three primary colors." --max-new-tokens 32
fastdc evaluate --model "$MODEL" --plan "$OUT/fastdc.safetensors" --task gsm8k --limit 2 --task-args '{"limit": 20}'

step "F_p table and selection"
fastdc sweep --model "$MODEL" --calibration "$OUT/calib.safetensors" --task ruler --limit 4 \
    --task-args '{"limit": 40, "context_tokens": 1024}' --grid 0.002,0.005 --out "$OUT/table_ruler.json"
fastdc select --table "$OUT/table_ruler.json" --threshold 0.5 --calibration "$OUT/calib.safetensors" \
    --out "$OUT/selected.safetensors"

step "kernels: full vs compressed widths, balanced vs unbalanced schedules"
python "$HERE/benchmarks/kernels/attention_bench.py" --plan "$OUT/fastdc.safetensors" --layers 4 --batch 8 \
    --decode-context 2048 --prefill 1024 --mixed 512 --warmup 2 --iters 5 --out "$OUT/kernels"

step "vLLM: offline generation with the FastDC backend"
python "$HERE/examples/vllm_generate.py" --model "$MODEL" --plan "$OUT/fastdc.safetensors" --max-tokens 24

step "serving harness: load balancer, Poisson arrivals"
python "$HERE/benchmarks/serving/run.py" --model "$MODEL" --gpus "$GPU" --task gsm8k --task-args '{"limit": 40}' \
    --num-requests 16 --rates 4 --output-len 32 --max-model-len 2048 --gpu-memory-utilization 0.5 \
    --method stock --method "fastdc=$OUT/fastdc.safetensors" --method "palu=$OUT/palu.safetensors" \
    --port-base "$PORT" --out "$OUT/serving"

step "prefill-decode disaggregation with the KV connector"
python -m fastdc.disagg.launch --model "$MODEL" --plan "$OUT/fastdc.safetensors" --prefill-gpus "$GPU" \
    --decode-gpus "$GPU" --max-model-len 2048 --gpu-memory-utilization 0.35 --port-base "$PORT" \
    --run-dir "$OUT/disagg" --smoke 4

step "done: outputs in $OUT"
