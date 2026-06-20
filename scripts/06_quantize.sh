#!/usr/bin/env bash
# Day 10-11: merge router adapter -> GGUF Q4_K_M for on-device inference.
# Usage: bash scripts/06_quantize.sh [ADAPTER_DIR]
set -e
ADAPTER=${1:-outputs/router-seed0}
mkdir -p outputs/gguf logs

python src/adaptivethink/quantize/export_gguf.py \
  --adapter "$ADAPTER" \
  --merged-dir outputs/router-merged \
  --out outputs/gguf/router-1p5b-Q4_K_M.gguf \
  --quant-type Q4_K_M 2>&1 | tee logs/quantize.log

echo "GGUF written to outputs/gguf/router-1p5b-Q4_K_M.gguf"
echo "On-device smoke test:"
echo "  python eval/run_benchmarks.py --gguf outputs/gguf/router-1p5b-Q4_K_M.gguf \\"
echo "    --verifier-ckpt outputs/verifier-400m/best.pt --route-mode threshold \\"
echo "    --benchmark gsm8k --n 50 --tag ondevice --out results/ondevice.json"
