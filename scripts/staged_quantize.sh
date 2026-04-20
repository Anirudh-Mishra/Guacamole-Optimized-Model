#!/usr/bin/env bash
# Staged quantization runner: tries increasing nsamples and larger max_split_size_mb
# Usage: bash scripts/staged_quantize.sh

MODEL=./artifacts/merged-llama3-8b
DATA=./SpQR/data/red_pajama_n=1024.pth
SAVE_DIR_BASE=./artifacts
GROUPSIZE=128
WBITS=3
OFFLOAD="--offload_activations"

SAMPLES=(2 4 8 16 32 64 128)
SPLITS=(2048 4096 8192)

mkdir -p "$SAVE_DIR_BASE"

for ns in "${SAMPLES[@]}"; do
  echo "\n=== Trying nsamples=${ns} ==="
  success=0
  for split in "${SPLITS[@]}"; do
    export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:${split},expandable_segments:True"
    LOG="${SAVE_DIR_BASE}/spqr-legacy-n${ns}-split${split}.log"
    OUTDIR="${SAVE_DIR_BASE}/spqr-legacy-n${ns}"
    mkdir -p "$OUTDIR"
    echo "  Trying split=${split}, log=${LOG}"

    python SpQR/main.py "$MODEL" "$DATA" --nsamples "$ns" $OFFLOAD --wbits "$WBITS" --groupsize "$GROUPSIZE" --save "$OUTDIR" > "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "  Success ns=${ns} split=${split}"
      success=1
      break
    else
      echo "  Failed ns=${ns} split=${split}, rc=$rc (see $LOG)"
      # small pause to let GPU fully free
      sleep 2
    fi
  done
  if [ $success -eq 0 ]; then
    echo "All split sizes failed for ns=${ns}, stopping." >&2
    exit 2
  fi
done

echo "Staged quantization completed."
