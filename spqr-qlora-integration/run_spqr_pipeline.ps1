param(
    [string]$MergedModelPath = "./artifacts/merged-llama3-8b",
    [string]$SpqrOutDir = "./artifacts/spqr-legacy",
    [string]$SpqrPtPath = "./artifacts/spqr-model.pt",
    [string]$Dataset = "pajama"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "./SpQR/main.py")) {
    throw "Run this script from repo root (where SpQR/main.py exists)."
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SpqrOutDir) | Out-Null

Write-Host "[1/2] Running SpQR quantization to legacy format..."
python ./SpQR/main.py $MergedModelPath $Dataset `
    --wbits 4 `
    --groupsize 16 `
    --perchannel `
    --qq_scale_bits 3 `
    --qq_zero_bits 3 `
    --qq_groupsize 16 `
    --outlier_threshold 0.2 `
    --permutation_order act_order `
    --percdamp 1.0 `
    --nsamples 128 `
    --save $SpqrOutDir

Write-Host "[2/2] Converting legacy SpQR output to .pt model for fast inference..."
python ./SpQR/convert_legacy_model_format.py `
    --base_model $MergedModelPath `
    --legacy_model_path $SpqrOutDir `
    --sparse_strategy optimize_latency `
    --save_pt $SpqrPtPath

Write-Host "Done. SpQR model path: $SpqrPtPath"
