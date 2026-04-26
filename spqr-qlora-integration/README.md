# QLoRA -> SpQR Integration (Deployment Path)

This folder gives you a direct path from your trained QLoRA adapter to a SpQR inference artifact.

## What we detected in repo root

- `adapter_config.json`
- `adapter_model.safetensors`
- `model.safetensors`
- `model.safetensors.index.json`

Your adapter config points to base model `meta-llama/Meta-Llama-3-8B`.

## Step 0: install dependencies

Use an environment that already has your training deps, then ensure SpQR deps are installed too.

```powershell
pip install -r ./qlora-base/requirements.txt
pip install -r ./SpQR/requirements.txt
```

## Step 1: merge adapter into dense model

Run from repo root:

```powershell
python ./spqr-qlora-integration/merge_lora_to_dense.py `
  --adapter_dir . `
  --output_dir ./artifacts/merged-llama3-8b `
  --dtype float16
```

If your base model is local, pass it explicitly:

```powershell
python ./spqr-qlora-integration/merge_lora_to_dense.py `
  --adapter_dir . `
  --output_dir ./artifacts/merged-llama3-8b `
  --base_model D:/path/to/local/Meta-Llama-3-8B
```

## Step 2: SpQR quantize + convert for inference

```powershell
./spqr-qlora-integration/run_spqr_pipeline.ps1 `
  -MergedModelPath ./artifacts/merged-llama3-8b `
  -SpqrOutDir ./artifacts/spqr-legacy `
  -SpqrPtPath ./artifacts/spqr-model.pt `
  -Dataset pajama
```

## Step 3: run inference demo on SpQR model

```powershell
python ./SpQR/inference_demo.py `
  --pretrained_model_path ./artifacts/merged-llama3-8b `
  --compressed_model_path ./artifacts/spqr-model.pt `
  --execution_mode 2
```

## Notes

- This is deployment integration (post-training). No additional finetuning is required.
- If quality drops too much, retrain LoRA, then repeat merge -> SpQR.
- Keep original adapter files as rollback artifacts.
