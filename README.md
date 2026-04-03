# Guacamole-Optimized-Model

Guacamole-Optimized-Model is a collection of QLoRA-based finetuning and inference resources optimized for Guanaco-style instruction-following models.

## Project Overview

This repository contains training, evaluation, and inference code, notebooks, and adapter artifacts for QLoRA finetuning and evaluation. It provides reproducible examples, scripts for common workflows, and prebuilt adapter assets for inference.

## Repository Layout

- `qlora-base/` — Finetuning notebooks, training scripts, evaluation utilities, and example outputs.
- `qlora-inference/` — Inference scripts, adapter files, conversion tools, and benchmarking utilities.

## Quickstart

1. Create and activate a Python environment (recommended Python 3.10+):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install -r qlora-base/requirements.txt
pip install -r qlora-inference/requirements.txt
```

2. Browse the example notebooks to understand workflows:

- `qlora-base/qlora.ipynb` — QLoRA finetuning walkthrough
- `qlora-inference/qlora.ipynb` — Inference and adapter conversion examples

3. Run a sample generation script (adjust model and adapter paths):

```bash
python qlora-base/examples/guanaco_generate.py --model <BASE_MODEL> --adapter <ADAPTER_PATH>
```

## Finetuning

See `qlora-base/scripts/` for shell scripts targeting different model sizes (7B, 13B, 33B, 65B). The notebooks provide smaller-scale examples suitable for experimentation and debugging.

## Inference & Adapters

- Adapter artifacts and tokenizer metadata live under `qlora-inference/adapter/`.
- Use `qlora-inference/convert_adapter.py` to convert or adapt adapter formats as needed.

## Evaluation

Evaluation scripts and benchmark notebooks are under `qlora-base/eval/` and `qlora-inference/mmlu/`. Use these to reproduce MMLU-style and qualitative evaluations.

## Contributing

Issues and pull requests are welcome. When opening an issue, include environment details, exact commands, and steps to reproduce.

## License

Check the `LICENSE` files in subdirectories for licensing details pertaining to each component.

---

If you'd like, I can also:

- Add CI badges and a consolidated `requirements-dev.txt`
- Create a `USAGE.md` with Windows-specific runnable commands
- Add a minimal example that runs entirely on CPU for quick testing

