import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_base_model(adapter_dir: Path, explicit_base: str | None) -> str:
    if explicit_base:
        return explicit_base

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    return base


def parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower().strip()
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def normalize_adapter_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize extra nested PEFT prefixes produced by some training stacks."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        while "base_model.model.model.model." in new_key:
            new_key = new_key.replace("base_model.model.model.model.", "base_model.model.model.")
        normalized[new_key] = value
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT/QLoRA adapter into a dense HF model")
    parser.add_argument("--adapter_dir", type=Path, required=True, help="Path with adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where merged model should be saved")
    parser.add_argument("--base_model", type=str, default=None, help="Optional explicit base model path/name")
    parser.add_argument("--dtype", type=str, default="float16", help="float16 | bfloat16 | float32")
    parser.add_argument("--device_map", type=str, default="cpu", help="HF device map, e.g. cpu, auto, or cuda:0")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = resolve_base_model(adapter_dir, args.base_model)
    dtype = parse_dtype(args.dtype)
    device_map = None if args.device_map.lower() in {"none", "null", ""} else args.device_map

    print(f"[merge] base model: {base_model}")
    print(f"[merge] adapter dir: {adapter_dir}")
    print(f"[merge] output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    except KeyError as e:
        # Some adapter exports add an extra '.model' nesting level in keys.
        if "base_model.model.model.model" not in str(e):
            raise

        print("[merge] adapter key prefix mismatch detected; retrying with normalized keys")
        peft_config = PeftConfig.from_pretrained(str(adapter_dir))
        peft_model = get_peft_model(base, peft_config)
        adapter_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
        adapter_weights = normalize_adapter_keys(adapter_weights)
        set_peft_model_state_dict(peft_model, adapter_weights, adapter_name="default")

    merged = peft_model.merge_and_unload()

    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    print("[merge] merged model + tokenizer saved")


if __name__ == "__main__":
    main()
