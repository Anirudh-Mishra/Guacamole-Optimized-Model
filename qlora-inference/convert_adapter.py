import torch
import numpy as np
import json
from safetensors.torch import load_file
from pathlib import Path

# Load the PEFT adapter weights
adapter_weights = load_file("./adapter/adapter_model.safetensors")

# Convert to MLX format
# PEFT names: base_model.model.model.layers.{i}.{module}.lora_A.weight / lora_B.weight
# MLX names: model.layers.{i}.{module}.lora_a / lora_b

mlx_weights = {}
for key, tensor in adapter_weights.items():
    # Convert to numpy
    w = tensor.numpy().astype(np.float16)
    
    # Rename from PEFT format to MLX format
    new_key = key.replace("base_model.model.model.", "model.")
    new_key = new_key.replace("base_model.model.", "model.")
    new_key = new_key.replace("lora_A.weight", "lora_a")
    new_key = new_key.replace("lora_B.weight", "lora_b")
    
    # MLX expects lora_a as (rank, in_features) — same as PEFT
    # MLX expects lora_b as (out_features, rank) — same as PEFT
    mlx_weights[new_key] = w

# Save in MLX format
output_dir = Path("./adapter-mlx")
output_dir.mkdir(exist_ok=True)

# Save weights as npz
np.savez(str(output_dir / "adapters.npz"), **mlx_weights)

# Save adapter config for MLX
mlx_config = {
    "lora_layers": 224,
    "lora_parameters": {
        "rank": 64,
        "alpha": 16,
        "dropout": 0.0,
        "scale": 16.0 / 64.0,
    },
}
with open(output_dir / "adapter_config.json", "w") as f:
    json.dump(mlx_config, f, indent=2)

print(f"Converted {len(mlx_weights)} weight tensors")
print(f"Saved to {output_dir}")