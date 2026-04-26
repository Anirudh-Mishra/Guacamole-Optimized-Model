"""
Phase 1: Generate Teacher Logits
Run LLaMA-3-8B-Instruct on every OASST1 training example and save
the top-K logits to disk. These will be used as the teacher signal
during Guacamole-KD training.

Expected runtime: ~1-2 hours on ml.g5.12xlarge
Output: teacher_logits/ directory with one .pt file per batch
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import math

# ─── Configuration ───
TEACHER_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "timdettmers/openassistant-guanaco"
OUTPUT_DIR = "./teacher_logits"
MAX_SEQ_LENGTH = 512
TOP_K = 50  # Save only top-K logits per position to save disk space
BATCH_SIZE = 1  # Process one example at a time to avoid OOM
SAVE_EVERY = 100  # Save checkpoint every N examples

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load Teacher Model in 4-bit (to fit in memory) ───
print("Loading teacher model (LLaMA-3-8B-Instruct) in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
teacher_model.eval()

tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─── Load Dataset ───
print("Loading OASST1 dataset...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Dataset size: {len(dataset)} examples")

# ─── Check for existing progress ───
existing_files = set()
if os.path.exists(os.path.join(OUTPUT_DIR, "progress.pt")):
    progress = torch.load(os.path.join(OUTPUT_DIR, "progress.pt"))
    start_idx = progress["last_completed"] + 1
    print(f"Resuming from example {start_idx}")
else:
    start_idx = 0

# ─── Generate Teacher Logits ───
print(f"\nGenerating teacher logits for {len(dataset) - start_idx} examples...")
print(f"Saving top-{TOP_K} logits per position")
print(f"Output directory: {OUTPUT_DIR}\n")

all_teacher_data = []

for idx in tqdm(range(start_idx, len(dataset)), desc="Generating logits"):
    example = dataset[idx]
    
    # Tokenize
    inputs = tokenizer(
        example["text"],
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
    ).to(teacher_model.device)
    
    # Forward pass (no gradients)
    with torch.no_grad():
        outputs = teacher_model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    
    # Extract top-K logits and indices to save disk space
    # Full logits would be seq_len * 128256 * 4 bytes ≈ 260 MB per example
    # Top-50 is seq_len * 50 * 4 bytes ≈ 100 KB per example
    top_k_values, top_k_indices = torch.topk(logits[0].float().cpu(), k=TOP_K, dim=-1)
    
    all_teacher_data.append({
        "idx": idx,
        "top_k_values": top_k_values.half(),  # (seq_len, TOP_K) in float16
        "top_k_indices": top_k_indices.int(),  # (seq_len, TOP_K) in int32
        "seq_len": inputs["input_ids"].shape[1],
    })
    
    # Save periodically to avoid losing progress
    if (idx + 1) % SAVE_EVERY == 0 or idx == len(dataset) - 1:
        batch_file = os.path.join(OUTPUT_DIR, f"batch_{idx // SAVE_EVERY:04d}.pt")
        torch.save(all_teacher_data, batch_file)
        
        # Save progress
        torch.save({"last_completed": idx}, os.path.join(OUTPUT_DIR, "progress.pt"))
        
        print(f"  Saved {len(all_teacher_data)} examples to {batch_file}")
        all_teacher_data = []
    
    # Free GPU memory
    del outputs, logits, top_k_values, top_k_indices
    if idx % 500 == 0:
        torch.cuda.empty_cache()

# ─── Save metadata ───
metadata = {
    "teacher_model": TEACHER_MODEL_ID,
    "dataset": DATASET_ID,
    "total_examples": len(dataset),
    "max_seq_length": MAX_SEQ_LENGTH,
    "top_k": TOP_K,
    "save_every": SAVE_EVERY,
}
torch.save(metadata, os.path.join(OUTPUT_DIR, "metadata.pt"))

print(f"\n{'='*60}")
print(f"Teacher logit generation complete!")
print(f"Total examples: {len(dataset)}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*60}")
