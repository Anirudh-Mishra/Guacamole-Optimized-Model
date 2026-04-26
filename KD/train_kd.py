"""
Phase 2: Train Guacamole-KD
Fine-tune LLaMA-3-8B using QLoRA with knowledge distillation.
The loss combines standard cross-entropy with KL divergence
from pre-computed teacher logits (LLaMA-3-8B-Instruct).

Expected runtime: ~2-3 hours on ml.g5.12xlarge (4x A10G)
Output: LoRA adapter in ./outputs/guacamole-kd/final-adapter
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import Dataset

# ═══════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DATASET_ID = "timdettmers/openassistant-guanaco"
TEACHER_LOGITS_DIR = "./teacher_logits"
OUTPUT_DIR = "./outputs/guacamole-kd"

# KD hyperparameters
ALPHA = 0.5          # Balance: 0 = pure CE, 1 = pure KD
TEMPERATURE = 2.0    # Softens teacher distribution
TOP_K = 50           # Must match generate_teacher_logits.py

# Training hyperparameters (same as baseline)
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-4
MAX_STEPS = 1875
BATCH_SIZE = 1
GRAD_ACCUM = 16
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ═══════════════════════════════════════════
# 1. Load Teacher Logits from Disk
# ═══════════════════════════════════════════
print("Loading pre-computed teacher logits...")

teacher_data = {}
batch_files = sorted(glob.glob(os.path.join(TEACHER_LOGITS_DIR, "batch_*.pt")))
for bf in batch_files:
    batch = torch.load(bf, map_location="cpu")
    for item in batch:
        teacher_data[item["idx"]] = {
            "top_k_values": item["top_k_values"],
            "top_k_indices": item["top_k_indices"],
        }

print(f"Loaded teacher logits for {len(teacher_data)} examples")

metadata = torch.load(os.path.join(TEACHER_LOGITS_DIR, "metadata.pt"))
print(f"Teacher model: {metadata['teacher_model']}")

# ═══════════════════════════════════════════
# 2. Custom Dataset with Teacher Logits
# ═══════════════════════════════════════════
class KDDataset(Dataset):
    """Dataset that returns tokenized text + corresponding teacher logits."""
    
    def __init__(self, hf_dataset, tokenizer, teacher_data, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.teacher_data = teacher_data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        # Get teacher logits for this example
        if idx in self.teacher_data:
            teacher_top_k_values = self.teacher_data[idx]["top_k_values"]
            teacher_top_k_indices = self.teacher_data[idx]["top_k_indices"]
            
            # Pad or truncate teacher logits to match max_length
            teacher_seq_len = teacher_top_k_values.shape[0]
            if teacher_seq_len < self.max_length:
                # Pad with zeros
                pad_len = self.max_length - teacher_seq_len
                teacher_top_k_values = F.pad(teacher_top_k_values, (0, 0, 0, pad_len), value=0)
                teacher_top_k_indices = F.pad(teacher_top_k_indices, (0, 0, 0, pad_len), value=0)
            else:
                teacher_top_k_values = teacher_top_k_values[:self.max_length]
                teacher_top_k_indices = teacher_top_k_indices[:self.max_length]
        else:
            # Fallback: no teacher data, use zeros (CE only for this example)
            teacher_top_k_values = torch.zeros(self.max_length, TOP_K, dtype=torch.float16)
            teacher_top_k_indices = torch.zeros(self.max_length, TOP_K, dtype=torch.int32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "teacher_top_k_values": teacher_top_k_values,
            "teacher_top_k_indices": teacher_top_k_indices,
        }

# ═══════════════════════════════════════════
# 3. Custom Trainer with KD Loss
# ═══════════════════════════════════════════
class KDTrainer(Trainer):
    """Custom trainer that computes combined CE + KL divergence loss."""
    
    def __init__(self, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract teacher data from inputs
        teacher_top_k_values = inputs.pop("teacher_top_k_values")
        teacher_top_k_indices = inputs.pop("teacher_top_k_indices")
        labels = inputs.get("labels")
        
        # Forward pass through student
        outputs = model(**inputs)
        student_logits = outputs.logits  # (batch, seq_len, vocab_size)
        
        # ─── Loss 1: Standard Cross-Entropy ───
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # ─── Loss 2: KL Divergence with Teacher ───
        # Reconstruct sparse teacher logits into full distribution
        # Only compute KL on positions where we have teacher data and valid labels
        T = self.temperature
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Create mask for valid positions (not padding, not the last position)
        valid_mask = (labels[..., 1:] != -100)  # (batch, seq_len-1)
        
        if valid_mask.any():
            # Shift student logits to align with teacher (predict next token)
            student_shifted = student_logits[..., :-1, :]  # (batch, seq_len-1, vocab)
            teacher_vals_shifted = teacher_top_k_values[..., :-1, :]  # (batch, seq_len-1, top_k)
            teacher_idx_shifted = teacher_top_k_indices[..., :-1, :]  # (batch, seq_len-1, top_k)
            
            # Reconstruct full teacher distribution from top-K
            # Set all positions to a very negative value, then fill in top-K
            teacher_full = torch.full_like(student_shifted, fill_value=-1e4)
            teacher_full.scatter_(
                dim=-1,
                index=teacher_idx_shifted.long().to(student_shifted.device),
                src=teacher_vals_shifted.float().to(student_shifted.device),
            )
            
            # Compute softened distributions
            student_log_prob = F.log_softmax(student_shifted / T, dim=-1)
            teacher_prob = F.softmax(teacher_full / T, dim=-1)
            
            # KL divergence (only on valid positions)
            kl_per_token = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction="none",
            ).sum(dim=-1)  # (batch, seq_len-1)
            
            # Mask and average
            kl_loss = (kl_per_token * valid_mask.float()).sum() / valid_mask.float().sum()
            kl_loss = kl_loss * (T * T)  # Scale by T^2 to balance gradients
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)
        
        # ─── Combined Loss ───
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        # Log individual losses
        if self.state.global_step % 25 == 0:
            print(f"  Step {self.state.global_step}: CE={ce_loss.item():.4f}, KL={kl_loss.item():.4f}, Combined={loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss

# ═══════════════════════════════════════════
# 4. Load Student Model
# ═══════════════════════════════════════════
print("\nLoading student model (LLaMA-3-8B) in 4-bit NF4...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ═══════════════════════════════════════════
# 5. Prepare for QLoRA Training
# ═══════════════════════════════════════════
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ═══════════════════════════════════════════
# 6. Load Dataset and Create KD Dataset
# ═══════════════════════════════════════════
print("\nLoading OASST1 dataset...")
raw_dataset = load_dataset(DATASET_ID, split="train")

kd_dataset = KDDataset(
    hf_dataset=raw_dataset,
    tokenizer=tokenizer,
    teacher_data=teacher_data,
    max_length=MAX_SEQ_LENGTH,
)

print(f"KD Dataset size: {len(kd_dataset)} examples")
print(f"Examples with teacher logits: {len(teacher_data)}")

# ═══════════════════════════════════════════
# 7. Training Arguments
# ═══════════════════════════════════════════
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="constant",
    max_steps=MAX_STEPS,
    optim="paged_adamw_32bit",
    bf16=True,
    gradient_checkpointing=True,
    group_by_length=False,  # Disabled because we need to align with teacher logits by index
    logging_steps=25,
    save_steps=250,
    save_total_limit=3,
    report_to="none",
    remove_unused_columns=False,  # Keep teacher logit columns
    dataloader_pin_memory=False,
)

# ═══════════════════════════════════════════
# 8. Create KD Trainer and Train
# ═══════════════════════════════════════════
trainer = KDTrainer(
    alpha=ALPHA,
    temperature=TEMPERATURE,
    model=model,
    args=training_args,
    train_dataset=kd_dataset,
    tokenizer=tokenizer,
)

print(f"\n{'='*60}")
print(f"Starting Guacamole-KD Training")
print(f"{'='*60}")
print(f"Alpha (KD weight): {ALPHA}")
print(f"Temperature: {TEMPERATURE}")
print(f"Total steps: {MAX_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Teacher model: {metadata['teacher_model']}")
print(f"{'='*60}\n")

trainer.train()

# ═══════════════════════════════════════════
# 9. Save Adapter
# ═══════════════════════════════════════════
save_path = f"{OUTPUT_DIR}/final-adapter"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\n{'='*60}")
print(f"Guacamole-KD training complete!")
print(f"Adapter saved to: {save_path}")
print(f"{'='*60}")
