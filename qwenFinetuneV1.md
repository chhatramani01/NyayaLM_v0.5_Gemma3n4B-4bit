This is designed to run on your **Nvidia RTX 5070 Ti (16GB VRAM)**.

### 1. Installation & Setup
Run this in your terminal or notebook to install the necessary libraries. This does not include `unsloth`.

```bash
# Install PyTorch (Assuming CUDA 12.x for your 5070 Ti)
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu124

# Install Hugging Face Libraries
pip install --upgrade \
  "transformers>=4.46.0" \
  "peft>=0.14.0" \
  "bitsandbytes>=0.45.0" \
  "accelerate>=1.2.0" \
  "trl>=0.15.0" \
  "datasets>=3.0.0"

# Optional: For logging
pip install tensorboard
```

### 2. Python Script for Fine-Tuning
You can run this script directly. It handles data loading, formatting, 4-bit quantization, and training using the Google Gemma/QLoRA methodology applied to your Qwen3 model.

```python
import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# -----------------------------------------------------------------------------
# 1. Configuration & Login
# -----------------------------------------------------------------------------
# Login to Hugging Face if the model is gated or you want to push to hub
# from huggingface_hub import login
# login(token="YOUR_HF_TOKEN_HERE")

# Model ID
model_id = "unsloth/Qwen3-4B-Instruct-2507"

# Check GPU capability (5070 Ti supports BF16, good for training)
# bf16 is generally preferred for stability over fp16 if supported
if torch.cuda.get_device_capability(0)[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

print(f"Using dtype: {torch_dtype}")

# -----------------------------------------------------------------------------
# 2. Dataset Loading (Logic from your attached notebook)
# -----------------------------------------------------------------------------
local_dataset_path = "legal_dataset_mixed_29K_dataset.jsonl"

if os.path.exists(local_dataset_path):
    raw_data = []
    with open(local_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
    
    dataset = Dataset.from_list(raw_data)
    print(f"Successfully loaded dataset with {len(dataset)} examples.")
else:
    # Fallback for demonstration if file doesn't exist locally yet
    print(f"File not found at {local_dataset_path}. Please ensure it is present.")
    # Create a dummy dataset for code structure demonstration
    dataset = Dataset.from_list([{"instruction": "Test", "output": "Test Answer"}])

# Train / Eval split (90 / 10)
split = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# -----------------------------------------------------------------------------
# 3. Tokenizer & Chat Template Formatting
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Qwen models often don't have a pad token set by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define the System Prompt (Exact copy from your notebook)
SYSTEM_PROMPT = (
    "You are NyayaLM, a legal assistant specialized in Nepal Laws. "
    "Answer legal questions accurately and professionally. "
    "Cite the exact Section (Dafa) or Artcile numbers where applicable. "
    "Do not invent or assume any legal provisions."
)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []

    for instruction, output in zip(instructions, outputs):
        # Construct the conversation messages list
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]

        # Apply the Qwen chat template
        # This replaces the unsloth.chat_templates function
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}

# Apply formatting to datasets
# We do this before passing to trainer to ensure the prompt is exactly as designed
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_dataset.column_names
)

eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=eval_dataset.column_names
)

print("Formatting complete. Example:")
print(train_dataset[0]["text"][:200] + "...")

# -----------------------------------------------------------------------------
# 4. Model Loading with QLoRA (4-bit Quantization)
# -----------------------------------------------------------------------------

# BitsAndBytesConfig for efficient memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager", # or "flash_attention_2" if installed/compatible
    torch_dtype=torch_dtype,
)

# Prepare model for training (enable gradient checkpointing to save VRAM)
model.config.use_cache = False # Silence warnings during training
model.gradient_checkpointing_enable()

# -----------------------------------------------------------------------------
# 5. LoRA Configuration (PEFT)
# -----------------------------------------------------------------------------
peft_config = LoraConfig(
    r=32,                # Rank from notebook
    lora_alpha=64,       # Alpha from notebook
    lora_dropout=0.05,   # Dropout from notebook
    bias="none",
    target_modules=[      # Target modules from notebook
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

# -----------------------------------------------------------------------------
# 6. Training Arguments (SFTConfig)
# -----------------------------------------------------------------------------
# Hyperparameters adapted for 12GB VRAM (5070 Ti)
# Reduce per_device_train_batch_size if you run out of memory (OOM)
training_args = SFTConfig(
    output_dir="./qwen3_nepal_legal_lora",
    
    # Data & Batch settings
    dataset_text_field="text", # The column created by formatting_prompts_func
    max_seq_length=2048,      # From notebook
    packing=False,            # False because we pre-formatted full strings
    per_device_train_batch_size=2, # Safe for 4-bit on 12GB
    gradient_accumulation_steps=8, # Effective batch size = 2 * 8 = 16
    num_train_epochs=1,       # From notebook
    
    # Optimizer & Scheduler
    learning_rate=1e-4,       # From notebook
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine", # From notebook
    warmup_ratio=0.05,        # From notebook
    
    # Regularization
    weight_decay=0.01,
    max_grad_norm=0.7,        # From notebook
    
    # Evaluation & Saving
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    
    # Logging & System
    logging_steps=50,
    logging_dir="./logs",
    report_to="tensorboard",
    fp16=(torch_dtype == torch.float16),
    bf16=(torch_dtype == torch.bfloat16),
    gradient_checkpointing=True, # Critical for 12GB VRAM
    seed=3407,
)

# -----------------------------------------------------------------------------
# 7. Trainer Initialization
# -----------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# -----------------------------------------------------------------------------
# 8. Train
# -----------------------------------------------------------------------------
print("Starting training...")
trainer.train()

# Save the model (LoRA adapters only)
trainer.save_model("./qwen3_nepal_legal_lora_final")
tokenizer.save_pretrained("./qwen3_nepal_legal_lora_final")

print("Training complete. Model saved to ./qwen3_nepal_legal_lora_final")

# -----------------------------------------------------------------------------
# 9. Inference Test (Using the trained adapter)
# -----------------------------------------------------------------------------
print("\n--- Testing Inference ---")

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch_dtype,
)

# Merge the adapter with the base model
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./qwen3_nepal_legal_lora_final")
model = model.merge_and_unload() # Optional: Merge for easier inference, or skip to keep adapters separate

# Test Question
test_instruction = "How can a person acquire citizenship of Nepal?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": test_instruction}
]

# Apply chat template
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

print("Question:", test_instruction)
print("Answer:", tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```

### Key Changes Explained:
1.  **Removed Unsloth**: Replaced `FastLanguageModel` with standard Hugging Face `AutoModelForCausalLM` and `BitsAndBytesConfig`.
2.  **Tokenizer**: Uses `AutoTokenizer` from Hugging Face. The `apply_chat_template` method is built-in to Hugging Face tokenizers for Qwen, so `unsloth.chat_templates` is no longer needed.
3.  **Formatting**: I created a standard Python function `formatting_prompts_func` that replicates your logic. It constructs the `messages` dictionary list and applies the template to a `text` field, which `SFTTrainer` then consumes.
4.  **Memory Optimization**:
    *   Added `bnb_4bit_use_double_quant=True` to save VRAM.
    *   Added `gradient_checkpointing=True` (essential for 12GB cards).
    *   Set `per_device_train_batch_size=2`. If you hit an Out Of Memory (OOM) error, lower this to 1 and increase `gradient_accumulation_steps` to 16.
5.  **Data Loading**: Kept the exact JSONL parsing logic from your notebook.
