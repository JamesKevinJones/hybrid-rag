"""Module 4b - QLoRA training. Runs on Colab, NOT on this machine.

Why this file is not a local script
-----------------------------------
4-bit QLoRA on a 3B model needs roughly 6-7 GB of VRAM once you count the base
weights, LoRA gradients, optimizer state and activations. This host has a 6 GB
card with about 5.2 GB usable after the Windows desktop compositor takes its
share. That is not a tuning problem - it is short by more than a gigabyte, and
the failure mode is either a hard CUDA OOM or, worse, silent paging into shared
system memory over a single-channel bus, where a run that should take 20
minutes takes nine hours.

A free Colab T4 has 15 GB. The same job fits four times over.

Upload this file to Colab, or open it as a notebook. It is standalone: it does
not import anything from `hybrid_rag`. The only local dependency is the dataset
produced by `hybrid_rag.distill.dataset_formatter`, which you upload alongside.

Runtime -> Change runtime type -> T4 GPU. Then run top to bottom.
"""

# ---------------------------------------------------------------------------
# 1. Install. Colab ships torch already; pin the rest.
# ---------------------------------------------------------------------------
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --no-deps "trl<0.12" peft accelerate bitsandbytes

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

# Must match what Ollama serves locally, or the adapter will not compose with
# the base model you actually run. llama3.2:3b-instruct-q4_K_M locally maps to
# this repo upstream.
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

DATASET_PATH = Path("distillation_train.jsonl")  # upload this from data/datasets/
OUTPUT_DIR = Path("adapter_out")
GGUF_QUANT = "q4_k_m"  # matches the local Ollama quantization

# 2048 fits comfortably on a T4 with 4-bit weights. Raising it is the fastest
# way to OOM a T4 - attention memory grows with the square of this number.
MAX_SEQ_LENGTH = 2048

# Rank 16 on a few hundred samples. Higher rank on a small dataset memorises
# rather than generalises, and a distillation set collected under Claude Pro
# message caps is small by construction.
LORA_RANK = 16
LORA_ALPHA = 16

EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch 8

# ---------------------------------------------------------------------------
# 3. Load the base model in 4-bit
# ---------------------------------------------------------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # None lets Unsloth pick; a T4 is fp16, it has no bf16 units
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,  # 0.0 is Unsloth's fused fast path
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # ~30% less VRAM, slightly slower
    random_state=1337,  # same seed the local pipeline pins
)

# ---------------------------------------------------------------------------
# 4. Load the ShareGPT dataset written by dataset_formatter.py
# ---------------------------------------------------------------------------

_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def load_sharegpt(path: Path) -> Dataset:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = [
            {"role": _ROLE_MAP[turn["from"]], "content": turn["value"]}
            for turn in record["conversations"]
        ]
        rows.append(
            {
                "text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            }
        )
    if not rows:
        raise SystemExit(f"No samples in {path}. Did the upload finish?")
    print(f"loaded {len(rows)} samples")
    return Dataset.from_list(rows)


dataset = load_sharegpt(DATASET_PATH)

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,  # packing corrupts short instruction samples
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),  # True on a T4
        bf16=is_bfloat16_supported(),  # True on an A100/L4 if you get one
        logging_steps=5,
        optim="adamw_8bit",  # 8-bit optimizer state; meaningful on 15 GB
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=1337,
        output_dir=str(OUTPUT_DIR),
        report_to="none",
    ),
)

stats = trainer.train()
print(f"train_runtime: {stats.metrics['train_runtime']:.0f}s")
print(f"peak VRAM: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB of 15 GB")

# ---------------------------------------------------------------------------
# 6. Export to GGUF for local Ollama
# ---------------------------------------------------------------------------
# This is the step that closes the loop: the adapter is merged into the base
# weights and quantized to the same q4_k_m the local runtime serves, so what
# comes back is a drop-in replacement rather than something needing a separate
# adapter-loading path at inference time.

GGUF_DIR = "gguf_out"
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method=GGUF_QUANT)

MODELFILE = f"""FROM ./{GGUF_QUANT}.gguf
PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER seed 1337
"""
Path(GGUF_DIR, "Modelfile").write_text(MODELFILE, encoding="utf-8")

for name in sorted(os.listdir(GGUF_DIR)):
    size = Path(GGUF_DIR, name).stat().st_size / 1e9
    print(f"  {name}  {size:.2f} GB")

print(
    "\n"
    "Download the .gguf and the Modelfile, put them in one folder locally, then:\n"
    "\n"
    "    ollama create hybrid-rag-distilled -f Modelfile\n"
    "    $env:HYBRID_RAG_OLLAMA__GENERATOR_MODEL = 'hybrid-rag-distilled'\n"
    "    uv run hybrid-rag doctor\n"
    "\n"
    "Then benchmark it against the Opus baseline BEFORE trusting it - a\n"
    "distilled model that is quietly worse costs more in re-escalations than\n"
    "it saves. That harness is Module 5.\n"
)
