import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from student.student_model import load_student_model
from utils.logger_config import setup_logger
import glob

logger = setup_logger(__name__)

def formatting_prompts_func(example):
    """
    Format the dataset samples into the Jarvis chat template.
    Distillation setup: We train the student to output reasoning + response.
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        # Construct the "gold" output from the teacher
        # We want the student to learn the thought process too!
        full_response = f"[THOUGHT]: {example['reasoning'][i]}\n\n[RESPONSE]: {example['teacher_response'][i]}"
        
        # Jarvis Persona System Prompt
        system_prompt = (
            "You are Jarvis, a highly intelligent and precise AI assistant. "
            "When given a task, you MUST follow this exact two-part format:\n\n"
            "[THOUGHT]: First, briefly reason through the problem step-by-step. "
            "[RESPONSE]: Then, provide the final, direct answer."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example['instruction'][i] + (f"\n\nInput: {example['input'][i]}" if example['input'][i] else "")},
            {"role": "assistant", "content": full_response}
        ]
        
        # We don't have a tokenizer here to apply_chat_template easily in the map func
        # So we'll just format it as a string or use the tokenizer later in the trainer
        # Actually, SFTTrainer can handle messages format if we provide it.
        output_texts.append(messages)
    return output_texts

def train():
    # 1. Load Dataset from chunks
    data_files = glob.glob("teacher_datasets/*.jsonl")
    if not data_files:
        logger.error("No dataset files found in teacher_datasets/. Please run Phase 1 first.")
        return

    dataset = load_dataset("json", data_files=data_files, split="train")
    logger.info(f"Loaded {len(dataset)} samples for training.")

    # 2. Load Model & Tokenizer
    tokenizer, model = load_student_model()

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir="./jarvis_sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # RTX 3060 supports bfloat16
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # 4. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=model.peft_config,
        dataset_text_field=None, # We use formatting_func or messages
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        formatting_func=lambda x: [tokenizer.apply_chat_template(msg, tokenize=False) for msg in formatting_prompts_func(x)]
    )

    # 5. Start Training
    logger.info("Starting training...")
    trainer.train()

    # 6. Save the model
    logger.info("Training complete. Saving model...")
    trainer.model.save_pretrained("./jarvis_lite_final")
    tokenizer.save_pretrained("./jarvis_lite_final")
    logger.info("✅ Model saved to ./jarvis_lite_final")

if __name__ == "__main__":
    train()
