import os
import json
import torch
from tqdm import tqdm

from teacher.teacher_model import load_teacher_model
from teacher.get_response import get_dataset, get_teacher_response
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


def get_completed_ids(output_dir: str) -> set:
    """
    Scan existing JSONL chunk files to find already-processed sample IDs.

    This enables crash recovery: if the script stops at sample 4500, restarting
    it will skip the first 4500 samples instead of regenerating them from scratch.

    Args:
        output_dir (str): Directory where chunk files are stored.

    Returns:
        set: A set of integer indices that have already been saved.
    """
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    for fname in os.listdir(output_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "sample_id" in record:
                            completed.add(record["sample_id"])
                    except json.JSONDecodeError:
                        continue
    logger.info(f"Resume: Found {len(completed)} already-completed samples.")
    return completed


if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Device Setup
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning("CUDA not available. Running on CPU will be very slow.")

    # -----------------------------------------------------------------------
    # Load Teacher Model
    # -----------------------------------------------------------------------
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer, model = load_teacher_model(model_id)

    # -----------------------------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------------------------
    alpaca = get_dataset(sample_size=8000, seed=42)
    logger.info(f"Loaded dataset with {len(alpaca)} samples.")

    # -----------------------------------------------------------------------
    # Output Setup
    # -----------------------------------------------------------------------
    output_dir = "teacher_datasets"
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = 2000

    # -----------------------------------------------------------------------
    # Resume Mechanism
    # Scans all existing chunks to find which sample IDs are already done.
    # -----------------------------------------------------------------------
    completed_ids = get_completed_ids(output_dir)
    skipped = len(completed_ids)
    if skipped > 0:
        logger.info(f"Resuming from sample {skipped}. Skipping {skipped} already-processed samples.")

    # Determine which chunk file to start writing to
    current_chunk = len(completed_ids) // chunk_size
    chunk_path = os.path.join(output_dir, f"alpaca_kd_chunk{current_chunk}.jsonl")
    kd_file = open(chunk_path, "a", encoding="utf-8")  # "a" = append, not overwrite

    # -----------------------------------------------------------------------
    # Main Generation Loop with tqdm Progress Bar
    # -----------------------------------------------------------------------
    errors = 0
    with tqdm(total=len(alpaca), initial=skipped, desc="Generating KD Data", unit="sample") as pbar:
        for i, sample in enumerate(alpaca):

            # Skip already processed samples (resume logic)
            if i in completed_ids:
                continue

            instruction = sample["instruction"]
            input_text  = sample["input"]
            ground_truth = sample["output"]

            try:
                result = get_teacher_response(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    device=device,
                    input_text=input_text,
                    use_cot=True,   # Chain-of-Thought: generates [THOUGHT] + [RESPONSE]
                )
            except Exception as e:
                logger.error(f"Error at sample {i}: {e}")
                errors += 1
                pbar.update(1)
                continue

            # Save record — includes sample_id for the resume mechanism
            record = {
                "sample_id":    i,
                "instruction":  instruction,
                "input":        input_text,
                "reasoning":    result["reasoning"],      # CoT thought process
                "teacher_response": result["response"],   # Clean final answer
                "ground_truth": ground_truth,
            }
            kd_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            kd_file.flush()  # Flush after each sample to minimize data loss on crash

            pbar.update(1)

            # Rotate to a new chunk file every `chunk_size` samples
            if (i + 1) % chunk_size == 0 and i not in completed_ids:
                kd_file.close()
                current_chunk += 1
                chunk_path = os.path.join(output_dir, f"alpaca_kd_chunk{current_chunk}.jsonl")
                kd_file = open(chunk_path, "a", encoding="utf-8")
                logger.info(f"📦 Rotated to chunk {current_chunk}.")

    kd_file.close()
    logger.info(f"✅ Finished! Total errors: {errors} / {len(alpaca)}")
