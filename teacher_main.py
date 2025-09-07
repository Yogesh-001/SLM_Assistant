from teacher.teacher_model import load_teacher_model
from teacher.get_response import get_dataset, get_teacher_response
from utils.logger_config import setup_logger
import os, json
import torch

logger = setup_logger(__name__)

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer, model = load_teacher_model(model_id)

    # Get alpaca subset (e.g., 8k samples instead of full 52k)
    alpaca = get_dataset()
    logger.info(f"Loaded dataset with {len(alpaca)} samples.")

    os.makedirs("teacher_datasets", exist_ok=True)

    chunk_size = 2000
    current_chunk = 0

    kd_file = open(f"teacher_datasets/alpaca_kd_chunk{current_chunk}.jsonl", "w", encoding="utf-8")

    for i, sample in enumerate(alpaca):
        instruction, input_text, output = sample["instruction"], sample["input"], sample["output"]
        
        try:
            teacher_response = get_teacher_response(model, tokenizer, instruction, device, input_text)
        except Exception as e:
            logger.error(f"Error generating response for sample {i}: {e}")
            continue

        kd_file.write(json.dumps({
            "instruction": instruction,
            "input": input_text,
            "teacher_response": teacher_response,
            "ground_truth": output
        } ,ensure_ascii=False) + "\n")

        if (i + 1) % chunk_size == 0:
            kd_file.close()
            current_chunk += 1
            kd_file = open(f"teacher_datasets/alpaca_kd_chunk{current_chunk}.jsonl", "w", encoding="utf-8")
            logger.info(f"Saved chunk {current_chunk} with {chunk_size} samples.")

    kd_file.close()
    logger.info("✅ Finished generating datasets!")


