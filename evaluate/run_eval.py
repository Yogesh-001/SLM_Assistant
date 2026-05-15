import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from teacher.teacher_model import load_teacher_model
from utils.logger_config import setup_logger
from tqdm import tqdm
from rouge_score import rouge_scorer

logger = setup_logger(__name__)

def load_fine_tuned_student(base_model_id="Qwen/Qwen2-1.5B-Instruct", adapter_path="./jarvis_lite_final"):
    """
    Load the fine-tuned student model (Base + LoRA Adapters).
    """
    logger.info(f"Loading base student model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model in 4-bit to save memory (same as training)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model

def generate_response(model, tokenizer, instruction, input_text="", device="cuda"):
    system_prompt = (
        "You are Jarvis, a highly intelligent and precise AI assistant. "
        "When given a task, you MUST follow this exact two-part format:\n\n"
        "[THOUGHT]: First, briefly reason through the problem step-by-step. "
        "[RESPONSE]: Then, provide the final, direct answer."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction + (f"\n\nInput: {input_text}" if input_text else "")}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # Keep it deterministic for eval
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_evaluation(num_samples=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Student
    try:
        student_tokenizer, student_model = load_fine_tuned_student()
    except Exception as e:
        logger.error(f"Could not load student model: {e}. Make sure Phase 2 is finished.")
        return

    # 2. Load Teacher
    teacher_tokenizer, teacher_model = load_teacher_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # 3. Test Queries (Mix of general and technical)
    test_queries = [
        {"instruction": "Explain the concept of quantum entanglement.", "input": ""},
        {"instruction": "Write a Python function to check if a number is prime.", "input": ""},
        {"instruction": "How do I fix a leaky faucet?", "input": ""},
        {"instruction": "What are the key differences between SQL and NoSQL?", "input": ""},
        {"instruction": "Plan a 3-day itinerary for Tokyo.", "input": ""},
    ]
    
    # Add some samples from the training set (optional) or a separate eval set
    # For now, let's just use these few for a quick demo.

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results = []

    logger.info(f"Starting evaluation on {len(test_queries)} samples...")

    for query in tqdm(test_queries):
        instr = query["instruction"]
        inp = query["input"]
        
        student_res = generate_response(student_model, student_tokenizer, instr, inp, device)
        teacher_res = generate_response(teacher_model, teacher_tokenizer, instr, inp, device)
        
        # Calculate ROUGE between Student Response and Teacher Response
        # (We treat Teacher as the 'reference'/ground truth)
        scores = scorer.score(teacher_res, student_res)
        
        results.append({
            "instruction": instr,
            "teacher": teacher_res,
            "student": student_res,
            "rouge1": scores['rouge1'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        })

    # 4. Save & Print Results
    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    avg_r1 = sum(r["rouge1"] for r in results) / len(results)
    avg_rL = sum(r["rougeL"] for r in results) / len(results)
    
    logger.info(f"Evaluation Complete!")
    logger.info(f"Average ROUGE-1: {avg_r1:.4f}")
    logger.info(f"Average ROUGE-L: {avg_rL:.4f}")
    logger.info(f"Detailed results saved to eval_results/comparison.json")

if __name__ == "__main__":
    run_evaluation()
