import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def load_student_model(model_id: str = "Qwen/Qwen2-1.5B-Instruct"):
    """
    Load the student model with QLoRA configuration.
    
    Args:
        model_id (str): Hugging Face model ID for the student.
        
    Returns:
        tokenizer: The model tokenizer.
        model: The PEFT-wrapped model ready for training.
    """
    logger.info(f"Loading student model: {model_id}")

    # 4-bit Quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Recommended for SFT

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA Configuration
    # target_modules for Qwen2 often include q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model
