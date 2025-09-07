import os
import torch
from dotenv import load_dotenv
from utils.logger_config import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
logger = setup_logger(__name__)

# -------------------------------
# Load Teacher Model
# -------------------------------
def load_teacher_model(model_name: str):
    """
    Load a teacher model and tokenizer from Hugging Face Hub with 4-bit quantization.
    
    Args:
        model_name (str): Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
    
    Returns:
        tokenizer (AutoTokenizer): Tokenizer for the model
        model (AutoModelForCausalLM): Quantized model ready for inference
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Hugging Face Token not found. Please set HF_TOKEN in the environment or .env file.")
        raise EnvironmentError("HF_TOKEN not found in environment variables.")
    
    logger.info(f"Loading teacher model: {model_name} with 4-bit quantization...")
    
    try:
        # Quantization config
        # bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            use_fast=True
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
        )

        logger.info(f"Successfully loaded teacher model: {model_name}")
        return tokenizer, model
    
    except Exception as e:
        logger.exception(f"Failed to load teacher model {model_name}: {e}")
        raise
