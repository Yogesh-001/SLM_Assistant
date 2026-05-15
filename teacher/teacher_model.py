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

    Uses BitsAndBytes NF4 quantization which is the current best practice for
    inference quality vs. memory trade-off on consumer GPUs like the RTX 3060.

    Args:
        model_name (str): Hugging Face model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        tokenizer (AutoTokenizer): Tokenizer for the model
        model (AutoModelForCausalLM): Quantized model ready for inference
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Hugging Face Token not found. Please set HF_TOKEN in .env file.")
        raise EnvironmentError("HF_TOKEN not found in environment variables.")

    logger.info(f"Loading teacher model: {model_name} with 4-bit NF4 quantization...")

    try:
        # -----------------------------------------------------------------------
        # NF4 Quantization Config (best practice for QLoRA-style inference)
        # - load_in_4bit: reduce VRAM footprint (~5GB for 8B model on RTX 3060)
        # - bnb_4bit_quant_type="nf4": NormalFloat4 is superior to fp4 for LLMs
        # - bnb_4bit_use_double_quant=True: further reduces memory ~0.4 bits/param
        # - bnb_4bit_compute_dtype=bfloat16: fast compute on Ampere GPUs (RTX 30xx)
        # -----------------------------------------------------------------------
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )

        # Llama-3 models don't have a pad token by default — set it to eos_token
        # This is essential to avoid errors during batch generation.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token = eos_token (required for Llama-3 models).")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",          # Auto-distributes across GPU/CPU
            token=hf_token,
            attn_implementation="eager",  # Use "flash_attention_2" if installed
        )
        model.eval()  # Set to eval mode — disables dropout for deterministic inference

        logger.info(f"✅ Successfully loaded teacher model: {model_name}")
        return tokenizer, model

    except Exception as e:
        logger.exception(f"Failed to load teacher model '{model_name}': {e}")
        raise
