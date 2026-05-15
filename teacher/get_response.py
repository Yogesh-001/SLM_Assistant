from utils.logger_config import setup_logger
from datasets import load_dataset

# -------------------------------
# Logging Config
# -------------------------------
logger = setup_logger(__name__)


# -------------------------------
# Teacher Response (Jarvis Persona + Chain-of-Thought)
# -------------------------------

def get_teacher_response(
    model,
    tokenizer,
    instruction: str,
    device: str,
    input_text: str = "",
    max_new_tokens: int = 512,
    use_cot: bool = True,
) -> dict:
    """
    Generate a structured response from the teacher model using the Jarvis persona.

    The function uses Chain-of-Thought (CoT) prompting when `use_cot=True`.
    The teacher is asked to first reason through the problem, then provide a
    clean, direct final answer. Both are stored separately in the output.

    Args:
        model: HuggingFace model instance (teacher).
        tokenizer: Corresponding tokenizer.
        instruction (str): The instruction/question prompt.
        device (str): Device to run inference on ("cuda" or "cpu").
        input_text (str, optional): Additional context. Defaults to "".
        max_new_tokens (int, optional): Max tokens to generate. Defaults to 512.
        use_cot (bool, optional): Whether to use Chain-of-Thought. Defaults to True.

    Returns:
        dict: A dictionary with keys:
            - "reasoning" (str): The model's step-by-step thought process.
            - "response" (str): The clean, final answer.
    """
    # -----------------------------------------------------------------------
    # System Prompt: Jarvis Persona
    # The teacher is instructed to THINK first, then give a clean final answer.
    # This is the key technique that makes distillation produce a "reasoning-aware"
    # student model, rather than just a mimicry of surface-level answers.
    # -----------------------------------------------------------------------
    if use_cot:
        system_prompt = (
            "You are Jarvis, a highly intelligent and precise AI assistant. "
            "When given a task, you MUST follow this exact two-part format:\n\n"
            "[THOUGHT]: First, briefly reason through the problem step-by-step. "
            "Be concise. No more than 3-4 sentences.\n\n"
            "[RESPONSE]: Then, provide the final, direct answer. "
            "No explanations, no notes. Just the answer.\n\n"
            "Always use [THOUGHT]: and [RESPONSE]: as exact headers."
        )
    else:
        system_prompt = (
            "You are Jarvis, a highly intelligent and precise AI assistant. "
            "Answer the user's request directly and concisely. "
            "Only return the final answer — no explanations, notes, or caveats."
        )

    # Construct user message
    user_prompt = f"{instruction}\n\nInput: {input_text}" if input_text else instruction

    # Apply chat template (handles Llama-3.1 special tokens correctly)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    logger.info("Generating teacher response...")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.4,       # Slightly higher than before for richer CoT
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # Suppress padding warnings
    )

    # -----------------------------------------------------------------------
    # Decode ONLY the newly generated tokens (not the prompt).
    # This is more robust than string-splitting and avoids edge cases where
    # the user_prompt text appears inside the generated response.
    # -----------------------------------------------------------------------
    generated_ids = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # -----------------------------------------------------------------------
    # Parse CoT output into structured fields
    # -----------------------------------------------------------------------
    reasoning = ""
    final_response = response_text  # fallback: return full text if tags are missing

    if use_cot and "[THOUGHT]:" in response_text and "[RESPONSE]:" in response_text:
        try:
            thought_part = response_text.split("[THOUGHT]:")[1].split("[RESPONSE]:")[0].strip()
            response_part = response_text.split("[RESPONSE]:")[-1].strip()
            reasoning = thought_part
            final_response = response_part
        except IndexError:
            logger.warning("CoT tags found but parsing failed — returning full response as-is.")

    logger.debug(f"[THOUGHT]: {reasoning}")
    logger.debug(f"[RESPONSE]: {final_response}")

    return {
        "reasoning": reasoning,
        "response": final_response,
    }


# -------------------------------
# Dataset Loader
# -------------------------------

def get_dataset(sample_size: int = 8000, seed: int = 42):
    """
    Load and sample Alpaca dataset for knowledge distillation.

    Args:
        sample_size (int): Number of samples to use (default=8000).
        seed (int): Random seed for reproducibility.

    Returns:
        Dataset: A HuggingFace dataset object with sampled data.
    """
    logger.info("Loading Alpaca dataset...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")

    logger.info(f"Full Alpaca dataset size: {len(alpaca)}")

    if sample_size < len(alpaca):
        alpaca = alpaca.shuffle(seed=seed).select(range(sample_size))
        logger.info(f"Sampled subset size: {len(alpaca)} (seed={seed})")
    else:
        logger.warning("Requested sample_size >= dataset size. Using full dataset.")

    return alpaca
