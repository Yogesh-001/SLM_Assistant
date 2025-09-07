from utils.logger_config import setup_logger
from datasets import load_dataset

# -------------------------------
# Logging Config
# -------------------------------
logger = setup_logger(__name__)

# # -------------------------------
# # Teacher Response Functions
# # -------------------------------

##  This below get_teacher_response version gives the too much of the data in the response like explanations, note ...etc. this may cause hallucination for the model and efficiency may decrease after finetuning the student model.

# def get_teacher_response(model, tokenizer, instruction, device, input="", max_new_tokens=512):
#     if input:
#         prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
#     else:
#         prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

#     inputs = tokenizer(prompt,return_tensors="pt").to(device)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens = max_new_tokens,
#         temperature = 0.7,
#         top_p = 0.9,
#         do_sample = True,
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.split("### Response:")[-1].strip()

## --------------------------------------------------------------------------------------------------------------

## This version specifies the instruct llm do not overthink and to give cleaner, short, structured responses instead of noisy explanations.

def get_teacher_response(
    model,
    tokenizer,
    instruction: str,
    device: str,
    input_text: str = "",
    max_new_tokens: int = 512
) -> str:
    """
    Generate a response from the teacher model using chat template.

    Args:
        model: HuggingFace model instance (teacher).
        tokenizer: Corresponding tokenizer.
        instruction (str): The instruction prompt.
        device (str): Device to run inference on ("cuda" or "cpu").
        input_text (str, optional): Additional input/context. Defaults to "".
        max_new_tokens (int, optional): Maximum tokens to generate. Defaults to 512.

    Returns:
        str: Teacher model's cleaned response.
    """
    logger.info("Loading {model} as teacher model to get the response.")

    # System prompt forces concise, direct answers
    system_prompt = (
        "You are a helpful AI assistant. Answer the user request directly and concisely. "
        "Do NOT provide explanations, notes, or references. Only return the final answer."
    )

    # Construct user message
    user_prompt = f"{instruction}\n\nInput: {input_text}" if input_text else instruction

    # Apply chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    logger.info("Generating teacher response...")
    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,  # more deterministic
        top_p=0.9,
        do_sample=True,
    )

    # Decode and clean
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split(user_prompt)[-1].strip()

    # Remove "assistant" artifacts if present
    if response.lower().startswith("assistant"):
        response = response.split("\n", 1)[-1].strip()

    logger.debug(f"Teacher response: {response}")
    return response


## get dataset function for loading the dataset to get the teacher response.

def get_dataset(sample_size: int = 8000, seed: int = 42):
    """
    Load and sample Alpaca dataset for training.

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
        logger.warning("Requested sample size is larger than dataset. Returning full dataset.")

    return alpaca
