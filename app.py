import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

# -----------------------------------------------------------------------
# Jarvis-Lite Model Loader
# -----------------------------------------------------------------------
def load_jarvis():
    adapter_path = "./jarvis_lite_final"
    base_model_id = "Qwen/Qwen2-1.5B-Instruct"
    
    if not os.path.exists(adapter_path):
        logger.warning("Fine-tuned adapters not found. Loading base model for demo.")
        model_path = base_model_id
        is_peft = False
    else:
        model_path = adapter_path
        is_peft = True

    logger.info(f"Loading Jarvis-Lite from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
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
    
    if is_peft:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
        
    model.eval()
    return tokenizer, model

# -----------------------------------------------------------------------
# Global Model Instances
# -----------------------------------------------------------------------
import os
tokenizer, model = load_jarvis()

def chat_with_jarvis(message, history):
    system_prompt = (
        "You are Jarvis, a highly intelligent and precise AI assistant. "
        "When given a task, you MUST follow this exact two-part format:\n\n"
        "[THOUGHT]: First, briefly reason through the problem step-by-step.\n"
        "[RESPONSE]: Then, provide the final, direct answer."
    )
    
    # Construct conversation history
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Use streaming for that Jarvis "real-time" feel
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# -----------------------------------------------------------------------
# Jarvis-Style Premium UI (Gradio)
# -----------------------------------------------------------------------
custom_css = """
body { background-color: #0b0f19; color: #e0e0e0; font-family: 'Inter', sans-serif; }
.gradio-container { border: 1px solid #1f2937; border-radius: 12px; background: rgba(17, 24, 39, 0.8); backdrop-filter: blur(10px); }
#title { text-align: center; color: #38bdf8; text-shadow: 0 0 10px rgba(56, 189, 248, 0.5); font-size: 2.5em; margin-bottom: 0.5em; }
#subtitle { text-align: center; color: #94a3b8; margin-bottom: 2em; }
.message.user { background-color: #1e293b !important; border-left: 4px solid #38bdf8 !important; }
.message.bot { background-color: #0f172a !important; border-left: 4px solid #10b981 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate")) as demo:
    gr.HTML("<h1 id='title'>JARVIS-LITE</h1>")
    gr.HTML("<p id='subtitle'>Small Language Model Assistant (Distilled from Llama-3.1-8B)</p>")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_jarvis,
        chatbot=gr.Chatbot(height=500, show_label=False),
        textbox=gr.Textbox(placeholder="What is your command, Sir?", container=False, scale=7),
        retry_btn="🔄 Recalibrate",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear Memory",
    )
    
    gr.Markdown("""
    ### System Status:
    - **Architecture**: Qwen2-1.5B (Fine-tuned with QLoRA)
    - **Optimization**: 4-bit Quantized (BitsAndBytes)
    - **Reasoning Engine**: Active (Chain-of-Thought)
    """)

if __name__ == "__main__":
    demo.launch()
