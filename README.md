# 🧠 SLM Assistant (Small Language Model Assistant)

This project implements a **Teacher-Student Knowledge Distillation pipeline** for training a small language model (SLM) using instruction-following data. The teacher model is a large LLaMA-based model from Hugging Face, and the student is trained on a distilled dataset to make it lightweight and suitable for **edge devices**.

---

## 📂 Project Structure

```
SLM_ASSISTANT/
│── teacher_response/         # Teacher model inference and dataset generation
│   │── teacher_model.py      # Loads teacher model with quantization
│   │── get_response.py       # Functions to query teacher responses
│   │── teacher_response.py   # Main script to generate distillation datasets
│   │── __init__.py
│
│── utils/                    # Utility functions
│   │── logger_config.py      # Logger setup for the whole project
│   │── __init__.py
│
│── student/                  # (Planned) Student training and evaluation pipeline
│
│── teacher_datasets/         # Saved knowledge distillation datasets (.jsonl)
│
│── README.md                 # Project documentation
```

---

## 🚀 Features

- ✅ Loads **Meta LLaMA-3.1 8B Instruct** as the teacher model  
- ✅ Generates **teacher-student distillation datasets** from Alpaca  
- ✅ Saves datasets in **JSONL format** (knowledge distillation + mixed dataset)  
- ✅ Supports **quantization with 4-bit (bitsandbytes)** for efficient inference  
- ✅ Logging system for clean, trackable pipeline execution  

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB/SLM_ASSISTANT.git
   cd SLM_ASSISTANT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate    # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your **Hugging Face Token** in `.env`:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   ```

---

## 📊 Usage

Run the dataset generation script from the project root:
```bash
python -m teacher_main.py
```

This will:
- Load the teacher model (`meta-llama/Llama-3.1-8B-Instruct`)
- Query Alpaca dataset samples
- Save teacher responses into `teacher_datasets/`

Example dataset entry (`alpaca_mskd_chunk0.jsonl`):
```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "ground_truth": "1. Eat a balanced diet ...",
  "teacher_response": "1. Drink plenty of water..."
}
```

---

## 🔮 Next Steps

- Train the **student model** on the distilled dataset  
- Evaluate student vs teacher responses  
- Optimize for **edge deployment**  

---

## 📜 License

Developed by **Yogesh Murala**
