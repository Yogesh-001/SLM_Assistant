🧠 SLM Assistant – Knowledge Distillation with Teacher & Student LLMs

This project implements a Small Language Model (SLM) Assistant by applying Knowledge Distillation (KD) techniques.
A large Teacher LLM (Meta LLaMA-3.1 8B Instruct) is used to generate high-quality responses on instruction datasets, which are then used to train a smaller Student Model.

The goal is to build an efficient, edge-friendly personal knowledge assistant that can perform reasoning and instruction-following tasks without requiring heavy compute.

🚀 Features

📂 Dataset Preparation

Uses Alpaca Dataset
 (52k instructions).

Supports chunked dataset saving in .jsonl format for Colab/GPU memory constraints.

Saves two versions:

kd_dataset: distilled teacher responses.

mskd_dataset: teacher responses + ground truth.

🧑‍🏫 Teacher Model Inference

Loads Meta LLaMA-3.1-8B-Instruct with 4-bit quantization (bitsandbytes).

Generates teacher responses using Hugging Face Transformers.

🧑‍🎓 Student Model Training (WIP)

Student model learns from teacher-generated instruction-response pairs.

Evaluation pipeline for comparing ground truth vs teacher vs student.

🛠 Clean Project Structure

teacher_response/ → teacher model + dataset generation.

utils/ → reusable helper functions (e.g., logging).

main.py at root for a clean entry point.

📝 Logging Support

Centralized logging setup (utils/logger_config.py).

Replaces print() with structured logs.
