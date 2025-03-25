import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# Load environment variables
load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Output config
TASK_NAME = "long-form-rag-eval"
OUTPUT_DIR = os.path.join("evaluation", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model generation settings (long-form specific)
MODEL_SETTINGS = {
    "max_tokens": 2999,
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Evaluation metrics
rouge = evaluate.load("rouge")
bleurt = evaluate.load("bleurt", config_name="bleurt-large-512")

# Dataset configuration
DATASET_CONFIGS = {
    "bioasq": {
        "path": "data/long-form/bioasq_test.csv",
        "finetuned_model": os.getenv("FINETUNED_MODEL_BIOASQ_LF"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_BIOASQ_LF"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_BIOASQ_LF"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_BIOASQ_LF"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_BIOASQ")
    },
    "liveqa": {
        "path": "data/long-form/liveqa_test.csv",
        "finetuned_model": os.getenv("FINETUNED_MODEL_LIVEQA"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_LIVEQA"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_LIVEQA"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_LIVEQA"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_LIVEQA")
    },
    "medicationqa": {
        "path": "data/long-form/medicationqa_test.csv",
        "finetuned_model": os.getenv("FINETUNED_MODEL_MEDICATIONQA"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_MEDICATIONQA"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_MEDICATIONQA"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_MEDICATIONQA"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_MEDICATIONQA")
    },
    "pubmedqa": {
        "path": "data/long-form/pubmedqa_test.csv",
        "finetuned_model": os.getenv("FINETUNED_MODEL_PUBMEDQA_LF"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_PUBMEDQA_LF"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_PUBMEDQA_LF"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_PUBMEDQA_LF"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_PUBMEDQA")
    }
}

def azure_search(query, config, top_k=10):
    headers = {"Content-Type": "application/json", "api-key": config["search_key"]}
    payload = {
        "search": query,
        "top": top_k,
        "queryType": "semantic",
        "semanticConfiguration": config["semantic_config"]
    }
    url = f"{config['search_endpoint']}/indexes/{config['search_index']}/docs/search?api-version=2021-04-30-Preview"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return [doc.get("content", "") for doc in response.json().get("value", [])]
    return []

def build_prompt_longform(query, context, use_rag):
    if use_rag:
        system_msg = "You are a medical and biological expert. Use the retrieved context to answer the question accurately."
        user_msg = f"Question: {query}\n\nContext: {context}"
    else:
        system_msg = "You are a medical and biological expert. Answer the question accurately."
        user_msg = query

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def generate_answer(query, config, model_name, use_rag=True):
    context = ""
    if use_rag:
        retrieved = azure_search(query, config)
        context = "\n\n".join(retrieved) if retrieved else "No relevant documents found."

    messages = build_prompt_longform(query, context, use_rag)

    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    payload = {"model": model_name, "messages": messages, **MODEL_SETTINGS}
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{model_name}/chat/completions?api-version=2023-05-15"

    try:
        res = requests.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Generation Error: {e}")
    return ""

def evaluate_longform_model(dataset_name, config, model_name, use_rag):
    test_data = pd.read_csv(config["path"])
    results = []
    smoothing = SmoothingFunction().method1

    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        query, true_answer = row["question"], row["answer"].strip()
        generated = generate_answer(query, config, model_name, use_rag)

        rouge_scores = rouge.compute(predictions=[generated], references=[true_answer])
        bleu_score = sentence_bleu([true_answer.split()], generated.split(), smoothing_function=smoothing)
        _, _, bert_f1 = bert_score([generated], [true_answer], lang="en", rescale_with_baseline=True)
        bleurt_score = bleurt.compute(predictions=[generated], references=[true_answer])["scores"][0]

        results.append({
            "question": query,
            "answer": true_answer,
            "prediction": generated,
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "BLEU": bleu_score,
            "BERTScore": bert_f1.mean().item(),
            "BLEURT": bleurt_score
        })

    df = pd.DataFrame(results)
    model_tag = f"{model_name}_{'RAG' if use_rag else 'BASE'}"
    date = datetime.now().strftime('%Y%m%d')
    out_path = os.path.join(OUTPUT_DIR, f"{date}_{TASK_NAME}_{dataset_name}_{model_tag}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    for name, config in DATASET_CONFIGS.items():
        # Fine-tuned model
        evaluate_longform_model(name, config, config["finetuned_model"], use_rag=True)
        evaluate_longform_model(name, config, config["finetuned_model"], use_rag=False)

        # Base models
        for base in config["base_models"]:
            evaluate_longform_model(name, config, base, use_rag=True)
            evaluate_longform_model(name, config, base, use_rag=False)