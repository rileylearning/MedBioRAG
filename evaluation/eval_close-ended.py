import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()

# Azure OpenAI Endpoint
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Output config
TASK_NAME = "close-ended-rag-eval"
OUTPUT_DIR = os.path.join("evaluation", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared model generation config
MODEL_SETTINGS = {
    "max_tokens": 800,
    "temperature": 0.7,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Dataset-specific configuration
DATASET_CONFIGS = {
    "medqa": {
        "path": "data/close-ended/medqa/medqa_test.csv",
        "type": "mc",
        "finetuned_model": os.getenv("FINETUNED_MODEL_MEDQA"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_MEDQA"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_MEDQA"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_MEDQA"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_MEDQA")
    },
    "pubmedqa": {
        "path": "data/close-ended/pubmedqa/pubmedqa_test.csv",
        "type": "ynm",
        "finetuned_model": os.getenv("FINETUNED_MODEL_PUBMEDQA"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_PUBMEDQA"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_PUBMEDQA"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_PUBMEDQA"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_PUBMEDQA")
    },
    "bioasq": {
        "path": "data/close-ended/bioasq/bioasq_test.csv",
        "type": "yesno",
        "finetuned_model": os.getenv("FINETUNED_MODEL_BIOASQ"),
        "base_models": ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-35-turbo-16k"],
        "search_index": os.getenv("SEARCH_INDEX_BIOASQ"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_BIOASQ"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_BIOASQ"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_BIOASQ")
    }
}

def clean_response(text):
    text = re.sub(r'\[doc\d+\]', '', text)
    text = re.sub(r'[\[\]]', '', text)
    return text.strip().lower()

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
    print(f"Azure Search Error {response.status_code}: {response.text}")
    return []

def build_prompt(query, context, dtype, use_rag):
    if dtype == "mc":
        instruction = "Answer with A, B, C, or D. Do not add anything else."
    elif dtype == "yesno":
        instruction = "Answer only with 'yes' or 'no'. Do not add anything else."
    elif dtype == "ynm":
        instruction = "Answer only with 'yes', 'no', or 'maybe'. Do not add anything else."
    else:
        raise ValueError("Unsupported type")

    if use_rag:
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
        ]
    else:
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
        ]

def ask_question(query, config, deployment_name, dtype, use_rag):
    context = ""
    if use_rag:
        retrieved = azure_search(query, config)
        context = "\n\n".join(retrieved) if retrieved else "No relevant documents found."

    messages = build_prompt(query, context, dtype, use_rag)

    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
    payload = {
        "model": deployment_name,
        "messages": messages,
        **MODEL_SETTINGS
    }

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-05-15"
    retries, delay = 5, 2
    for attempt in range(retries):
        try:
            res = requests.post(url, headers=headers, json=payload)
            if res.status_code == 200:
                return clean_response(res.json()["choices"][0]["message"]["content"])
            elif res.status_code == 429:
                print(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise Exception(f"OpenAI Error {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Error: {e}")
    return "error"

def evaluate_model(test_csv_path, config, deployment_name, use_rag):
    df = pd.read_csv(test_csv_path)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        query, gold = row["question"], row["answer"].strip().lower()
        pred = ask_question(query, config, deployment_name, config["type"], use_rag)
        correct = int(pred == gold)
        results.append({"question": query, "answer": gold, "prediction": pred, "correct": correct})

    result_df = pd.DataFrame(results)
    accuracy = round(accuracy_score(result_df['answer'], result_df['prediction']), 5)
    return accuracy, result_df

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d")

    for name, config in DATASET_CONFIGS.items():
        test_path = config["path"]

        # Finetuned model + RAG
        acc_rag, df_rag = evaluate_model(test_path, config, config["finetuned_model"], use_rag=True)
        rag_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{TASK_NAME}_{name}_finetuned+RAG_acc_{acc_rag*100:.2f}.csv")
        df_rag.to_csv(rag_path, index=False)
        print(f"Saved Fine-Tuned + RAG results to {rag_path}")

        # Finetuned model only (no RAG)
        acc_base, df_base = evaluate_model(test_path, config, config["finetuned_model"], use_rag=False)
        base_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{TASK_NAME}_{name}_finetuned_only_acc_{acc_base*100:.2f}.csv")
        df_base.to_csv(base_path, index=False)
        print(f"Saved Fine-Tuned Only results to {base_path}")

        # Base models (with and without RAG)
        for model in config["base_models"]:
            acc_b, df_b = evaluate_model(test_path, config, model, use_rag=False)
            out_b = os.path.join(OUTPUT_DIR, f"{timestamp}_{TASK_NAME}_{name}_{model}_BASE_acc_{acc_b*100:.2f}.csv")
            df_b.to_csv(out_b, index=False)
            print(f"Saved Base Model Only results to {out_b}")

            acc_r, df_r = evaluate_model(test_path, config, model, use_rag=True)
            out_r = os.path.join(OUTPUT_DIR, f"{timestamp}_{TASK_NAME}_{name}_{model}_RAG_acc_{acc_r*100:.2f}.csv")
            df_r.to_csv(out_r, index=False)
            print(f"Saved Base Model + RAG results to {out_r}")