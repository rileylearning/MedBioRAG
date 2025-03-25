import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Load credentials
load_dotenv()

# Constants
K = 10
MAX_QUERY_LENGTH = 2999
TASK_NAME = "retrieval-eval"
OUTPUT_DIR = os.path.join("evaluation", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Azure config for each dataset
DATASET_CONFIGS = {
    "nfcorpus": {
        "qrels_path": "data/retrieval/nfcorpus_qrels.csv",
        "queries_path": "data/long-form/nfcorpus_test.csv",
        "query_id_col": "query_id",
        "query_text_col": "question",
        "doc_id_col": "doc_id",
        "relevance_col": "relevance",
        "search_index": os.getenv("SEARCH_INDEX_NFCORPUS"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_NFCORPUS"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_NFCORPUS"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_NFCORPUS")
    },
    "trec-covid": {
        "qrels_path": "data/retrieval/qrels-covid_d5_j0.5-5_filtered.csv",
        "queries_path": "data/retrieval/topics-rnd5.csv",
        "query_id_col": "query-id",
        "query_text_col": "query",
        "doc_id_col": "doc_id",
        "relevance_col": "relevance",
        "search_index": os.getenv("SEARCH_INDEX_TREC_COVID"),
        "semantic_config": os.getenv("SEMANTIC_CONFIG_TREC_COVID"),
        "search_endpoint": os.getenv("SEARCH_ENDPOINT_TREC_COVID"),
        "search_key": os.getenv("AZURE_SEARCH_API_KEY_TREC_COVID")
    }
}

def truncate_query(query):
    return " ".join(str(query).split()[:MAX_QUERY_LENGTH])

def search_azure(query, config, mode="semantic"):
    url = f"{config['search_endpoint']}/indexes/{config['search_index']}/docs/search?api-version=2021-04-30-Preview"
    headers = {"Content-Type": "application/json", "api-key": config["search_key"]}
    payload = {
        "search": truncate_query(query),
        "top": K,
        "queryType": "semantic" if mode == "semantic" else "simple",
        "semanticConfiguration": config["semantic_config"] if mode == "semantic" else "default"
    }
    if mode == "semantic":
        payload.update({"captions": "extractive", "answers": "extractive|count-3"})
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return [doc.get("id") or doc.get("doc_id") for doc in response.json().get("value", [])]
    else:
        print(f"Search Error ({mode}) {response.status_code}: {response.text}")
        return []

# Evaluation metrics
def dcg_at_k(rels, k):
    rels = np.array(rels[:k], dtype=np.float32)
    return np.sum(rels / np.log2(np.arange(2, len(rels) + 2)))

def ndcg_at_k(actual, ideal, k):
    return dcg_at_k(actual, k) / dcg_at_k(sorted(ideal, reverse=True), k) if ideal else 0.0

def mrr_at_k(rels, k):
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1 / (i + 1)
    return 0.0

def precision_at_k(rels, k):
    return sum(r > 0 for r in rels[:k]) / k

def recall_at_k(actual, ideal, k):
    return sum(r > 0 for r in actual[:k]) / len(ideal) if ideal else 0.0

def f1_score_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def map_at_k(rels, k):
    hits = sum_prec = 0
    for i, r in enumerate(rels[:k]):
        if r > 0:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / hits if hits else 0.0

def evaluate_search(name, config, mode="semantic"):
    queries = pd.read_csv(config["queries_path"])
    qrels = pd.read_csv(config["qrels_path"])
    results = []

    for _, row in tqdm(queries.iterrows(), total=len(queries), desc=f"{name.upper()} ({mode})"):
        qid = row[config["query_id_col"]]
        query = row[config["query_text_col"]]
        retrieved = search_azure(query, config, mode)

        actual = [
            qrels[(qrels[config["query_id_col"]] == qid) & (qrels[config["doc_id_col"]] == doc)][config["relevance_col"]].values[0]
            if not qrels[(qrels[config["query_id_col"]] == qid) & (qrels[config["doc_id_col"]] == doc)].empty else 0
            for doc in retrieved
        ]

        ideal = qrels[qrels[config["query_id_col"]] == qid][config["relevance_col"]].sort_values(ascending=False).tolist()

        p = precision_at_k(actual, K)
        r = recall_at_k(actual, ideal, K)

        results.append({
            "query_id": qid,
            "DCG@10": dcg_at_k(actual, K),
            "NDCG@10": ndcg_at_k(actual, ideal, K) * 100,
            "MRR@10": mrr_at_k(actual, K) * 100,
            "Precision@10": p * 100,
            "Recall@10": r * 100,
            "F1-score@10": f1_score_at_k(p, r) * 100,
            "MAP@10": map_at_k(actual, K) * 100
        })

    df = pd.DataFrame(results)
    avg = df.drop(columns=["query_id"]).mean().to_dict()
    avg["query_id"] = "Average"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    tag = f"{mode.upper()}_{round(avg['NDCG@10'])}_{round(avg['MRR@10'])}"
    today = datetime.today().strftime('%Y%m%d')
    out_path = os.path.join(OUTPUT_DIR, f"{today}_{TASK_NAME}_{name}_{tag}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {mode} results to: {out_path}\n")
    print(df.tail(1))

if __name__ == "__main__":
    for dataset, cfg in DATASET_CONFIGS.items():
        evaluate_search(dataset, cfg, mode="semantic")
        evaluate_search(dataset, cfg, mode="keyword")
