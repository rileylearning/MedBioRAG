# 📦 Unified Preprocessing Scripts for BioASQ, MedQA, PubMedQA (clean path structure)

import os
import json
import pandas as pd

# Base prompt templates
BIOASQ_PROMPT = "You are a multilingual medical assistant. Answer only with 'yes' or 'no'. Do not add anything else."
PUBMEDQA_PROMPT = "You are a multilingual medical assistant. Answer only with 'yes', 'maybe', or 'no'. Do not add anything else."
MEDQA_PROMPT = "You are taking the United States Medical Licensing Examination (USMLE). Respond with only one option (A, B, C, D, or E) based on the question and options provided."

def process_bioasq():
    train_input_path = os.path.join("data", "bioasq", "BioASQ12-task-b", "BioASQ-training12b", "training12b_new.json")
    test_input_path = os.path.join("data", "bioasq", "BioASQ12-task-b", "Task12BGoldenEnriched", "12B4_golden.json")
    output_dir = os.path.join("data", "close-ended", "bioasq")
    os.makedirs(output_dir, exist_ok=True)

    def process_questions(data, include_context=False):
        jsonl_data, csv_data = [], []
        for idx, q in enumerate(data.get("questions", [])):
            answer = q.get("exact_answer")
            if answer not in ["yes", "no"]:
                continue
            user_content = f"Question: {q.get('body')}"
            if include_context:
                context = " ".join(s.get("text", "") for s in q.get("snippets", []))
                user_content += f"\nContext: {context}"
            jsonl_data.append({
                "messages": [
                    {"role": "system", "content": BIOASQ_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ]
            })
            csv_data.append({"question": q.get("body"), "answer": answer})
        return jsonl_data, csv_data

    with open(train_input_path, 'r') as f:
        train_jsonl, train_csv = process_questions(json.load(f), include_context=False)
    with open(test_input_path, 'r') as f:
        test_jsonl, test_csv = process_questions(json.load(f), include_context=True)

    pd.DataFrame(train_csv).to_csv(os.path.join(output_dir, "bioasq_train.csv"), index=False)
    pd.DataFrame(test_csv).to_csv(os.path.join(output_dir, "bioasq_test.csv"), index=False)
    with open(os.path.join(output_dir, "bioasq_train.jsonl"), 'w') as f:
        for line in train_jsonl: json.dump(line, f); f.write('\n')
    with open(os.path.join(output_dir, "bioasq_test.jsonl"), 'w') as f:
        for line in test_jsonl: json.dump(line, f); f.write('\n')


def process_pubmedqa():
    train_input_path = os.path.join("data", "pubmedqa", "ori_pqaa.json")
    test_input_path = os.path.join("data", "pubmedqa", "ori_pqal.json")
    output_dir = os.path.join("data", "close-ended", "pubmedqa")
    os.makedirs(output_dir, exist_ok=True)

    def process(data_dict, include_id=True):
        jsonl_data, csv_data = [], []
        for idx, (key, entry) in enumerate(data_dict.items()):
            if entry.get("reasoning_required_pred") != "yes": continue
            decision = entry.get("final_decision")
            if decision not in ["yes", "no", "maybe"]: continue
            context = " ".join(entry.get("CONTEXTS", []))
            question = entry.get("QUESTION", "")
            user_content = f"{context} {question}".strip()
            jsonl_data.append({
                "messages": [
                    {"role": "system", "content": PUBMEDQA_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": decision}
                ]
            })
            row = {"question": question, "answer": decision}
            if include_id: row["id"] = idx + 1
            csv_data.append(row)
        return jsonl_data, csv_data

    with open(train_input_path, 'r') as f:
        train_jsonl, train_csv = process(json.load(f), include_id=True)
    with open(test_input_path, 'r') as f:
        test_jsonl, test_csv = process(json.load(f), include_id=False)

    pd.DataFrame(train_csv).to_csv(os.path.join(output_dir, "pubmedqa_train.csv"), index=False)
    pd.DataFrame(test_csv).to_csv(os.path.join(output_dir, "pubmedqa_test.csv"), index=False)
    with open(os.path.join(output_dir, "pubmedqa_train.jsonl"), 'w') as f:
        for line in train_jsonl: json.dump(line, f); f.write('\n')
    with open(os.path.join(output_dir, "pubmedqa_test.jsonl"), 'w') as f:
        for line in test_jsonl: json.dump(line, f); f.write('\n')


def process_medqa():
    input_base = os.path.join("data", "medqa", "questions", "US")
    output_base = os.path.join("data", "close-ended", "medqa")
    os.makedirs(output_base, exist_ok=True)

    splits = {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"}
    out_names = {k: f"medqa_{k}.jsonl" for k in splits}

    for split, fname in splits.items():
        jsonl_data, csv_rows = [], []
        with open(os.path.join(input_base, fname), 'r') as f:
            for line in f:
                entry = json.loads(line)
                question = entry["question"]
                options = entry["options"]
                answer = entry["answer_idx"]
                user_content = question + "\nOptions:" + ''.join([f"\n{k}) {v}" for k,v in options.items()])
                jsonl_data.append({
                    "messages": [
                        {"role": "system", "content": MEDQA_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer}
                    ]
                })
                if split == "test":
                    csv_rows.append({"question": question, "answer": answer})

        with open(os.path.join(output_base, out_names[split]), 'w') as f:
            for item in jsonl_data: json.dump(item, f); f.write('\n')
        if split == "test":
            pd.DataFrame(csv_rows).to_csv(os.path.join(output_base, "medqa_test.csv"), index=False)


if __name__ == '__main__':
    print("Preprocessing datasets...")
    process_bioasq()
    process_pubmedqa()
    process_medqa()
    print("All datasets processed and saved under data/close-ended/")
