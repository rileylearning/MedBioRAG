# 📂 Data Directory Setup Guide

This project requires multiple biomedical datasets for evaluation and experimentation. Please follow the instructions below to set up the `data/` directory properly.

## 📥 Step 1: Download Source Datasets
Download the original source datasets from the following official links:

### Close-Ended QA
- [MedQA](https://github.com/jind11/MedQA)
- [PubMedQA](https://github.com/pubmedqa/pubmedqa)
- [BioASQ](https://participants-area.bioasq.org/)

### Long-Form QA
- [MedicationQA](https://github.com/abachaa/Medication_QA_MedInfo2019)
- [LiveQA](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017)

### Retrieval Datasets
- [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
- [TREC-COVID](https://ir.nist.gov/trec-covid/)

---

## 📁 Step 2: Organize Data into Folders
Once the datasets are downloaded, organize them into the following directory structure under the `data/` folder:

```
data/
├── close-ended/
│   ├── bioasq/           # BioASQ QA pairs
│   ├── medqa/            # MedQA question set
│   ├── pubmedqa/         # PubMedQA yes/no/maybe questions
├── long-form/
│   ├── bioasq/           # BioASQ long-form answers (if applicable)
│   ├── liveqa/           # LiveQA question-answer pairs
│   ├── medicationqa/     # MedicationQA long-form answers
│   ├── pubmedqa/         # PubMedQA long-form answers
├── retrieval/
│   ├── nfcorpus/         # NFCorpus raw + qrels
│   ├── trec-covid/       # TREC-COVID metadata, qrels, topics
```

> 📌 **Note:** Use the attached preprocessing scripts (e.g., `preprocess_close-ended.py`, `bioasq.py`, `liveqa.py`) to convert raw data into the format used for training and evaluation.

---

## 📂 Scripts
The following Python scripts located under each subdirectory help preprocess the datasets:

- `data/close-ended/`: Handles multiple-choice or yes/no/maybe QA formats.
- `data/long-form/`: Prepares long-form QA datasets for generative evaluation.
- `data/retrieval/`: Converts corpus and query files for RAG and retrieval evaluation.

Make sure the datasets are placed in the correct folders before running any preprocessing or evaluation pipeline.


