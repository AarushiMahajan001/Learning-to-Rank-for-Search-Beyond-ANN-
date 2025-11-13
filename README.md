# Learning-to-Rank-for-Search-Beyond-ANN-
This project implements a Learning-to-Rank (LTR) pipeline for search systems — combining dense retrievers, lexical baselines, and a Cross-Encoder re-ranker.
The goal is to improve how relevant search results are ordered using modern Transformer models.

We start from traditional BM25 retrieval, move to dense retrieval using Sentence Transformers + FAISS, and finally apply a re-ranking model fine-tuned on the MS MARCO dataset.
Performance is evaluated using nDCG@k and Recall@k metrics.

# What You’ll Learn

How search engines rank documents for a query.

How to build and evaluate both BM25 and bi-encoder (dense) retrieval models.

How to train a Cross-Encoder for re-ranking top results.

How to evaluate ranking quality using nDCG and Recall.

How to run the full experiment on Google Colab with a GPU.

# Tech Stack
| Component          | Library / Tool                                         |
| ------------------ | ------------------------------------------------------ |
| Dataset            | `mteb/msmarco` (Hugging Face)                          |
| Embeddings         | `sentence-transformers/msmarco-MiniLM-L6-v3`           |
| Dense Retrieval    | FAISS (`faiss-cpu`)                                    |
| Lexical Retrieval  | BM25 (`rank-bm25`)                                     |
| Re-Ranker          | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Training Framework | PyTorch Lightning                                      |
| Evaluation         | `sklearn.metrics.ndcg_score`                           |
| Visualization      | Matplotlib                                             |
| Environment        | Google Colab (T4 / A100 GPU)                           |

# Project Structure

learning-to-rank/
│
├── src/
│   └── models_ce.py               # Cross-Encoder Lightning model
│
├── data/
│   ├── raw/msmarco/               # Downloaded MS MARCO dataset
│   └── indices/msmarco_flat.index # FAISS dense retriever index
│
├── msmarco_pairs.parquet          # Query-document training pairs
├── lightning_logs/                # Model training checkpoints
├── learning_to_rank.ipynb         # Main Colab notebook
└── README.md                      # Project documentation

# Data Preparation

We load the MS MARCO dataset from Hugging Face and join:

Queries (queries_ds)

Documents (corpus_ds)

Relevance labels (qrels_ds)

This produces a dataset of (query, document, label) pairs:
query: "what is the Manhattan Project?"
document: "The Manhattan Project was a research project..."
label: 1
Saved as:
/content/msmarco_pairs.parquet

# Dense Retrieval with FAISS

Encode 1M documents using sentence-transformers/msmarco-MiniLM-L6-v3

Build a FAISS inner-product (cosine) index

Retrieve top-100 candidates per query
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "msmarco_flat.index")


# Lexical Retrieval (BM25)

A baseline using rank-bm25:

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized_corpus)
scores = bm25.get_scores(query.lower().split())

# Cross-Encoder Fine-Tuning

We fine-tune a re-ranker that reads both query and document together:
model = CrossEncoderLTR(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', lr=2e-5)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

This model learns pairwise ranking — predicting how relevant each document is for a given query.

# Evaluation Metrics

We compute nDCG@k and Recall@k for k = 10, 20, 50, 100:

| k   | nDCG@k | Recall@k |
| --- | ------ | -------- |
| 10  | 0.10   | 0.09     |
| 20  | 0.15   | 0.18     |
| 50  | 0.27   | 0.48     |
| 100 | 0.45   | 1.00     |

