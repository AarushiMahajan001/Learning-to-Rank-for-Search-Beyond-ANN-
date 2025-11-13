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

# Key Steps
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

