# System architecture

## Two linked branches

**Numerical branch:** timestamped sensor CSV → audit and preprocessing → anomaly scores → threshold/persistence → event episodes.

**Knowledge branch:** permitted boiler references → page-wise text extraction → overlapping chunks → Sentence Transformers embeddings → FAISS vector index → top-k passages.

**Integration:** episode summary + retrieved passages → constrained local Qwen2.5-1.5B-Instruct prompt → structured draft maintenance-support report → human review.

## Responsibilities

- **Pandas/NumPy:** ingestion, time alignment, aggregation and numerical transformations.
- **scikit-learn:** MAD calculations, Isolation Forest, PCA, linear/ridge regression, scaling and time-series splitting.
- **Sentence Transformers:** maps document chunks and queries to embedding vectors.
- **FAISS:** performs vector similarity search; it does not generate or interpret maintenance advice.
- **Qwen:** drafts a structured explanation from the episode and retrieved evidence; it is not the anomaly detector.

## Safety boundary

The system produces an academic triage note, not an engineering diagnosis, work order, operating instruction or autonomous control action.
