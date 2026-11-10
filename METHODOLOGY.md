# Methodology and algorithm rationale

## Data preparation

The pipeline parses the `date` timestamp, sorts observations, selects numeric columns, aggregates to one-minute intervals using the median, and forward-fills at most three intervals. It uses 70/15/15 chronological train/validation/test partitions. The configuration defines two context tags and four indicator tags; tag semantics and units are not assumed to be known.

## Detectors

- **MAD:** robust per-feature deviation baseline using median and scaled median absolute deviation.
- **Isolation Forest:** unsupervised tree-based baseline that isolates unusual observations.
- **PCA:** multivariate baseline retaining 95% training variance; combines Hotelling T² and squared prediction error (SPE).
- **Soft Sensor:** one Ridge regression per target sensor, predicting it from all other monitored sensors plus context variables. Robustly scaled out-of-fold residuals provide an inconsistency score. Five-fold `TimeSeriesSplit` uses a 30-minute gap.
- **SS-EWMA:** smooths residual-related signals with lambda 0.1 to emphasise sustained deviations.

Thresholds are selected at the 99th percentile of validation scores. A five-minute persistence rule requires at least 80% of observations (at least four of five) to exceed the threshold.

## Synthetic evaluation

Three benchmark benches contain eight events each (24 total). Injected patterns include drift, offset, noise burst, stuck sensor and group shift. The final checked results are PCA 7/24, Soft Sensor 13/24 and SS-EWMA 9/24. These are sensitivity results for the selected artificial patterns only.

## RAG and LLM

PDF/TXT/Markdown references are extracted with source/page metadata. Text is chunked into 220-word segments with 40-word overlap. `all-MiniLM-L6-v2` generates normalised embeddings. FAISS `IndexFlatIP` retrieves top-k passages; inner product on normalised vectors corresponds to cosine similarity. Local Qwen2.5-1.5B-Instruct drafts a structured report with observed evidence, possible interpretations, document evidence, inspection questions and uncertainty/human-review notes.
