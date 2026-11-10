# Boiler Predictive Maintenance — Anomaly Detection with RAG-Assisted Reporting

An academic proof-of-concept that combines multivariate sensor anomaly detection with retrieval-augmented generation (RAG) to prepare document-grounded maintenance-support notes for boiler-related operating data.

> **Scope:** This is a research prototype, not an industrially validated predictive-maintenance system. It identifies unusual sensor windows and drafts a report for qualified human review. It does not diagnose confirmed faults, predict failure probability or remaining useful life, issue operating instructions, or control plant equipment.

## Project goals

1. Audit and prepare timestamped multivariate sensor data.
2. Compare robust statistical and machine-learning anomaly detectors.
3. Convert persistent anomaly alerts into reviewable event episodes.
4. Retrieve relevant passages from permitted boiler references.
5. Use a local instruction-tuned language model to draft a structured report with source references and explicit uncertainty.
6. Document the workflow so the analysis can be inspected and reproduced.

## Repository structure

```text
.
├── notebooks/   # Consolidated end-to-end notebook
├── src/          # Numerical pipeline and RAG/LLM scripts
├── data/
│   ├── raw/      # Place the permitted source CSV here
│   └── processed/
├── documents/    # Place permitted PDF/TXT/MD reference manuals here
├── outputs/      # Generated run outputs (not committed)
├── results/      # Final checked synthetic benchmark summary
├── reports/      # Detailed guide and final project report
├── presentation/ # Project slides
├── docs/         # Architecture, methodology, validation notes
├── requirements.txt
└── README.md
```

## System architecture

The workflow has two linked branches:

- **Numerical branch:** sensor CSV → data audit and preprocessing → anomaly scores → validation-calibrated threshold and persistence rule → event episodes.
- **Knowledge branch:** permitted manuals → text extraction with source/page metadata → overlapping chunks → Sentence Transformers embeddings → FAISS similarity search → top retrieved passages.
- **Report generation:** episode summary + retrieved passages → local Qwen2.5-1.5B-Instruct → structured draft maintenance-support note → human review.

Pandas/NumPy handle data processing; scikit-learn provides anomaly-detection and regression methods; Sentence Transformers creates embeddings; FAISS searches vectors; Qwen drafts text. The LLM is not the anomaly detector.

## Methodology at a glance

- One-minute median aggregation and limited forward filling.
- Chronological 70% / 15% / 15% train-validation-test split.
- MAD, Isolation Forest, PCA, Soft Sensor (Ridge residuals) and SS-EWMA methods.
- Threshold calibrated at the 99th percentile of validation scores.
- Five-minute persistence rule requiring at least four of five exceedances.
- Synthetic benchmark with 24 controlled events across three benches.
- RAG uses 220-word chunks with 40-word overlap, `all-MiniLM-L6-v2` embeddings and top-5 retrieval.
- Local generation uses `Qwen/Qwen2.5-1.5B-Instruct`.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and the detailed Word guide in [reports/](reports/).

## Final checked synthetic benchmark

| Detector | Events detected | Total injected events | Event detection rate |
|---|---:|---:|---:|
| PCA | 7 | 24 | 29.17% |
| Soft Sensor | 13 | 24 | 54.17% |
| SS-EWMA | 9 | 24 | 37.50% |

These are event-detection rates on the selected artificial patterns. They are **not** accuracy, precision, recall, F1, or real-world boiler-failure prediction performance. Soft Sensor was selected as the primary episode-generation detector because it detected the most events in this particular benchmark; this does not establish universal superiority.

## Setup

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For Google Colab, open `notebooks/Boiler_Predictive_Maintenance_Final_Consolidated.ipynb` and run the cells in order. Install the dependencies in the notebook and upload or mount the permitted dataset and reference documents when prompted.

## Data and document requirements

The original sensor CSV and boiler manuals are intentionally not included in this public repository. Before adding or redistributing them, verify their source, licence and permissions. Place the dataset in the location requested by the notebook/pipeline and place permitted PDF/TXT/Markdown references in `documents/`.

The audited source dataset contained approximately five days of 5-second observations. Sensor tag meanings and units were not independently verified; consult an authoritative tag dictionary before assigning physical interpretations. Do not replace negative readings with zero without engineering justification.

## Running the scripts

The numerical pipeline is in `src/boiler_pipeline.py`. The RAG index and report workflow is in `src/rag_llm.py`. Inspect each script's command-line help and configuration before running:

```bash
python src/boiler_pipeline.py --help
python src/rag_llm.py --help
```

The notebook is the primary end-to-end reproduction entry point. Run-specific plots, episode tables, indexes and generated reports are produced by executing the workflow; they are not represented as pre-existing results in this repository.

## Validation and limitations

The workflow checks timestamp quality, missingness, chronological splitting, validation-based thresholding, synthetic-event overlap, retrieval metadata and report structure. A citation-ID audit only checks that cited passage identifiers exist; it does not prove semantic support. Retrieved passages and generated claims require manual verification.

There are no verified historical fault labels or maintenance outcomes. Synthetic injections are not actual failures; natural anomaly candidates are not confirmed faults. The dataset covers a short period. There is no live plant integration, expert-validated diagnosis, safety certification, autonomous control or remaining-useful-life estimation.

Read [docs/VALIDATION_AND_LIMITATIONS.md](docs/VALIDATION_AND_LIMITATIONS.md) before interpreting results.

## Reproducibility checklist

1. Obtain the same source dataset and verify its provenance/licence.
2. Obtain permitted reference manuals and retain their source/page metadata.
3. Install the listed dependencies in a clean environment.
4. Run the consolidated notebook from the first cell to the last.
5. Retain the generated configuration, benchmark tables, episode outputs and RAG reports.
6. Compare the synthetic benchmark event counts with the checked summary in `results/`.
7. Manually verify retrieved source passages and every citation in generated reports.

## Project documents

- [Detailed project goals, architecture, methodology and reproduction guide](reports/Detailed_Project_Goals_Architecture_Methodology_and_Reproduction_Guide.docx)
- [Final project report (DOCX)](reports/Final_Project_Report.docx)
- [Final project report (PDF)](reports/Final_Project_Report.pdf)
- [Project presentation](presentation/Project_Presentation.pptx)
- [System architecture](docs/ARCHITECTURE.md)
- [Methodology](docs/METHODOLOGY.md)
- [Validation and limitations](docs/VALIDATION_AND_LIMITATIONS.md)

## Responsible-use note

This prototype is intended for academic analysis and human-reviewed decision support only. It must not be used to make plant operating or maintenance decisions without qualified engineering assessment.
