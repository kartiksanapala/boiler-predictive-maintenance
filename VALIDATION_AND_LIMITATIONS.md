# Validation, limitations and interpretation

## Checks performed in the project workflow

- Dataset audit: timestamp structure, duplicate timestamps, missingness, negative/constant tags and time-series characteristics.
- Chronological split: 70% train, 15% validation, 15% test.
- Threshold calibration: validation 99th percentile; five-minute persistence with 80% exceedance requirement.
- Detector comparison: known synthetic event times and tags are compared with detector alerts.
- Final benchmark: PCA 7/24 (29.17%), Soft Sensor 13/24 (54.17%), SS-EWMA 9/24 (37.50%).
- RAG technical checks: index and metadata consistency, source/page metadata and top-k retrieval display.
- LLM checks: structured report output and citation-ID audit.

## What these checks do not establish

- No verified historical boiler fault or maintenance labels are available.
- Synthetic injections are not substitutes for actual failure events.
- The benchmark rates are not classification accuracy, precision, recall or F1 on real plant failures.
- The data covers only a short observation period (about five days in the audited source file).
- Sensor tag meanings and units require an authoritative dictionary; negative values must not be arbitrarily zeroed.
- Natural anomaly windows are not confirmed faults.
- A valid citation ID does not prove the passage supports the associated claim.
- Retrieval relevance and LLM faithfulness have not been comprehensively evaluated with expert-labelled test sets.
- No live plant integration, safety certification or autonomous control is implemented.

## Responsible wording

Say: “The prototype identifies unusual sensor windows and generates a document-grounded draft maintenance-support note.”

Do not say: “The model predicts boiler failures with 54.17% accuracy” or “the LLM identifies the root cause.”

## Before public release

Verify the original dataset URL and licence; confirm document reuse permissions; obtain an authoritative sensor-tag dictionary; record software versions; run the notebook from a clean runtime; and manually review generated citations against the exact source pages.
