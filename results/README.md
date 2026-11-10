# Benchmark results

The CSV and JSON files in this directory preserve the final checked event-level benchmark summary for 24 controlled synthetic events (three benchmark benches, eight events each).

| Detector | Detected | Total | Event detection rate |
|---|---:|---:|---:|
| PCA | 7 | 24 | 29.17% |
| Soft Sensor | 13 | 24 | 54.17% |
| SS-EWMA | 9 | 24 | 37.50% |

These values are synthetic-event detection rates, not real-world failure-prediction accuracy or classification metrics. The original dataset has no verified historical fault labels. See the validation and limitations document before interpreting them.
