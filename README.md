# Results included in this repository

`synthetic_benchmark_summary.csv` and `.json` preserve the final checked benchmark figures reported in the project notebook/report. They are event-level detection counts for 24 controlled synthetic events (three benchmark benches, eight events each).

| Detector | Detected | Total | Event detection rate |
|---|---:|---:|---:|
| PCA | 7 | 24 | 29.17% |
| Soft Sensor | 13 | 24 | 54.17% |
| SS-EWMA | 9 | 24 | 37.50% |

These figures are not accuracy, precision, recall, F1, or real-world boiler-failure prediction performance. The original dataset has no verified historical fault labels. The Soft Sensor is selected as the primary episode-generation detector only because it detected the most events in this particular synthetic benchmark.

Full run-specific plots, CSV outputs, episode files and generated RAG reports are not fabricated or included here. Generate them by following the notebook/script workflow after obtaining the permitted source dataset and manuals.
