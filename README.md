
# Multi-source Multi-level Trend Analysis

LLM-based pipeline and tooling for labeling documents from multiple sources
into a shared taxonomy, then analyzing trend signals across time and/or taxonomy placement.

## Project layout

- `llmdap/`: core labeling pipeline and configs.
- `data/`: dataset download and data prep scripts.
- `metrics/`: metrics used in analysis.
- `output_labels/`: generated label CSVs.
- `results/`: notebooks and analysis outputs.
- `taxonomies/`: taxonomy definition.
- `ui/`: Streamlit trend explorer UI.

For the UI, see `ui/trend_explorer/README.md`.
