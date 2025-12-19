# ACM Trend Explorer UI

This directory contains a standalone Streamlit interface for exploring the
custom ACM CCS hierarchy against the long-range trend datasets under
`output_labels` (`arx_long_trends.csv.bz2`, `dlw_long_trends.csv.bz2`,
and `hf_trends.csv.bz2`).

## Quick start

```bash
python3 -m venv .venv-ui
source .venv-ui/bin/activate
pip install --upgrade pip
pip install -r ui/trend_explorer/requirements.txt
PYTHONPATH=$(pwd) streamlit run ui/trend_explorer/app.py
```

The `PYTHONPATH` flag lets the UI import the existing `llmdap` package and
re-use the hierarchy and dataset without modifying the core project tree.

Once the app is running you can pick any taxonomy branch and use the smoothing
slider (4-100 weeks) to compare how activity evolved over time across the
arXiv long, DL Weekly long, and Hugging Face datasets.
