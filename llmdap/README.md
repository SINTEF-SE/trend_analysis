# llmdap

Core pipeline for labeling documents with a taxonomy/ontology using LLM-based
form filling.

## Setup


setup environment (from repo root):
```
conda create --name llmdap python==3.10.14
conda activate llmdap
pip install -r llmdap/requirements.txt
pip install --upgrade werkzeug==2.3.8
```
(werkzeug must be downgraded from the required version, causing a warning)

If using the OpenAI API, add your key to `llmdap/openai_key.py`

## Run (example)

```bash
python llmdap/run_inference.py
```

## Key paths

- `llmdap/main.py`: CLI entry point for batch runs.
- `llmdap/run_inference.py`: programmatic API for single or small-batch runs.
- `llmdap/arguments.yaml`: CLI argument defaults and help text.
- `llmdap/metadata_schemas/`: pydantic schemas and taxonomy traversers.
- `llmdap/context_shortening/`: this file is used for context selection (retrieval) in llmdap, however, in this experiment we only use the entire context.
- `llmdap/form_filling/`: prompt templates and form-filling logic.
- `llmdap/dataset_loader.py`: dataset loading/parsing helpers.
