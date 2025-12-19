# This file downloads arxiv and HF datasets

import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TREND_DATA_DIR", REPO_ROOT / "data"))

def download_latest_arxiv_data():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    return path

def update_arxiv_file(path):
    download_path = download_latest_arxiv_data()
    download_path += "/arxiv-metadata-oai-snapshot.json"

    arxiv_data = []  # create empty list
    for line in open(download_path, 'r'):  # open file line by line
        arxiv_data.append(json.loads(line))  # append data to the initially created list

    df = pd.DataFrame.from_records(arxiv_data)  # convert data to dataframe
    df = df.set_index("id")  # set unique id as index

    # Drop Irrelevant features
    df = df.drop(['submitter', 'authors', 'comments', 'journal-ref', 'doi', 'report-no', 'license', 'update_date'], axis=1)

    # Filter by category
    category_list = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.MA', 'cs.NE', 'cs.RO', 'stat.ML']
    df = df[df['categories'].isin(category_list)]

    # Rename the columns to be engineered
    df.rename(columns={'versions': 'submission_date'}, inplace=True)


    # Convert dictionary items in submission_date to a list and access the second element
    df['submission_date'] = df['submission_date'].apply(
        lambda x: x[0]['created'] if isinstance(x, list) and len(x) > 0 and 'created' in x[0] else None)

    df["submission_date"] = pd.to_datetime(df["submission_date"])
    df.to_csv(path)

def update_HF_dataset(path):
    from datasets import load_dataset
    ds = load_dataset("librarian-bots/model_cards_with_metadata")
    ds.save_to_disk(path)


if __name__ == "__main__":
    arxiv_path = DATA_ROOT / "arxiv_ai_taxonomy_papers.csv"
    hf_path = DATA_ROOT / "model_cards_with_metadata"
    update_arxiv_file(path=arxiv_path)
    update_HF_dataset(path=hf_path)
