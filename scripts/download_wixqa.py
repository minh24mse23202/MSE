from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Iterable, List

from aragbiz.preprocessing import (
    wixqa_kb_rows_to_documents,
    wixqa_rows_to_qac_records,
    write_documents_jsonl,
    write_qac_jsonl,
)

BASE_URL = "https://huggingface.co/datasets/Wix/WixQA/resolve/{revision}/{path}"
QA_PATHS = {
    "wixqa_expertwritten": "wixqa_expertwritten/test.jsonl",
    "wixqa_simulated": "wixqa_simulated/test.jsonl",
    "wixqa_synthetic": "wixqa_synthetic/test.jsonl",
}
KB_PATH = "wix_kb_corpus/wix_kb_corpus.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WixQA from Hugging Face and convert it to local ARagBiz files.")
    parser.add_argument("--subset", choices=[*sorted(QA_PATHS), "all"], default="wixqa_expertwritten")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--raw-dir", default="data/raw/wixqa")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    kb_raw_path = raw_dir / "wix_kb_corpus.jsonl"
    _download(resolve_url(KB_PATH, args.revision), kb_raw_path)
    kb_rows = read_jsonl(kb_raw_path)
    kb_lookup = {str(row["id"]): row for row in kb_rows}
    documents = wixqa_kb_rows_to_documents(kb_rows)
    docs_path = processed_dir / "wix_kb_corpus_documents.jsonl"
    write_documents_jsonl(documents, docs_path)
    print(f"Downloaded {len(kb_rows)} KB articles to {kb_raw_path}")
    print(f"Wrote {len(documents)} retrieval documents to {docs_path}")

    subsets = list(QA_PATHS) if args.subset == "all" else [args.subset]
    for subset in subsets:
        qa_raw_path = raw_dir / f"{subset}.jsonl"
        _download(resolve_url(QA_PATHS[subset], args.revision), qa_raw_path)
        qa_rows = read_jsonl(qa_raw_path)
        qac_records = wixqa_rows_to_qac_records(qa_rows, kb_lookup, subset)
        qac_path = processed_dir / f"{subset}_qac.jsonl"
        write_qac_jsonl(qac_records, qac_path)
        print(f"Downloaded {len(qa_rows)} {subset} QA rows to {qa_raw_path}")
        print(f"Wrote {len(qac_records)} QAC records to {qac_path}")


def resolve_url(path: str, revision: str) -> str:
    return BASE_URL.format(revision=revision, path=path)


def _download(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Using existing {output_path}")
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        output_path.write_bytes(response.read())


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
