from __future__ import annotations

import argparse

from aragbiz.preprocessing import read_csv_dataset, write_qac_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a CSV QA dataset into QAC JSONL.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", default="data/processed/qac_dataset.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    records = read_csv_dataset(args.input)
    write_qac_jsonl(records, args.output)
    print(f"Wrote {len(records)} QAC records to {args.output}")


if __name__ == "__main__":
    main()

