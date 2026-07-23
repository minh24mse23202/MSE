from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from aragbiz.classifier import T5QueryClassifier, evaluate_classifier
from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.factory import existing_dataset_path
from aragbiz.schemas import COMPLEXITY_LABELS, QACRecord
from aragbiz.training import grouped_stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune T5-small as a query complexity classifier.")
    parser.add_argument("--dataset", default=None, help="Primary QAC JSONL path.")
    parser.add_argument("--extra-dataset", action="append", default=[], help="Additional QAC JSONL path.")
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--output-dir", default="data/artifacts/query_classifier_t5_v2")
    parser.add_argument("--metrics-output", default="docs/evaluation/t5_query_classifier_v2_metrics.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--target-max-length", type=int, default=8)
    args = parser.parse_args()

    transformers, datasets = import_ml_dependencies()
    AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM
    AutoTokenizer = transformers.AutoTokenizer
    DataCollatorForSeq2Seq = transformers.DataCollatorForSeq2Seq
    Trainer = getattr(transformers, "Seq2SeqTrainer", transformers.Trainer)
    TrainingArguments = getattr(transformers, "Seq2SeqTrainingArguments", transformers.TrainingArguments)
    Dataset = datasets.Dataset

    config = load_config()
    dataset_path = args.dataset or existing_dataset_path(config)
    records = load_qac_jsonl(dataset_path)
    for extra_dataset in args.extra_dataset:
        records.extend(load_qac_jsonl(extra_dataset))
    train_records, validation_records, split_manifest = grouped_stratified_split(
        records, args.validation_ratio, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    train_dataset = make_dataset(Dataset, train_records, tokenizer, args.max_length, args.target_max_length)
    validation_dataset = make_dataset(Dataset, validation_records, tokenizer, args.max_length, args.target_max_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = make_training_args(
        TrainingArguments,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    trainer = make_trainer(
        Trainer,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(
        {"artifact_version": 2, "architecture": "t5", "complexity_labels": list(COMPLEXITY_LABELS)},
        str(Path(args.output_dir) / "classifier_manifest.json"),
    )

    classifier = T5QueryClassifier(args.output_dir, max_length=args.max_length, generation_max_length=args.target_max_length)
    metrics_payload = {
        "dataset": dataset_path,
        "extra_datasets": args.extra_dataset,
        "model_name": args.model_name,
        "model_path": args.output_dir,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "complexity_labels": list(COMPLEXITY_LABELS),
        "artifact_version": 2,
        "split": split_manifest,
        "metrics": evaluate_classifier(validation_records, classifier),
    }
    write_json(metrics_payload, args.metrics_output)
    print(json.dumps(metrics_payload, indent=2, sort_keys=True))


def import_ml_dependencies():
    try:
        import datasets
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "Missing optional ML dependencies. Install them with: python -m pip install -e \".[ml]\""
        ) from exc
    return transformers, datasets


def make_dataset(Dataset, records: List[QACRecord], tokenizer, max_length: int, target_max_length: int):
    dataset = Dataset.from_dict(
        {
            "text": [format_t5_input(record.question) for record in records],
            "target": [record.complexity_label for record in records],
        }
    )

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        labels = tokenize_targets(tokenizer, batch["target"], target_max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize, batched=True).remove_columns(["text", "target"])


def tokenize_targets(tokenizer, targets: List[str], target_max_length: int):
    try:
        return tokenizer(
            text_target=targets,
            truncation=True,
            padding="max_length",
            max_length=target_max_length,
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            return tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=target_max_length,
            )


def make_training_args(TrainingArguments, output_dir: str, learning_rate: float, batch_size: int, epochs: float, seed: int):
    kwargs = {
        "output_dir": output_dir,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "report_to": [],
        "seed": seed,
    }
    parameters = TrainingArguments.__init__.__code__.co_varnames
    if "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    if "predict_with_generate" in parameters:
        kwargs["predict_with_generate"] = True
    return TrainingArguments(**kwargs)


def make_trainer(Trainer, model, training_args, train_dataset, validation_dataset, tokenizer, data_collator):
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": data_collator,
    }
    parameters = Trainer.__init__.__code__.co_varnames
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def write_json(payload: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_t5_input(query: str) -> str:
    return f"classify query complexity: {query}"


if __name__ == "__main__":
    main()
