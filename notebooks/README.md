# Notebooks

Before using the Phase 2 notebook, follow the reproducible final-training
workflow in
[`docs/classifier_training/README.md`](../docs/classifier_training/README.md).
The guide is authoritative for the frozen dataset, output paths, metrics, and
artifact transfer.

Use these notebooks for Colab-oriented exploratory work:

- Phase 1: dataset inspection, normalization, and labeling checks.
- Phase 2: classifier fine-tuning or distillation with Hugging Face models.

Keep production interfaces in `src/aragbiz/`; notebooks should call package code instead of duplicating it.
