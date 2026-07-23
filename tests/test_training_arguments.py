from scripts.train_hf_query_classifier import make_training_args as make_hf_training_args
from scripts.train_t5_query_classifier import make_training_args as make_t5_training_args


class FakeTrainingArguments:
    def __init__(
        self,
        output_dir,
        save_strategy,
        save_total_limit,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        weight_decay,
        report_to,
        seed,
        load_best_model_at_end=None,
        metric_for_best_model=None,
        eval_strategy=None,
        predict_with_generate=None,
    ):
        self.save_total_limit = save_total_limit


def test_distilbert_training_retains_one_checkpoint():
    arguments = make_hf_training_args(
        FakeTrainingArguments,
        output_dir="artifact",
        learning_rate=2e-5,
        batch_size=8,
        epochs=3,
        seed=42,
    )

    assert arguments.save_total_limit == 1


def test_t5_training_retains_one_checkpoint():
    arguments = make_t5_training_args(
        FakeTrainingArguments,
        output_dir="artifact",
        learning_rate=3e-4,
        batch_size=4,
        epochs=3,
        seed=42,
    )

    assert arguments.save_total_limit == 1
