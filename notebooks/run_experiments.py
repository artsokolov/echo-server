"""
MLflow + LoRA NLP Experiments runner script.
Fine-tunes distilbert-base-uncased with LoRA on SST-2 sentiment classification.
Runs 4 MLflow experiments with different hyperparameter configurations.
"""
import os
import tempfile
import warnings
import numpy as np
import mlflow
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

print(f"PyTorch: {torch.__version__}, MLflow: {mlflow.__version__}")

# ── MLflow setup ────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "NLP-LoRA-SST2-Experiments"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# ── Data ────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
TRAIN_SIZE = 200
VAL_SIZE = 100

print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
train_raw = dataset["train"].select(range(TRAIN_SIZE))
val_raw = dataset["validation"].select(range(VAL_SIZE))
print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"], truncation=True, padding="max_length", max_length=128
    )


def prepare_dataset(raw_data):
    ds = raw_data.map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


tokenized_train = prepare_dataset(train_raw)
tokenized_val = prepare_dataset(val_raw)
print("Tokenization complete")


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted")),
        "precision": float(
            precision_score(labels, preds, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(labels, preds, average="weighted")),
    }


# ── Experiment runner ────────────────────────────────────────────────────────
def run_experiment(
    run_name,
    lora_rank=4,
    lora_alpha=16,
    lora_dropout=0.1,
    learning_rate=2e-4,
    batch_size=16,
    num_epochs=1,
    notes="",
):
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"LoRA: r={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"Training: lr={learning_rate}, bs={batch_size}, epochs={num_epochs}")
    print(f"Notes: {notes}")
    print("=" * 60)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    training_args = TrainingArguments(
        output_dir=f"./tmp_results/{run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "target_modules": "q_lin,v_lin",
                "dataset": "glue/sst2",
                "train_size": len(tokenized_train),
                "val_size": len(tokenized_val),
                "max_seq_length": 128,
                "trainable_params": trainable,
            }
        )
        mlflow.set_tag("notes", notes)
        mlflow.set_tag("task", "sentiment_classification")
        mlflow.set_tag("framework", "PEFT+LoRA")

        trainer.train()
        eval_results = trainer.evaluate()

        mlflow.log_metrics(
            {
                "accuracy": eval_results["eval_accuracy"],
                "f1": eval_results["eval_f1"],
                "precision": eval_results["eval_precision"],
                "recall": eval_results["eval_recall"],
                "eval_loss": eval_results["eval_loss"],
            }
        )

        # Save LoRA adapter weights as MLflow artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = os.path.join(tmpdir, "lora_adapter")
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            mlflow.log_artifacts(adapter_dir, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        print(
            f"Accuracy: {eval_results['eval_accuracy']:.4f}, "
            f"F1: {eval_results['eval_f1']:.4f}, "
            f"Loss: {eval_results['eval_loss']:.4f}"
        )

    return eval_results


# ── Run experiments ──────────────────────────────────────────────────────────
r1 = run_experiment(
    "exp1-baseline",
    lora_rank=4,
    lora_alpha=16,
    lora_dropout=0.1,
    learning_rate=2e-4,
    batch_size=16,
    num_epochs=1,
    notes="Baseline: small rank=4, alpha=16, standard LR 2e-4",
)

r2 = run_experiment(
    "exp2-higher-rank",
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    learning_rate=2e-4,
    batch_size=16,
    num_epochs=1,
    notes="Higher LoRA rank=8 and alpha=32 — more model capacity",
)

r3 = run_experiment(
    "exp3-higher-lr",
    lora_rank=4,
    lora_alpha=16,
    lora_dropout=0.1,
    learning_rate=5e-4,
    batch_size=16,
    num_epochs=1,
    notes="Higher LR=5e-4 to test faster convergence with baseline LoRA",
)

r4 = run_experiment(
    "exp4-best-config",
    lora_rank=16,
    lora_alpha=64,
    lora_dropout=0.05,
    learning_rate=3e-4,
    batch_size=32,
    num_epochs=2,
    notes="Best config attempt: rank=16, alpha=64, 2 epochs, low dropout, larger batch",
)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
results = {
    "exp1-baseline": r1,
    "exp2-higher-rank": r2,
    "exp3-higher-lr": r3,
    "exp4-best-config": r4,
}
for name, res in results.items():
    print(
        f"{name:25s} | acc={res['eval_accuracy']:.4f} | "
        f"f1={res['eval_f1']:.4f} | loss={res['eval_loss']:.4f}"
    )
best = max(results, key=lambda k: results[k]["eval_accuracy"])
print(f"\nBest run: {best}")
print("\nRun 'mlflow ui' to open MLflow UI at http://127.0.0.1:5000")
