import json
import random
from pathlib import Path

import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "verification"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "deberta_scifact"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-small"

# Smaller sequence length for CPU training
MAX_LENGTH = 128

# CPU-friendly dataset size
TRAIN_SAMPLE_SIZE = 300
DEV_SAMPLE_SIZE = 150

# Keep one epoch for testing
NUM_EPOCHS = 1

# CPU-friendly batch size
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

# Reduced learning rate for stability
LEARNING_RATE = 1e-5

# Gradient clipping
MAX_GRAD_NORM = 1.0


# ============================================================
# LABEL CONFIGURATION
# ============================================================

LABEL_NAMES = [
    "SUPPORT",
    "CONTRADICT",
    "NEI",
]

LABEL2ID = {
    "SUPPORT": 0,
    "CONTRADICT": 1,
    "NEI": 2,
}

ID2LABEL = {
    0: "SUPPORT",
    1: "CONTRADICT",
    2: "NEI",
}


# ============================================================
# SET RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(file_path):

    data = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                data.append(
                    json.loads(line)
                )

    return data


# ============================================================
# NORMALIZE LABELS
# ============================================================

def normalize_label(example):

    """
    Converts different possible label formats into:
    0 = SUPPORT
    1 = CONTRADICT
    2 = NEI
    """

    label = example.get("label")

    # If label is already numeric
    if isinstance(label, (int, np.integer)):

        label = int(label)

    elif isinstance(label, float):

        label = int(label)

    # If label is a string
    elif isinstance(label, str):

        label_upper = label.strip().upper()

        # Numeric strings
        if label_upper.isdigit():

            label = int(label_upper)

        else:

            # Text labels
            if label_upper in LABEL2ID:

                label = LABEL2ID[label_upper]

            else:

                raise ValueError(
                    f"Unknown label found: {label}"
                )

    else:

        # Try label_name if available
        label_name = example.get("label_name")

        if label_name is not None:

            label_upper = str(
                label_name
            ).strip().upper()

            if label_upper in LABEL2ID:

                label = LABEL2ID[label_upper]

            else:

                raise ValueError(
                    f"Unknown label_name: {label_name}"
                )

        else:

            raise ValueError(
                f"Invalid label: {label}"
            )

    # Final validation
    if label not in [0, 1, 2]:

        raise ValueError(
            f"Invalid numeric label: {label}"
        )

    example["label"] = int(label)

    return example


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(dataset, dataset_name):

    print(
        f"\n🔍 Validating {dataset_name} dataset..."
    )

    labels = []

    for example in dataset:

        label = example["label"]

        if label not in [0, 1, 2]:

            raise ValueError(
                f"Invalid label in {dataset_name}: {label}"
            )

        labels.append(label)

    print(
        f"✅ {dataset_name} labels are valid"
    )

    print(
        f"📊 SUPPORT: {labels.count(0)}"
    )

    print(
        f"📊 CONTRADICT: {labels.count(1)}"
    )

    print(
        f"📊 NEI: {labels.count(2)}"
    )


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    train_path = DATA_DIR / "train.jsonl"

    dev_path = DATA_DIR / "dev.jsonl"

    print("\n📂 Loading verification datasets...")

    train_data = load_jsonl(train_path)

    dev_data = load_jsonl(dev_path)

    print(
        f"📊 Original training examples: {len(train_data)}"
    )

    print(
        f"📊 Original development examples: {len(dev_data)}"
    )


    # --------------------------------------------------------
    # NORMALIZE LABELS BEFORE TRAINING
    # --------------------------------------------------------

    print("\n🏷️ Normalizing labels...")

    train_data = [
        normalize_label(example)
        for example in train_data
    ]

    dev_data = [
        normalize_label(example)
        for example in dev_data
    ]


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.shuffle(train_data)

    random.shuffle(dev_data)


    # --------------------------------------------------------
    # CPU-FRIENDLY SUBSETS
    # --------------------------------------------------------

    train_data = train_data[
        :TRAIN_SAMPLE_SIZE
    ]

    dev_data = dev_data[
        :DEV_SAMPLE_SIZE
    ]

    print(
        "\n⚡ CPU-friendly training mode enabled"
    )

    print(
        f"✅ Training examples used: {len(train_data)}"
    )

    print(
        f"✅ Development examples used: {len(dev_data)}"
    )


    # --------------------------------------------------------
    # CREATE DATASETS
    # --------------------------------------------------------

    train_dataset = Dataset.from_list(
        train_data
    )

    dev_dataset = Dataset.from_list(
        dev_data
    )


    # --------------------------------------------------------
    # VALIDATE LABELS
    # --------------------------------------------------------

    validate_dataset(
        train_dataset,
        "Training"
    )

    validate_dataset(
        dev_dataset,
        "Development"
    )

    return train_dataset, dev_dataset


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(

        examples["claim"],

        examples["evidence"],

        truncation=True,

        max_length=MAX_LENGTH,
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            labels,

            predictions,

            average="weighted",

            zero_division=0,
        )
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,
    }


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

def save_detailed_results(
    trainer,
    dataset,
):

    print(
        "\n📊 Generating detailed evaluation metrics..."
    )

    predictions_output = trainer.predict(
        dataset
    )

    logits = predictions_output.predictions

    labels = predictions_output.label_ids

    predictions = np.argmax(
        logits,
        axis=-1,
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        labels,

        predictions,

        labels=[0, 1, 2],

        target_names=LABEL_NAMES,

        digits=4,

        zero_division=0,
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(

        labels,

        predictions,

        labels=[0, 1, 2],
    )


    print("\n" + "=" * 65)

    print(
        "📊 CLASSIFICATION REPORT"
    )

    print("=" * 65)

    print(report)


    print("\n" + "=" * 65)

    print(
        "📊 CONFUSION MATRIX"
    )

    print("=" * 65)

    print(matrix)


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report_path = (
        RESULTS_DIR
        / "deberta_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "DEBERTA CLAIM VERIFICATION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n\n"
        )

        file.write(report)

        file.write(
            "\n\nCONFUSION MATRIX\n\n"
        )

        file.write(
            str(matrix)
        )


    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    matrix_path = (
        RESULTS_DIR
        / "deberta_confusion_matrix.txt"
    )

    np.savetxt(

        matrix_path,

        matrix,

        fmt="%d",
    )


    print(
        "\n💾 Results saved successfully!"
    )

    print(
        f"📁 Report: {report_path}"
    )

    print(
        f"📁 Matrix: {matrix_path}"
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    global tokenizer


    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed(42)


    print("\n" + "=" * 70)

    print(
        "🧠 DEBERTA CLAIM VERIFICATION TRAINING"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\n🖥️ Device: {device.upper()}"
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train_dataset, dev_dataset = (
        load_datasets()
    )


    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print(
        "\n🔤 Loading DeBERTa tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )


    # --------------------------------------------------------
    # TOKENIZE TRAINING DATA
    # --------------------------------------------------------

    print(
        "\n⚙️ Tokenizing training dataset..."
    )

    train_dataset = train_dataset.map(

        tokenize_function,

        batched=True,

        desc="Tokenizing training data",
    )


    # --------------------------------------------------------
    # TOKENIZE DEVELOPMENT DATA
    # --------------------------------------------------------

    print(
        "\n⚙️ Tokenizing development dataset..."
    )

    dev_dataset = dev_dataset.map(

        tokenize_function,

        batched=True,

        desc="Tokenizing development data",
    )


    # --------------------------------------------------------
    # REMOVE UNNECESSARY COLUMNS
    # --------------------------------------------------------

    columns_to_remove = [

        "claim_id",

        "claim",

        "evidence",

        "label_name",

        "document_id",
    ]


    for column in columns_to_remove:

        if column in train_dataset.column_names:

            train_dataset = (
                train_dataset.remove_columns(
                    column
                )
            )

        if column in dev_dataset.column_names:

            dev_dataset = (
                dev_dataset.remove_columns(
                    column
                )
            )


    # --------------------------------------------------------
    # RENAME LABEL COLUMN
    # --------------------------------------------------------

    if "label" in train_dataset.column_names:

        train_dataset = (
            train_dataset.rename_column(
                "label",
                "labels",
            )
        )


    if "label" in dev_dataset.column_names:

        dev_dataset = (
            dev_dataset.rename_column(
                "label",
                "labels",
            )
        )


    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print("\n🔍 Final dataset columns:")

    print(
        f"Training: {train_dataset.column_names}"
    )

    print(
        f"Development: {dev_dataset.column_names}"
    )


    print("\n🏷️ Sample labels:")

    print(
        train_dataset["labels"][:10]
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\n🧠 Loading DeBERTa model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(

            MODEL_NAME,

            num_labels=3,

            label2id=LABEL2ID,

            id2label=ID2LABEL,
        )
    )


    # --------------------------------------------------------
    # DATA COLLATOR
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(

        tokenizer=tokenizer,

        return_tensors="pt",
    )


    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=str(MODEL_DIR),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        num_train_epochs=NUM_EPOCHS,

        weight_decay=0.01,

        max_grad_norm=MAX_GRAD_NORM,

        logging_strategy="steps",

        logging_steps=10,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=False,

        report_to="none",

        dataloader_pin_memory=False,

        fp16=False,

        bf16=False,
    )


    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=dev_dataset,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )


    # ========================================================
    # START TRAINING
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "🚀 STARTING STABLE CPU-FRIENDLY DEBERTA FINE-TUNING"
    )

    print("=" * 70)


    print(
        f"\n📊 Training samples: {len(train_dataset)}"
    )

    print(
        f"📊 Development samples: {len(dev_dataset)}"
    )

    print(
        f"📏 Maximum sequence length: {MAX_LENGTH}"
    )

    print(
        f"🔄 Epochs: {NUM_EPOCHS}"
    )

    print(
        f"📚 Learning rate: {LEARNING_RATE}"
    )

    print(
        f"✂️ Max gradient norm: {MAX_GRAD_NORM}\n"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    trainer.train()


    # ========================================================
    # SAVE MODEL
    # ========================================================

    print(
        "\n💾 Saving trained model..."
    )

    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "📊 FINAL EVALUATION"
    )

    print("=" * 70)


    results = trainer.evaluate()


    print(
        f"\n🎯 Accuracy: "
        f"{results.get('eval_accuracy', 0):.4f}"
    )

    print(
        f"🎯 Precision: "
        f"{results.get('eval_precision', 0):.4f}"
    )

    print(
        f"🎯 Recall: "
        f"{results.get('eval_recall', 0):.4f}"
    )

    print(
        f"🎯 F1 Score: "
        f"{results.get('eval_f1', 0):.4f}"
    )


    # ========================================================
    # DETAILED EVALUATION
    # ========================================================

    save_detailed_results(

        trainer,

        dev_dataset,
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "✅ DEBERTA TRAINING COMPLETED SUCCESSFULLY!"
    )

    print("=" * 70)


    print(
        f"\n📁 Model location:\n{MODEL_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()