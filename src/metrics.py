import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def compute_classification_metrics(y_true, y_pred, class_names=None):
    """Computes comprehensive evaluation metrics for fruit quality classification."""
    labels_idx = list(range(len(class_names))) if class_names else None

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_idx)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_idx)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_idx)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_idx)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_idx)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_idx)

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    
    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels_idx)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels_idx)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels_idx)

    per_class_df = pd.DataFrame({
        "Class": class_names if class_names else [f"Class_{i}" for i in range(len(prec_per_class))],
        "Precision": prec_per_class,
        "Recall": rec_per_class,
        "F1-Score": f1_per_class
    })

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_df.to_dict(orient="records")
    }

def format_table_1_dataset_distribution(split_info):
    """Generates Table 1: Dataset Distribution."""
    classes = split_info["classes"]
    records = []
    for idx, cls_name in enumerate(classes):
        train_cnt = sum(1 for item in split_info["train"] if item["label"] == idx)
        val_cnt = sum(1 for item in split_info["val"] if item["label"] == idx)
        test_cnt = sum(1 for item in split_info["test"] if item["label"] == idx)
        total = train_cnt + val_cnt + test_cnt
        records.append({
            "Class Name": cls_name,
            "Train Count": train_cnt,
            "Val Count": val_cnt,
            "Test Count": test_cnt,
            "Total": total
        })
    df = pd.DataFrame(records)
    return df

def format_table_5_final_results(results_dict):
    """Generates Table 5: Final Test Performance Comparison."""
    records = []
    for model_name, metrics in results_dict.items():
        if metrics is None or "accuracy" not in metrics:
            records.append({
                "Model": model_name,
                "Accuracy": "Not evaluated yet.",
                "Precision": "Not evaluated yet.",
                "Recall": "Not evaluated yet.",
                "F1": "Not evaluated yet."
            })
        else:
            records.append({
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision_macro']:.4f}",
                "Recall": f"{metrics['recall_macro']:.4f}",
                "F1": f"{metrics['f1_macro']:.4f}"
            })
    return pd.DataFrame(records)
