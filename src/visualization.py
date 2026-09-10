import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image

def ensure_figures_dir(output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Figure 1: Dataset Class Distribution
def plot_figure_1_class_distribution(split_info, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, "fig01_class_distribution.png")
    classes = split_info["classes"]
    counts = []
    for idx, cls in enumerate(classes):
        cnt = sum(1 for item in split_info["train"] + split_info["val"] + split_info["test"] if item["label"] == idx)
        counts.append(cnt)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, counts, color='#1A365D')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel("Number of Images")
    plt.title("Figure 1: FruitsGB Dataset Class Distribution", fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

# Figure 2: Sample Fruit Images
def plot_figure_2_sample_images(split_info, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, "fig02_sample_fruit_images.png")
    classes = split_info["classes"]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    axes = axes.flatten()

    for idx, cls_name in enumerate(classes):
        matching = [item["path"] for item in split_info["train"] if item["label"] == idx]
        if matching:
            img = Image.open(matching[0])
            axes[idx].imshow(img)
            axes[idx].set_title(cls_name, fontsize=9, fontweight='bold')
            axes[idx].axis("off")
            
    plt.suptitle("Figure 2: Sample FruitsGB Images (12 Classes)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

# Figure 3: Augmentation Examples
def plot_figure_3_augmentation_examples(sample_img_path, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, "fig03_augmentation_examples.png")
    from src.preprocessing import get_transforms
    orig_img = Image.open(sample_img_path).convert("RGB")
    transform = get_transforms(image_size=256, is_training=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=9)
    axes[0].axis("off")

    for i in range(1, 4):
        aug_t = transform(orig_img)
        aug_np = aug_t.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[i].imshow(aug_np)
        axes[i].set_title(f"Augmented View {i}", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Figure 3: Training Image Augmentation Examples", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

# Figures 4-8: System Architecture Diagrams (Proportional & Aspect-Ratio Preserved)
def plot_architecture_schematic(title, box_labels, filename, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    
    n = len(box_labels)
    y_positions = np.linspace(0.88, 0.12, n)
    
    for i, label in enumerate(box_labels):
        plt.text(0.5, y_positions[i], label, ha='center', va='center', fontsize=9.5, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#EBF8FF', edgecolor='#2B6CB0', lw=1.5))
        if i < n - 1:
            plt.annotate('', xy=(0.5, y_positions[i+1]+0.04), xytext=(0.5, y_positions[i]-0.04),
                         arrowprops=dict(arrowstyle="->", lw=2, color='#1A365D'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def generate_architecture_figures(output_dir="results/figures"):
    plot_architecture_schematic(
        "Figure 4: Overall Proposed System Architecture",
        ["FruitsGB Dataset (12,000 images)", "Stratified Split (70/15/15)", "Stage 1: SSL Feature Learning (MAE, DINOv2, iBOT)",
         "Stage 2: Downstream Fine-Tuning (Good/Bad Quality)", "Evaluation & Baseline Comparison (ResNet-18)"],
        "fig04_overall_system_architecture.png", output_dir
    )
    plot_architecture_schematic(
        "Figure 5: Masked Autoencoder (MAE) Architecture",
        ["Input Fruit Image (256x256)", "Patch Embedding (16x16 Patches)", "75% Random Patch Masking",
         "ViT Encoder (Visible Patches)", "Latent Representations", "Lightweight Decoder", "MSE Reconstruction Loss"],
        "fig05_mae_architecture.png", output_dir
    )
    plot_architecture_schematic(
        "Figure 6: DINOv2 Architecture (Self-Distillation)",
        ["Input Fruit Image", "Multi-Crop Augmentation (View 1 & View 2)", "Student ViT & Teacher ViT",
         "Centering & Sharpening Cross-Entropy Loss", "EMA Teacher Update", "Learned Feature Representation", "Downstream Linear Classifier"],
        "fig06_dinov2_architecture.png", output_dir
    )
    plot_architecture_schematic(
        "Figure 7: iBOT Architecture (Masked Image Modeling + Distillation)",
        ["Input Fruit Image", "Patch Masking & Tokenization", "Student Transformer & Teacher Transformer",
         "Masked Patch Token Loss + Global Image Token Loss", "Learned Visual Representation", "Fruit Quality Classification Head"],
        "fig07_ibot_architecture.png", output_dir
    )
    plot_architecture_schematic(
        "Figure 8: Supervised Baseline (ResNet-18) Architecture",
        ["Input Fruit Image (256x256)", "7x7 Conv + MaxPool", "4 Residual Blocks (BasicBlocks with Skip Connections)",
         "Global Average Pooling", "Fully Connected Classifier (12 Classes)", "Cross-Entropy Loss (Direct Supervised)"],
        "fig08_resnet18_architecture.png", output_dir
    )

# Figures 9-12: Convergence Curves
def plot_convergence_curve(model_name, train_losses, val_losses, train_accs=None, val_accs=None, filename="conv.png", output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if train_accs and val_accs:
        ax2 = ax1.twinx()
        ax2.plot(epochs, train_accs, 'g--^', label='Train Acc')
        ax2.plot(epochs, val_accs, 'm--v', label='Val Acc')
        ax2.set_ylabel('Accuracy', color='m')
        ax2.tick_params(axis='y', labelcolor='m')
        
    plt.title(f"Convergence Analysis - {model_name}", fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

# Figures 13-16: Model Metric Comparisons
def plot_metric_comparisons(results_dict, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    models = list(results_dict.keys())
    metrics_list = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    titles = [
        ("Figure 13: Accuracy Comparison across Models", "fig13_accuracy_comparison.png", "Accuracy"),
        ("Figure 14: Precision Comparison across Models", "fig14_precision_comparison.png", "Precision"),
        ("Figure 15: Recall Comparison across Models", "fig15_recall_comparison.png", "Recall"),
        ("Figure 16: F1-Score Comparison across Models", "fig16_f1_score_comparison.png", "F1-Score")
    ]
    
    colors = ['#1A365D', '#2B6CB0', '#319795', '#E53E3E']

    for metric_key, (title, fname, label) in zip(metrics_list, titles):
        vals = [results_dict[m].get(metric_key, 0.0) if results_dict[m] else 0.0 for m in models]
        out_path = os.path.join(output_dir, fname)
        
        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(models, vals, color=colors[:len(models)])
        plt.title(title, fontsize=11, fontweight='bold')
        plt.ylabel(label)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.4f}' if h > 0 else 'N/A', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

# Figures 17-20: Confusion Matrices
def plot_confusion_matrix_fig(cm, model_name, class_names, filename, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

# Figure 21: Per-Class Performance
def plot_figure_21_per_class(results_dict, class_names, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, "fig21_per_class_performance.png")
    df_list = []
    for model_name, res in results_dict.items():
        if res and "per_class" in res:
            for item in res["per_class"]:
                df_list.append({
                    "Model": model_name,
                    "Class": item["Class"],
                    "F1-Score": item["F1-Score"]
                })
    if not df_list:
        return out_path

    df = pd.DataFrame(df_list)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=df, x="Class", y="F1-Score", hue="Model")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Figure 21: Per-Class F1-Score Performance Comparison", fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

# Figure 22: Limited-Label Experiment
def plot_figure_22_limited_labels(limited_results, output_dir="results/figures"):
    output_dir = ensure_figures_dir(output_dir)
    out_path = os.path.join(output_dir, "fig22_limited_label_experiment.png")
    plt.figure(figsize=(8, 4.5))
    
    ratios = [10, 20, 100]
    for model_name, accs in limited_results.items():
        plt.plot(ratios, accs, marker='o', label=model_name, linewidth=2)
        
    plt.xlabel("Percentage of Labeled Training Data (%)")
    plt.ylabel("Downstream Test Accuracy")
    plt.title("Figure 22: Performance Under Limited Labeled Training Data", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path
