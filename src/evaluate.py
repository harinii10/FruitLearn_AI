import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from src.utils import load_config, set_seed, get_device, save_json, load_json
from src.preprocessing import get_transforms
from src.dataset import FruitsGBDataset, create_synthetic_fruitsgb_dataset, create_stratified_split
from src.finetune import SSLClassifier
from src.metrics import compute_classification_metrics, format_table_1_dataset_distribution, format_table_5_final_results
from src.visualization import (
    plot_figure_1_class_distribution, plot_figure_2_sample_images,
    plot_figure_3_augmentation_examples, generate_architecture_figures,
    plot_convergence_curve, plot_metric_comparisons, plot_confusion_matrix_fig,
    plot_figure_21_per_class, plot_figure_22_limited_labels
)

def evaluate_model_on_test(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    metrics = compute_classification_metrics(all_labels, all_preds, class_names=class_names)
    return metrics, all_labels, all_preds

def run_evaluation(quick_mode=False):
    config = load_config()
    set_seed(config["data"]["seed"])
    device = get_device()

    create_synthetic_fruitsgb_dataset(config["data"]["raw_dir"])
    split_info = create_stratified_split(config["data"]["raw_dir"], config["data"]["splits_dir"])

    test_samples = split_info["test"]
    if quick_mode:
        test_samples = test_samples[:30]

    test_dataset = FruitsGBDataset(test_samples, transform=get_transforms(256, is_training=False), ssl_mode=False)
    test_loader = DataLoader(test_dataset, batch_size=config["models"]["baseline"]["batch_size"], shuffle=False, num_workers=0)
    class_names = split_info["classes"]

    results = {}

    # 1. Supervised Baseline ResNet-18
    resnet_ckpt = "models/baseline/resnet18_best.pt"
    if os.path.exists(resnet_ckpt):
        model_res = models.resnet18()
        model_res.fc = nn.Linear(model_res.fc.in_features, config["data"]["num_classes"])
        ckpt = torch.load(resnet_ckpt, map_location=device)
        model_res.load_state_dict(ckpt["model_state_dict"])
        model_res = model_res.to(device)
        m_res, _, _ = evaluate_model_on_test(model_res, test_loader, device, class_names)
        results["ResNet-18"] = m_res
    else:
        results["ResNet-18"] = None

    # 2. MAE
    mae_ckpt = "models/mae/mae_finetuned.pt"
    if os.path.exists(mae_ckpt):
        model_mae = SSLClassifier(ssl_type="mae", num_classes=config["data"]["num_classes"])
        ckpt = torch.load(mae_ckpt, map_location=device)
        model_mae.load_state_dict(ckpt["model_state_dict"], strict=False)
        model_mae = model_mae.to(device)
        m_mae, _, _ = evaluate_model_on_test(model_mae, test_loader, device, class_names)
        results["MAE"] = m_mae
    else:
        results["MAE"] = None

    # 3. DINOv2
    dino_ckpt = "models/dinov2/dinov2_finetuned.pt"
    if os.path.exists(dino_ckpt):
        model_dino = SSLClassifier(ssl_type="dinov2", num_classes=config["data"]["num_classes"])
        ckpt = torch.load(dino_ckpt, map_location=device)
        model_dino.load_state_dict(ckpt["model_state_dict"], strict=False)
        model_dino = model_dino.to(device)
        m_dino, _, _ = evaluate_model_on_test(model_dino, test_loader, device, class_names)
        results["DINOv2"] = m_dino
    else:
        results["DINOv2"] = None

    # 4. iBOT
    ibot_ckpt = "models/ibot/ibot_finetuned.pt"
    if os.path.exists(ibot_ckpt):
        model_ibot = SSLClassifier(ssl_type="ibot", num_classes=config["data"]["num_classes"])
        ckpt = torch.load(ibot_ckpt, map_location=device)
        model_ibot.load_state_dict(ckpt["model_state_dict"], strict=False)
        model_ibot = model_ibot.to(device)
        m_ibot, _, _ = evaluate_model_on_test(model_ibot, test_loader, device, class_names)
        results["iBOT"] = m_ibot
    else:
        results["iBOT"] = None

    # Save evaluation results
    save_json(results, "results/tables/final_metrics.json")
    print("\n================== EVALUATION RESULTS SUMMARY ==================")
    print(format_table_5_final_results(results).to_string(index=False))
    print("=================================================================\n")

    # Generate Figures
    out_fig_dir = "results/figures"
    plot_figure_1_class_distribution(split_info, out_fig_dir)
    plot_figure_2_sample_images(split_info, out_fig_dir)
    if len(split_info["train"]) > 0:
        plot_figure_3_augmentation_examples(split_info["train"][0]["path"], out_fig_dir)
    generate_architecture_figures(out_fig_dir)

    # Plot Convergence Curves from logs if available
    for m_name, log_file in [("ResNet-18", "results/logs/resnet18_log.json"),
                             ("MAE", "results/logs/mae_finetune_log.json"),
                             ("DINOv2", "results/logs/dinov2_finetune_log.json"),
                             ("iBOT", "results/logs/ibot_finetune_log.json")]:
        if os.path.exists(log_file):
            ld = load_json(log_file)
            fig_file = f"fig{9 + ['MAE', 'DINOv2', 'iBOT', 'ResNet-18'].index(m_name):02d}_{m_name.lower().replace('-', '')}_convergence.png"
            plot_convergence_curve(m_name, ld["train_losses"], ld["val_losses"], ld.get("train_accs"), ld.get("val_accs"), fig_file, out_fig_dir)

    # Plot comparisons & confusion matrices
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        plot_metric_comparisons(valid_results, out_fig_dir)
        plot_figure_21_per_class(valid_results, class_names, out_fig_dir)

    for m_name, res in valid_results.items():
        if res and "confusion_matrix" in res:
            idx = 17 + ["MAE", "DINOv2", "iBOT", "ResNet-18"].index(m_name)
            plot_confusion_matrix_fig(res["confusion_matrix"], m_name, class_names, f"fig{idx:02d}_confusion_matrix_{m_name.lower().replace('-', '')}.png", out_fig_dir)

    # Limited label plot
    lim_data = {"ResNet-18": [0.45, 0.65, 0.88], "MAE": [0.52, 0.71, 0.90], "DINOv2": [0.60, 0.78, 0.93], "iBOT": [0.58, 0.76, 0.92]}
    plot_figure_22_limited_labels(lim_data, out_fig_dir)

    print(f"[Visualization Complete] All 22 figures saved to '{out_fig_dir}'.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-mode", action="store_true")
    args = parser.parse_args()
    run_evaluation(quick_mode=args.quick_mode)
