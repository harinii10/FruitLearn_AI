import os
import sys
import time

from src.train_baseline import train_resnet18
from src.train_mae import train_mae
from src.train_dinov2 import train_dinov2
from src.train_ibot import train_ibot
from src.finetune import finetune_ssl_model
from src.evaluate import run_evaluation
from src.generate_pdf_reports import generate_all_pdf_reports

def main():
    print("=========================================================")
    print(" STARTING FULL GPU TRAINING & BENCHMARKING PIPELINE")
    print(" Dataset: FruitsGB (12,000 Real Images, 12 Classes)")
    print("=========================================================")

    start_total = time.time()

    print("\n--- STEP 1: Training Supervised Baseline (ResNet-18) ---")
    train_resnet18()

    print("\n--- STEP 2: Stage 1 SSL Pretraining (MAE) ---")
    train_mae()

    print("\n--- STEP 3: Stage 1 SSL Pretraining (DINOv2) ---")
    train_dinov2()

    print("\n--- STEP 4: Stage 1 SSL Pretraining (iBOT) ---")
    train_ibot()

    print("\n--- STEP 5: Stage 2 Downstream Fine-Tuning (MAE) ---")
    finetune_ssl_model("mae")

    print("\n--- STEP 6: Stage 2 Downstream Fine-Tuning (DINOv2) ---")
    finetune_ssl_model("dinov2")

    print("\n--- STEP 7: Stage 2 Downstream Fine-Tuning (iBOT) ---")
    finetune_ssl_model("ibot")

    print("\n--- STEP 8: Evaluating Models & Generating 22 Figures ---")
    run_evaluation()

    print("\n--- STEP 9: Generating All Final PDF Reports ---")
    generate_all_pdf_reports()

    elapsed = time.time() - start_total
    print("\n=========================================================")
    print(f" PIPELINE COMPLETE! Total Execution Time: {elapsed:.2f}s")
    print(" All PDF Reports saved in 'reports/'")
    print(" All 22 Figures saved in 'results/figures/'")
    print("=========================================================")

if __name__ == "__main__":
    main()
