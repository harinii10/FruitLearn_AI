import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import load_config, set_seed, get_device, save_json, save_checkpoint
from src.preprocessing import get_transforms
from src.dataset import FruitsGBDataset, create_synthetic_fruitsgb_dataset, create_stratified_split
from src.train_mae import MaskedAutoencoderViT
from src.train_dinov2 import DINOv2ViT
from src.train_ibot import iBOTViT

class SSLClassifier(nn.Module):
    """Downstream classification wrapper for pretrained SSL models."""
    def __init__(self, ssl_type="mae", num_classes=12, checkpoint_path=None):
        super().__init__()
        self.ssl_type = ssl_type.lower()
        
        if self.ssl_type == "mae":
            config = load_config()
            self.encoder = MaskedAutoencoderViT(
                img_size=256,
                patch_size=config["models"]["mae"]["patch_size"],
                embed_dim=config["models"]["mae"]["embed_dim"],
                decoder_embed_dim=config["models"]["mae"]["decoder_embed_dim"],
                depth=config["models"]["mae"]["depth"],
                decoder_depth=config["models"]["mae"]["decoder_depth"],
                num_heads=config["models"]["mae"]["num_heads"],
                mask_ratio=config["models"]["mae"]["mask_ratio"]
            )
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                self.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.head = nn.Linear(config["models"]["mae"]["embed_dim"], num_classes)
        elif self.ssl_type == "dinov2":
            self.encoder = DINOv2ViT(out_dim=256, pretrained=True)
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                self.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.head = nn.Linear(512, num_classes)
        elif self.ssl_type == "ibot":
            self.encoder = iBOTViT(out_dim=256, pretrained=True)
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                self.encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.head = nn.Linear(512, num_classes)
        else:
            raise ValueError(f"Unknown SSL type: {ssl_type}")

    def forward(self, x):
        if self.ssl_type == "mae":
            # Extract latent representation from encoder
            latent, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)  # No masking during downstream evaluation
            feat = latent[:, 0, :]  # CLS token representation
        elif self.ssl_type in ["dinov2", "ibot"]:
            _, feat = self.encoder.backbone(x), self.encoder.backbone(x)
        else:
            feat = x
        out = self.head(feat)
        return out

def finetune_ssl_model(model_name="mae", label_ratio=1.0, quick_mode=False, epochs=None, batch_size=None, lr=None):
    config = load_config()
    set_seed(config["data"]["seed"])
    device = get_device()

    create_synthetic_fruitsgb_dataset(config["data"]["raw_dir"])
    split_info = create_stratified_split(config["data"]["raw_dir"], config["data"]["splits_dir"])

    train_samples = split_info["train"]
    val_samples = split_info["val"]

    # Subsample training data for limited label experiment
    if label_ratio < 1.0:
        sub_size = max(int(len(train_samples) * label_ratio), 12)
        train_samples = train_samples[:sub_size]
        print(f"[Limited Label] Using {label_ratio*100:.0f}% labeled training data ({len(train_samples)} samples).")

    if quick_mode:
        epochs = epochs or config["finetune"]["quick_epochs"]
        train_samples = train_samples[:60]
        val_samples = val_samples[:30]
    else:
        epochs = epochs or config["finetune"]["epochs"]

    batch_size = batch_size or config["finetune"]["batch_size"]
    lr = lr or config["finetune"]["learning_rate"]

    train_dataset = FruitsGBDataset(train_samples, transform=get_transforms(256, is_training=True), ssl_mode=False)
    val_dataset = FruitsGBDataset(val_samples, transform=get_transforms(256, is_training=False), ssl_mode=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pretrained_ckpt = f"models/{model_name.lower()}/{model_name.lower()}_pretrained.pt"
    model = SSLClassifier(ssl_type=model_name, num_classes=config["data"]["num_classes"], checkpoint_path=pretrained_ckpt).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / max(total, 1)
        epoch_train_acc = correct / max(total, 1)

        # Validation Loop
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Fine-Tuning {model_name.upper()} ({label_ratio*100:.0f}% labels) Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            out_ckpt = f"models/{model_name.lower()}/{model_name.lower()}_finetuned.pt"
            if label_ratio < 1.0:
                out_ckpt = f"models/{model_name.lower()}/{model_name.lower()}_finetuned_{int(label_ratio*100)}pct.pt"
            save_checkpoint(model, optimizer, epoch, {"val_acc": epoch_val_acc}, out_ckpt)

    elapsed = time.time() - start_time
    log_data = {
        "model": model_name.upper(),
        "label_ratio": label_ratio,
        "epochs": epochs,
        "training_time_sec": elapsed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc
    }
    log_file = f"results/logs/{model_name.lower()}_finetune_log.json"
    if label_ratio < 1.0:
        log_file = f"results/logs/{model_name.lower()}_finetune_{int(label_ratio*100)}pct_log.json"
    save_json(log_data, log_file)
    return log_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mae", choices=["mae", "dinov2", "ibot"])
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--quick-mode", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    finetune_ssl_model(model_name=args.model, label_ratio=args.ratio, quick_mode=args.quick_mode, epochs=args.epochs)
