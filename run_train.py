"""LJSpeech 训练启动脚本 - 使用新 Conv1d Decoder 架构"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model.vits import VITS
from config import (
    TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
    HiFiGANConfig, TrainConfig
)
from dataset import get_dataloader
from train import compute_loss


def train_ljspeech():
    # 设备选择
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # 模型配置
    configs = {
        "text_encoder": TextEncoderConfig(),
        "decoder": DecoderConfig(),
        "duration_predictor": DurationPredictorConfig(),
        "hifigan": HiFiGANConfig(),
        "train": TrainConfig(),
    }

    model = VITS(configs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} ({total_params/1e6:.1f}M)")

    # 数据加载（子集快速验证模式）
    data_dir = "data/processed"
    train_filelist = f"{data_dir}/filelists/train_subset.txt"
    val_filelist = f"{data_dir}/filelists/val_subset.txt"

    # 如果子集文件不存在则用全量
    if not os.path.exists(train_filelist):
        train_filelist = f"{data_dir}/filelists/train.txt"
        val_filelist = f"{data_dir}/filelists/val.txt"

    train_dl = get_dataloader(train_filelist, data_dir,
                              batch_size=16, shuffle=True, num_workers=0)
    val_dl = get_dataloader(val_filelist, data_dir,
                            batch_size=16, shuffle=False, num_workers=0)
    print(f"Train: {train_filelist} ({len(train_dl)} batches)")
    print(f"Val: {val_filelist} ({len(val_dl)} batches)")

    # 优化器 + 学习率调度
    epochs = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Checkpoint 目录
    ckpt_dir = "data/checkpoints_v2"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    log_file = open("train_v2.log", "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"total": 0, "mel": 0, "dur": 0, "kl": 0}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_dl):
            phoneme_ids = batch["phoneme_ids"].to(device)
            phoneme_lengths = batch["phoneme_lengths"].to(device)
            mel = batch["mel"].to(device)
            durations = batch["duration"].to(device)

            # 10% 的 batch 使用预测 duration
            use_pred = (batch_idx % 10 == 0)
            outputs = model(phoneme_ids, phoneme_lengths, mel,
                          durations=durations, use_predicted_duration=use_pred)

            targets = {"mel": mel, "duration": durations}
            losses = compute_loss(outputs, targets, phoneme_lengths,
                                duration_loss_weight=5.0, kl_loss_weight=0.1)

            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["total"] += losses["total_loss"].item()
            epoch_losses["mel"] += losses["mel_loss"].item()
            epoch_losses["dur"] += losses["duration_loss"].item()
            epoch_losses["kl"] += losses["kl_loss"].item()
            n_batches += 1

            if batch_idx % 50 == 0:
                log(f"  E{epoch+1} B{batch_idx}/{len(train_dl)} "
                    f"loss={losses['total_loss'].item():.4f} "
                    f"mel={losses['mel_loss'].item():.4f} "
                    f"dur={losses['duration_loss'].item():.4f}")

        scheduler.step()
        dt = time.time() - t0
        avg = {k: v / n_batches for k, v in epoch_losses.items()}

        # 验证
        model.eval()
        val_loss = 0
        val_mel = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                phoneme_ids = batch["phoneme_ids"].to(device)
                phoneme_lengths = batch["phoneme_lengths"].to(device)
                mel = batch["mel"].to(device)
                durations = batch["duration"].to(device)

                outputs = model(phoneme_ids, phoneme_lengths, mel, durations=durations)
                targets = {"mel": mel, "duration": durations}
                losses = compute_loss(outputs, targets, phoneme_lengths)
                val_loss += losses["total_loss"].item()
                val_mel += losses["mel_loss"].item()
                val_n += 1

        avg_val = val_loss / val_n
        avg_val_mel = val_mel / val_n

        log(f"Epoch {epoch+1}/{epochs} ({dt:.0f}s) | "
            f"Train: total={avg['total']:.4f} mel={avg['mel']:.4f} dur={avg['dur']:.4f} kl={avg['kl']:.4f} | "
            f"Val: total={avg_val:.4f} mel={avg_val_mel:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存 checkpoint
        if (epoch + 1) % 10 == 0:
            path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg["total"],
                "val_loss": avg_val,
            }, path)
            log(f"  Saved checkpoint: {path}")

        # 保存最佳模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": avg["total"],
                "val_loss": avg_val,
            }, path)
            log(f"  New best model! val_loss={avg_val:.4f}")

    # 保存最终模型
    final_path = "models/vits_final.pt"
    os.makedirs("models", exist_ok=True)
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "train_loss": avg["total"],
        "val_loss": avg_val,
    }, final_path)
    log(f"Training complete! Final model saved to {final_path}")
    log_file.close()


if __name__ == "__main__":
    train_ljspeech()
