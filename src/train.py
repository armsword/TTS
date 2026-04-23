"""训练模块 - 损失函数和训练循环"""
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def compute_loss(model_output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                 phoneme_lengths: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """计算 VITS 模型的总损失

    Args:
        model_output: 模型输出，包含:
            - mel_output: 预测的梅尔频谱 (batch, n_mels, time)
            - duration_pred: 预测的时长 (batch, time)
            - mu: VAE 均值 (batch, time, latent_dim)
            - log_var: VAE 对数方差 (batch, time, latent_dim)
        targets: 目标，包含:
            - mel: 目标梅尔频谱 (batch, n_mels, time)
        phoneme_lengths: 每个样本的实际音素数 (batch,)
    """
    mel_output = model_output["mel_output"]
    duration_pred = model_output["duration_pred"]
    mu = model_output["mu"]
    log_var = model_output["log_var"]
    mel_target = targets["mel"]

    # 梅尔频谱重建损失 (L1)
    mel_loss = torch.nn.functional.l1_loss(mel_output, mel_target)

    # 时长预测损失 (MSE) - 只在有效音素位置计算
    duration_target = targets.get("duration", None)
    if duration_target is not None and phoneme_lengths is not None:
        # 构建 mask：只保留有效音素位置
        batch_size, max_len = duration_pred.shape
        mask = torch.zeros_like(duration_pred)
        for b in range(batch_size):
            mask[b, :phoneme_lengths[b]] = 1.0

        duration_pred_log = torch.log(duration_pred.clamp(min=1e-5))
        duration_target_log = torch.log(duration_target.clamp(min=1e-5))
        diff = (duration_pred_log - duration_target_log) ** 2
        duration_loss = (diff * mask).sum() / mask.sum().clamp(min=1)
    else:
        duration_loss = torch.tensor(0.0, device=mel_output.device)

    # KL 散度损失
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / (mu.size(0) * mu.size(1) * mu.size(2))

    total_loss = mel_loss + duration_loss + kl_loss

    return {
        "total_loss": total_loss,
        "mel_loss": mel_loss,
        "duration_loss": duration_loss,
        "kl_loss": kl_loss,
    }


def get_device() -> str:
    """自动选择设备

    Returns:
        设备字符串: "cuda", "mps", 或 "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 200,
    lr: float = 1e-4,
    save_interval: int = 10,
    checkpoint_dir: str = "checkpoints",
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, list]:
    """训练循环

    Args:
        model: VITS 模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器（可选）
        epochs: 训练轮数
        lr: 学习率
        save_interval: 保存 checkpoint 的间隔（轮）
        checkpoint_dir: checkpoint 保存目录
        resume_from_checkpoint: 从指定 checkpoint 恢复训练

    Returns:
        训练历史 {train_losses: [], val_losses: []}
    """
    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_losses": [],
        "val_losses": [],
    }

    start_epoch = 0

    # 从 checkpoint 恢复
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"从 epoch {start_epoch} 恢复训练")

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            # 将数据移到设备
            phoneme_ids = batch["phoneme_ids"].to(device)
            phoneme_lengths = batch["phoneme_lengths"].to(device)
            mel = batch["mel"].to(device)
            durations = batch["duration"].to(device)

            # 前向传播（使用 ground truth durations 做 teacher forcing）
            model_outputs = model(phoneme_ids, phoneme_lengths, mel, durations=durations)

            # 计算损失（只对有效音素位置计算 duration loss）
            targets = {"mel": mel, "duration": durations}
            losses = compute_loss(model_outputs, targets, phoneme_lengths)
            losses = compute_loss(model_outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            epoch_losses.append(losses["total_loss"].item())

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, "
                      f"Loss: {losses['total_loss'].item():.4f}")

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        history["train_losses"].append(avg_train_loss)

        # 验证
        if val_dataloader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dataloader:
                    phoneme_ids = batch["phoneme_ids"].to(device)
                    phoneme_lengths = batch["phoneme_lengths"].to(device)
                    mel = batch["mel"].to(device)
                    durations = batch["duration"].to(device)

                    model_outputs = model(phoneme_ids, phoneme_lengths, mel, durations=durations)
                    targets = {"mel": mel, "duration": durations}
                    losses = compute_loss(model_outputs, targets, phoneme_lengths)
                    val_losses.append(losses["total_loss"].item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            history["val_losses"].append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # 保存 checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return history


if __name__ == "__main__":
    import argparse
    from model.vits import VITS
    from config import (
        TextEncoderConfig, DecoderConfig, DurationPredictorConfig,
        HiFiGANConfig, TrainConfig
    )
    from dataset import get_dataloader

    parser = argparse.ArgumentParser(description="Train VITS model")
    parser.add_argument("--train_filelist", type=str, required=True)
    parser.add_argument("--val_filelist", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # 创建模型
    configs = {
        "text_encoder": TextEncoderConfig(),
        "decoder": DecoderConfig(),
        "duration_predictor": DurationPredictorConfig(),
        "hifigan": HiFiGANConfig(),
        "train": TrainConfig(),
    }
    model = VITS(configs)

    # 创建数据加载器
    train_dataloader = get_dataloader(args.train_filelist, args.data_dir, batch_size=args.batch_size)
    val_dataloader = None
    if args.val_filelist:
        val_dataloader = get_dataloader(args.val_filelist, args.data_dir, batch_size=args.batch_size)

    # 训练
    train(model, train_dataloader, val_dataloader, epochs=args.epochs, lr=args.lr,
         checkpoint_dir=args.checkpoint_dir, resume_from_checkpoint=args.resume_from_checkpoint)
