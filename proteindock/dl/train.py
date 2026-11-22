# -*- coding: utf-8 -*-
"""EGNN 打分模型训练

 训练目标(knowledge distillation): 以经验打分函数在大量
 native/decoy pose 上的能量值为教师标签, 训练等变 GNN 拟合
 「结构 -> 能量」映射; 若数据集带实验亲和力标签亦可直接拟合 pKd。

 产物: best.pt(权重), history.json(loss 曲线数据), loss_curve.png。
"""

import json
import os
import time

import numpy as np
import torch

from .dataset import PoseSampleDataset, collate_graphs
from .model import EGNNPoseScorer


def train_egnn(data_root, save_dir, epochs=30, batch_size=8, lr=1e-3,
               hidden_dim=96, n_layers=4, device=None, log_every=10):
    """训练主入口。

    :param data_root: dockdata 数据集根目录
    :param save_dir: 模型与日志输出目录
    :return: dict 训练摘要(best_val_loss, epochs, seconds, ...)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseSampleDataset(data_root)
    if len(dataset) == 0:
        raise ValueError("数据集为空: {}".format(data_root))
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(2025)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    train_set = torch.utils.data.Subset(dataset, perm[:n_train])
    val_set = torch.utils.data.Subset(dataset, perm[n_train:])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_graphs, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_graphs, num_workers=2)

    model = EGNNPoseScorer(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    history = {"train": [], "val": [], "lr": []}
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch, labels in train_loader:
            if batch is None:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            labels = labels.to(device)
            pred = model(batch)
            loss = torch.nn.functional.smooth_l1_loss(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch, labels in val_loader:
                if batch is None:
                    continue
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                labels = labels.to(device)
                pred = model(batch)
                val_losses.append(float(
                    torch.nn.functional.smooth_l1_loss(pred, labels).item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "hidden_dim": hidden_dim, "n_layers": n_layers,
                        "epoch": epoch, "val_loss": val_loss},
                       os.path.join(save_dir, "best.pt"))
        if epoch % log_every == 0 or epoch == 1:
            print("epoch {:3d}/{:3d}  train {:.4f}  val {:.4f}  best {:.4f}"
                  .format(epoch, epochs, train_loss, val_loss, best_val))

    seconds = time.time() - t0
    summary = {
        "best_val_loss": best_val,
        "epochs": epochs,
        "n_train": n_train, "n_val": n_val,
        "n_systems": dataset.n_systems(),
        "device": str(device),
        "seconds": round(seconds, 1),
    }
    with open(os.path.join(save_dir, "history.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"summary": summary, "history": history}, fh,
                  ensure_ascii=False, indent=2)
    _plot_history(history, os.path.join(save_dir, "loss_curve.png"))
    print("训练完成: best val loss {:.4f}, 用时 {:.0f}s -> {}"
          .format(best_val, seconds, save_dir))
    return summary


def _plot_history(history, path):
    """绘制训练/验证损失曲线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["train"], label="train loss")
        ax.plot(history["val"], label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("SmoothL1 loss")
        ax.set_title("ProteinDock EGNN Scorer Training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception as exc:      # matplotlib 缺失不影响训练
        print("跳过曲线绘制: {}".format(exc))
