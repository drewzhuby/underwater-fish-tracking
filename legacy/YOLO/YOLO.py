from ultralytics import YOLO
import torch
import os
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

# 检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 打印当前工作目录
print("当前工作目录:", os.getcwd())

# 自定义记录损失的列表
loss_records = []  # 用于存储每轮损失的列表

def on_train_epoch_end(trainer):
    """在每轮训练结束后计算损失系数"""
    epoch = trainer.epoch
    # 从 trainer.loss_items 获取训练损失分量
    # loss_items 通常包含当前批次的平均损失
    if hasattr(trainer, "loss_items"):
        loss_items = trainer.loss_items
        # 假设 loss_items 的格式为 [box_loss, cls_loss, dfl_loss, ...]
        train_box_loss = float(loss_items[0]) if len(loss_items) > 0 else 0
        train_cls_loss = float(loss_items[1]) if len(loss_items) > 1 else 0
        train_dfl_loss = float(loss_items[2]) if len(loss_items) > 2 else 0
    else:
        train_box_loss, train_cls_loss, train_dfl_loss = 0, 0, 0

    # 从 trainer 中获取验证损失（如果存在）
    val_box_loss = trainer.metrics.get("val/box_loss", 0)
    val_cls_loss = trainer.metrics.get("val/cls_loss", 0)
    val_dfl_loss = trainer.metrics.get("val/dfl_loss", 0)

    # 记录损失
    losses = {
        "epoch": epoch + 1,
        "train/box_loss": train_box_loss,
        "train/cls_loss": train_cls_loss,
        "train/dfl_loss": train_dfl_loss,
        "val/box_loss": val_box_loss,
        "val/cls_loss": val_cls_loss,
        "val/dfl_loss": val_dfl_loss,
    }
    loss_records.append(losses)
    print(f"Epoch {epoch + 1}: {losses}")

# 平滑曲线函数
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

if __name__ == '__main__':
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'

    # 加载模型
    model = YOLO("yolov8n.pt")

    # 训练模型
    epochs = 50
    batch_size = 8
    imgsz = 640
    patience = 0

    # 记录训练开始时间
    total_start_time = time.time()

    # 添加回调函数
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    results = model.train(
        data=r"<LOCAL_PATH>",
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        verbose=True,
        patience=patience,
        weight_decay=0.0005,
        dropout=0.3,
        lr0=0.01,
        lrf=0.0001
    )

    # 显示训练进度（仅用于可视化）
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        pass

    # 记录总训练时间
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"总训练时间: {total_training_time:.2f} 秒")

    # 获取检测指标
    metrics_dict = results.results_dict

    # 保存评估结果
    save_path = r"<LOCAL_PATH>"
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("评估结果：\n")
        f.write(f"总训练时间: {total_training_time:.2f} 秒\n")
        f.write(f"最终精度 (Precision): {metrics_dict['metrics/precision(B)']:.4f}\n")
        f.write(f"最终召回率 (Recall): {metrics_dict['metrics/recall(B)']:.4f}\n")
        f.write(f"最终mAP@0.5 (mAP@0.5): {metrics_dict['metrics/mAP50(B)']:.4f}\n")
        f.write(f"最终mAP@0.5:0.95 (mAP@0.5:0.95): {metrics_dict['metrics/mAP50-95(B)']:.4f}\n")
        f.write(f"最终适应度 (Fitness): {metrics_dict['fitness']:.4f}\n")
    print(f"评估结果已保存到: {save_path}")

    # 导出训练过程中的损失系数到 CSV
    loss_df = pd.DataFrame(loss_records)
    loss_csv_path = r"<LOCAL_PATH>"
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"损失系数已保存到: {loss_csv_path}")

    # 计算总训练损失和验证损失
    train_loss = [r["train/box_loss"] + r["train/cls_loss"] + r["train/dfl_loss"] for r in loss_records]
    val_loss = [r["val/box_loss"] + r["val/cls_loss"] + r["val/dfl_loss"] for r in loss_records]
    epochs_list = [r["epoch"] for r in loss_records]

    # 平滑损失曲线
    smooth_train_loss = smooth_curve(train_loss)
    smooth_val_loss = smooth_curve(val_loss)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_loss, "r-", label="train loss")
    plt.plot(epochs_list, val_loss, "g--", label="val loss")
    plt.plot(epochs_list, smooth_train_loss, "b-", label="smooth train loss")
    plt.plot(epochs_list, smooth_val_loss, "m--", label="smooth val loss")
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)

    # 保存损失曲线图
    loss_plot_path = r"<LOCAL_PATH>"
    plt.savefig(loss_plot_path, bbox_inches='tight')
    plt.close()
    print(f"损失曲线图已保存到: {loss_plot_path}")

    # 验证模型
    metrics_val = model.val()
    print("Validation Metrics:", metrics_val)

    # 保存模型到指定路径
    save_model_path = r"<LOCAL_PATH>"
    model.save(save_model_path)
    print(f"模型已保存到: {os.path.abspath(save_model_path)}")












