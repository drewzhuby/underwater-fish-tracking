import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import xml.etree.ElementTree as ET
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_available = True
except ImportError:
    coco_available = False

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 计算 IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_g, y1_g, x2_g, y2_g = box2
    xi1 = max(x1, x1_g)
    yi1 = max(y1, y1_g)
    xi2 = min(x2, x2_g)
    yi2 = min(y2, y2_g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# 自定义数据集
class FishDataset(Dataset):
    def __init__(self, img_dir, xml_dir, transforms=None):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transforms = transforms
        self.class_names = ['background', 'fish 1', 'fish 2', 'fish 3', 'fish 4']
        self.images = []
        img_files = {f.replace('.jpg', '') for f in os.listdir(img_dir) if f.endswith('.jpg')}
        xml_files = {f.replace('.xml', '') for f in os.listdir(xml_dir) if f.endswith('.xml')}
        valid_files = img_files & xml_files
        for img_file in sorted(img_files):
            if img_file in valid_files:
                self.images.append(f"{img_file}.jpg")
            else:
                print(f"Skipping {img_file}.jpg: no corresponding XML")
        for xml_file in sorted(xml_files):
            if xml_file not in valid_files:
                print(f"Skipping {xml_file}.xml: no corresponding image")
        if not self.images:
            raise ValueError("No valid images with corresponding XML found")
        print(f"Loaded {len(self.images)} valid image-XML pairs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace('.jpg', '.xml'))

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_names[1:]:
                print(f"Warning: Invalid class {name} in {xml_path}, skipping object")
                continue
            label = self.class_names.index(name)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                print(f"Warning: Invalid box [{xmin}, {ymin}, {xmax}, {ymax}] in {xml_path}, skipping")
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transforms:
            img = self.transforms(img)

        return img, target

# XML 转 COCO JSON
def xml_to_coco(xml_dir, img_dir, output_json):
    images = []
    annotations = []
    categories = [{'id': i + 1, 'name': name, 'supercategory': 'fish'}
                  for i, name in enumerate(['fish 1', 'fish 2', 'fish 3', 'fish 4'])]
    ann_id = 1

    img_files = {f.replace('.jpg', '') for f in os.listdir(img_dir) if f.endswith('.jpg')}
    xml_files = {f.replace('.xml', '') for f in os.listdir(xml_dir) if f.endswith('.xml')}
    valid_files = img_files & xml_files

    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.endswith('.jpg'):
            continue
        img_base = img_file.replace('.jpg', '')
        if img_base not in valid_files:
            print(f"Skipping {img_file} in COCO JSON: no corresponding XML")
            continue
        img_id = len(images) + 1
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_file}: failed to load image")
            continue
        height, width = img.shape[:2]
        images.append({
            'id': img_id,
            'file_name': img_file,
            'height': height,
            'width': width,
            'date_captured': datetime.now().isoformat()
        })

        xml_file = img_file.replace('.jpg', '.xml')
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall('object'):
            name = obj.find('name').text
            if name not in ['fish 1', 'fish 2', 'fish 3', 'fish 4']:
                print(f"Warning: Invalid class {name} in {xml_path}, skipping object")
                continue
            cat_id = next(c['id'] for c in categories if c['name'] == name)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                print(f"Warning: Invalid box [{xmin}, {ymin}, {xmax}, {ymax}] in {xml_path}, skipping")
                continue
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'area': (xmax - xmin) * (ymax - ymin),
                'iscrowd': 0
            })
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"COCO JSON saved to {output_json}")

# 评估函数
def evaluate(model, data_loader, device, coco_json, score_threshold=0.3):
    if not coco_available:
        print("pycocotools not installed, skipping evaluation")
        return 0.0, 0.0, 0.0, 0.0, 0.0
    model.eval()
    coco = COCO(coco_json)
    coco_dt = []
    true_positives = 0
    false_positives = 0
    total_ground_truths = 0

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)
        for i, (output, target) in enumerate(zip(outputs, targets)):
            img_id = target['image_id'].item() + 1
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()

            total_ground_truths += len(gt_boxes)
            matched_gt = set()
            for pred_idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                if score < score_threshold or label == 0:
                    continue
                coco_dt.append({
                    'image_id': img_id,
                    'category_id': int(label),
                    'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    'score': float(score)
                })
                max_iou = 0
                max_gt_idx = -1
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
                if max_iou >= 0.5 and max_gt_idx != -1 and pred_labels[pred_idx] == gt_labels[max_gt_idx]:
                    true_positives += 1
                    matched_gt.add(max_gt_idx)
                else:
                    false_positives += 1

    if not coco_dt:
        print("No detections, all metrics=0")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0.0
    fitness = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    coco_dt = coco.loadRes(coco_dt)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')

    # mAP@0.5
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_50 = coco_eval.stats[0]

    # mAP@0.5:0.95
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_50_95 = coco_eval.stats[0]

    return precision, recall, fitness, map_50, map_50_95

def main():
    # 记录开始时间
    start_time = time.time()

    # 可调参数
    batch_size = 2
    num_epochs = 20
    smooth_alpha = 0.1  # 平滑参数，用于指数移动平均
    initial_lr = 0.01  # 初始学习率
    min_lr = 0.0001    # 最小学习率

    # 路径配置
    base_dir = "<LOCAL_PATH>"
    train_img_dir = os.path.join(base_dir, "train/images")
    train_xml_dir = os.path.join(base_dir, "train/annotations")
    val_img_dir = os.path.join(base_dir, "val/images")
    val_xml_dir = os.path.join(base_dir, "val/annotations")
    output_dir = "<LOCAL_PATH>"
    coco_train_json = os.path.join(base_dir, "train.json")
    coco_val_json = os.path.join(base_dir, "val.json")
    os.makedirs(output_dir, exist_ok=True)

    # 生成 COCO JSON
    xml_to_coco(train_xml_dir, train_img_dir, coco_train_json)
    xml_to_coco(val_xml_dir, val_img_dir, coco_val_json)

    # 数据集
    train_dataset = FishDataset(
        train_img_dir, train_xml_dir,
        T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    val_dataset = FishDataset(
        val_img_dir, val_xml_dir,
        T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=lambda x: tuple(zip(*x)))

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
    model.to(device)
    model.train()

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)

    # 学习率调度器：余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # 记录损失
    loss_history = []

    # 初始化平滑损失
    smooth_train_loss = None
    smooth_val_loss = None

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        valid_train_batches = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if any(len(t['boxes']) == 0 for t in targets):
                continue
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if torch.isnan(losses) or torch.isinf(losses):
                print("Warning: Invalid loss, skipping batch")
                continue
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += losses.item()
            valid_train_batches += 1

        avg_train_loss = total_train_loss / valid_train_batches if valid_train_batches > 0 else float('nan')

        # 验证阶段
        model.eval()
        total_val_loss = 0
        valid_val_batches = 0
        # 临时切换回训练模式以计算损失
        with torch.no_grad():
            model.train()  # 切换回训练模式以计算损失
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if any(len(t['boxes']) == 0 for t in targets):
                    continue
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                if torch.isnan(losses) or torch.isinf(losses):
                    continue
                total_val_loss += losses.item()
                valid_val_batches += 1
            model.eval()  # 恢复评估模式

        avg_val_loss = total_val_loss / valid_val_batches if valid_val_batches > 0 else float('nan')

        # 计算平滑损失
        if smooth_train_loss is None:
            smooth_train_loss = avg_train_loss
        else:
            smooth_train_loss = (1 - smooth_alpha) * smooth_train_loss + smooth_alpha * avg_train_loss

        if smooth_val_loss is None:
            smooth_val_loss = avg_val_loss
        else:
            smooth_val_loss = (1 - smooth_alpha) * smooth_val_loss + smooth_alpha * avg_val_loss

        # 记录损失
        current_lr = optimizer.param_groups[0]['lr']
        loss_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'smooth_train_loss': smooth_train_loss,
            'smooth_val_loss': smooth_val_loss,
            'learning_rate': current_lr
        })

        # 更新学习率
        scheduler.step()

        # 评估
        precision, recall, fitness, map_50, map_50_95 = evaluate(model, val_loader, device, coco_val_json,
                                                                 score_threshold=0.3)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {current_lr:.6f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Fitness: {fitness:.4f}, mAP@0.5: {map_50:.4f}, mAP@0.5:0.95: {map_50_95:.4f}")

    # 最终评估
    precision, recall, fitness, map_50, map_50_95 = evaluate(model, val_loader, device, coco_val_json,
                                                             score_threshold=0.3)
    print(
        f"Final: Precision: {precision:.4f}, Recall: {recall:.4f}, Fitness: {fitness:.4f}, mAP@0.5: {map_50:.4f}, mAP@0.5:0.95: {map_50_95:.4f}")

    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到 {final_model_path}")

    # 计算总训练时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # 导出最终指标到 txt 文件
    txt_path = r"<LOCAL_PATH>"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("最终评估指标\n")
        f.write(f"精度 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"适应度 (Fitness): {fitness:.4f}\n")
        f.write(f"mAP@0.5: {map_50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map_50_95:.4f}\n")
        f.write(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
    print(f"最终指标已导出到 {txt_path}")

    # 导出损失历史到 csv 文件
    csv_path = r"<LOCAL_PATH>"
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"损失历史已导出到 {csv_path}")

    # 绘制损失曲线并保存
    plot_path = r"<LOCAL_PATH>"
    epochs = [entry['epoch'] for entry in loss_history]
    train_losses = [entry['train_loss'] for entry in loss_history]
    val_losses = [entry['val_loss'] for entry in loss_history]
    smooth_train_losses = [entry['smooth_train_loss'] for entry in loss_history]
    smooth_val_losses = [entry['smooth_val_loss'] for entry in loss_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'r-', label='train loss')
    plt.plot(epochs, val_losses, 'g--', label='val loss')
    plt.plot(epochs, smooth_train_losses, 'b-', label='smooth train loss')
    plt.plot(epochs, smooth_val_losses, 'purple', linestyle='--', label='smooth val loss')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=18)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"损失图表已保存到 {plot_path}")

    print("训练完成！最终模型、评估结果和损失图表已保存")

if __name__ == "__main__":
    main()