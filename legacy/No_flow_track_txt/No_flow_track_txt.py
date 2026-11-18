import sys
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import os

sys.path.append("<LOCAL_PATH>")
from deep_sort.deep_sort import DeepSort
import torch.cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO("<LOCAL_PATH>").to(device)

try:
    deepsort = DeepSort(
        model_path=None, max_dist=0.7, min_confidence=0.3, nms_max_overlap=1.0,
        max_iou_distance=0.7, max_age=70, n_init=1, nn_budget=100, use_cuda=torch.cuda.is_available()
    )
    print("DeepSort initialized.")
except Exception as e:
    print(f"DeepSort init failed: {e}")
    exit()

cap = cv2.VideoCapture("<LOCAL_PATH>")
if not cap.isOpened():
    print("Cannot open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

def extract_yolo_features(model, frame, bboxes):
    features = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            print(f"Invalid feature bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            features.append(np.zeros(512))
            continue
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"Empty crop: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            features.append(np.zeros(512))
            continue
        crop = cv2.resize(crop, (128, 128))
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float().to(device) / 255.0
        crop_tensor = crop_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = model.model(crop_tensor, augment=False)
            feat = F.adaptive_avg_pool2d(outputs[0], (1, 1)).flatten()
            if feat.shape[0] < 512:
                feat = F.pad(feat, (0, 512 - feat.shape[0]))
            elif feat.shape[0] > 512:
                feat = feat[:512]
            feat = feat.cpu().numpy()
            if np.isnan(feat).any() or not np.isfinite(feat).all():
                print(f"Invalid feature: {feat}")
                feat = np.zeros(512)
            features.append(feat)
    features = np.array(features)
    print(f"Features shape: {features.shape}")
    return features if features.size > 0 else np.empty((0, 512))

frame_count = 0
tracks = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_frame = frame_count
    print(f"Processing frame {current_frame}/{total_frames}")

    frame_processed = frame
    frame_resized = cv2.resize(frame_processed, (640, 384))
    results = model(frame_resized, conf=0.3, iou=0.5, imgsz=640, max_det=10)
    detections = results[0].boxes.data.cpu().numpy()
    print(f"Detected {len(detections)} objects.")
    if len(detections) > 0:
        print(f"Raw detections: x1={detections[:, 0].min():.2f}-{detections[:, 0].max():.2f}, "
              f"y1={detections[:, 1].min():.2f}-{detections[:, 1].max():.2f}")

    if len(detections) > 0:
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, 640)
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, 384)
        scale_x = 1920 / 640  # 3
        scale_y = 1080 / 384  # ≈2.8125
        detections[:, [0, 2]] *= scale_x
        detections[:, [1, 3]] *= scale_y
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, 1920)
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, 1080)
        print(f"Scaled detections: x=[{detections[:, 0].min():.2f}, {detections[:, 2].max():.2f}], "
              f"y=[{detections[:, 1].min():.2f}, {detections[:, 3].max():.2f}]")

    bboxes = []
    scores = []
    classes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            x = x1
            y = y1
            bboxes.append([x + w/2, y + h/2, w, h])
            scores.append(conf)
            classes.append(int(cls))
        else:
            print(f"Invalid bbox: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, w={w:.2f}, h={h:.2f}")

    bboxes = np.array(bboxes)
    scores = np.array(scores)
    classes = np.array(classes)
    print(f"Bboxes count: {len(bboxes)}")

    features = None
    if bboxes.size > 0:
        bboxes_xyxy = []
        for bbox in bboxes:
            x_c, y_c, w, h = bbox
            x1 = x_c - w/2
            y1 = y_c - h/2
            x2 = x_c + w/2
            y2 = y_c + h/2
            bboxes_xyxy.append([x1, y1, x2, y2])
        bboxes_xyxy = np.array(bboxes_xyxy)
        features = extract_yolo_features(model, frame, bboxes_xyxy)
    else:
        print("No valid bboxes for feature extraction.")

    outputs = []
    if bboxes.size > 0 and features.size > 0:
        try:
            outputs, _ = deepsort.update(bboxes, scores, classes, frame, features=features)
            print(f"Tracks: {len(outputs)}")
            if len(outputs) > 0:
                print(f"Tracks range: x=[{min(o[0] for o in outputs):.2f}, {max(o[2] for o in outputs):.2f}], "
                      f"y=[{min(o[1] for o in outputs):.2f}, {max(o[3] for o in outputs):.2f}]")
        except Exception as e:
            print(f"DeepSort update failed: {e}")

    if len(outputs) > 0:
        for output in outputs:
            x1, y1, x2, y2, cls, track_id = output
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                tracks.append((current_frame, track_id, x, y, w, h))
            else:
                print(f"Invalid track: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")

    if len(outputs) == 0:
        print(f"No tracks for frame {current_frame}, appending placeholder")
        tracks.append((current_frame, -1, 0.0, 0.0, 0.0, 0.0))  # 占位符

output_file = "<LOCAL_PATH>"
with open(output_file, 'w') as f:
    f.write("frame_id,track_id,x,y,w,h\n")
    for track in tracks:
        frame_id, track_id, x, y, w, h = track
        if track_id == -1:
            f.write(f"{frame_id},{track_id},0.00,0.00,0.00,0.00\n")
        else:
            f.write(f"{frame_id},{track_id},{x:.5f},{y:.5f},{w:.5f},{h:.5f}\n")

print(f"Tracking results saved to {output_file}")
cap.release()