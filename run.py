"""Portable enhancement demo and optional YOLO / DeepSORT video runner."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def enhance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def demo(output):
    image = np.full((240, 400, 3), 40, np.uint8)
    cv2.ellipse(image, (160, 120), (45, 20), 0, 0, 360, (60, 70, 65), -1)
    current = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, 0]]), (400, 240))
    previous_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, .5, 3, 25, 5, 7, 1.5, 0)
    cv2.imwrite(str(output / 'demo.jpg'), np.hstack([image, enhance(image)]))
    result = {'source': 'synthetic translated ellipse; no detector or tracker',
              'center_flow_dx_dy': flow[120, 160].tolist()}
    (output / 'demo.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def video(args, output):
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    if not args.weights or not Path(args.weights).is_file():
        raise ValueError('--weights must point to a local detector checkpoint')
    model = YOLO(args.weights)
    tracker = DeepSort(max_age=30, n_init=3, embedder=None)
    # Caller-supplied simple color histograms avoid an implicit ReID weight download.
    # This portable example differs from the historical YOLO-feature association.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {args.video}')
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        size = (int(cap.get(3)), int(cap.get(4)))
        writer = cv2.VideoWriter(str(output/'tracked.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            raise RuntimeError('Video encoder unavailable')
        with (output/'tracks.csv').open('w', newline='') as f:
            table = csv.writer(f)
            table.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                image = enhance(frame) if args.clahe else frame
                detections, embeddings = [], []
                result = model.predict(image, conf=args.conf, verbose=False)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, x2 = np.clip([x1, x2], 0, size[0])
                    y1, y2 = np.clip([y1, y2], 0, size[1])
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    hist = cv2.calcHist([crop], [0, 1, 2], None, [4, 4, 4], [0, 256]*3).flatten()
                    hist = hist / max(float(np.linalg.norm(hist)), 1e-8)
                    embeddings.append(hist)
                    detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf.item()), int(box.cls.item())))
                tracks = tracker.update_tracks(detections, embeds=embeddings)
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue
                    coords = track.to_ltrb()
                    table.writerow([index, track.track_id, *map(float, coords)])
                    a, b, c, d = map(int, coords)
                    cv2.rectangle(frame, (a, b), (c, d), (0, 255, 0), 2)
                    cv2.putText(frame, str(track.track_id), (a, max(b, 15)), 0, .6, (0,255,0), 2)
                writer.write(frame)
    finally:
        cap.release()
        if writer:
            writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--video')
    parser.add_argument('--weights')
    parser.add_argument('--clahe', action='store_true')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--output', default='outputs/demo')
    args = parser.parse_args()
    if not args.demo and not args.video:
        parser.error('Choose --demo or --video')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    demo(output) if args.demo else video(args, output)
