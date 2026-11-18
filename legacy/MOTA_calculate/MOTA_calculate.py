import pandas as pd
import motmetrics as mm
import numpy as np
from pathlib import Path


def load_mot_file(file_path, is_groundtruth=False, skip_first_frame=False):
    """
    读取 MOT 格式文件
    - gt.txt 格式：frame_id, gt_id, x, y, w, h, conf, -1, -1
    - tracking_results.txt 格式：frame_id, track_id, x, y, w, h
    - skip_first_frame: 对 tracking_results.txt 跳过第一帧
    - 返回 DataFrame：frame, id, x, y, w, h
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} not found.")

    if is_groundtruth:
        df = pd.read_csv(file_path, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])
        df = df[['frame', 'id', 'x', 'y', 'w', 'h']]
    else:
        df = pd.read_csv(file_path, names=['frame', 'id', 'x', 'y', 'w', 'h'], skiprows=1)
        if skip_first_frame:
            df = df[df['frame'] > 1]

    df['frame'] = df['frame'].astype(int)
    df['id'] = df['id'].astype(int)
    df[['x', 'y', 'w', 'h']] = df[['x', 'y', 'w', 'h']].astype(float)

    df = df[(df['w'] > 0) & (df['h'] > 0) & (df['id'] >= 0)]

    return df


def compute_iou(box1, box2):
    """
    计算 IOU
    - box: [x, y, w, h]（左上角）
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_mot(gt_file, hyp_file, iou_threshold=0.5):
    """
    计算 MOTA 等指标
    """
    gt_df = load_mot_file(gt_file, is_groundtruth=True, skip_first_frame=False)
    hyp_df = load_mot_file(hyp_file, is_groundtruth=False, skip_first_frame=True)

    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(gt_df['frame']).union(set(hyp_df['frame'])))

    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        hyp_frame = hyp_df[hyp_df['frame'] == frame]

        gt_ids = gt_frame['id'].values
        hyp_ids = hyp_frame['id'].values

        gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
        hyp_boxes = hyp_frame[['x', 'y', 'w', 'h']].values

        distances = np.zeros((len(gt_boxes), len(hyp_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, hyp_box in enumerate(hyp_boxes):
                iou = compute_iou(gt_box, hyp_box)
                distances[i, j] = 1.0 - iou if iou >= 0 else np.inf

        acc.update(gt_ids, hyp_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames', 'mota', 'motp', 'idf1',
            'num_matches', 'num_false_positives', 'num_misses',
            'num_switches', 'num_objects'
        ],
        return_dataframe=False  # 返回字典
    )

    return summary


def main():
    gt_file = "<LOCAL_PATH>"
    hyp_file = "<LOCAL_PATH>"

    try:
        summary = evaluate_mot(gt_file, hyp_file, iou_threshold=0.5)

        # 打印结果
        print("\nMOT Evaluation Results:")
        for metric, value in summary.items():
            print(f"{metric}: {value}")

        # 提取指标
        mota = float(summary['mota']) * 100
        motp = float(summary['motp'])
        idf1 = float(summary['idf1']) * 100
        fp = int(summary['num_false_positives'])
        fn = int(summary['num_misses'])
        idsw = int(summary['num_switches'])
        matches = int(summary['num_matches'])
        objects = int(summary['num_objects'])
        num_frames = int(summary['num_frames'])

        print("\nKey Metrics:")
        print(f"Number of Frames: {num_frames}")
        print(f"MOTA: {mota:.2f}%")
        print(f"MOTP: {motp:.4f}")
        print(f"IDF1: {idf1:.2f}%")
        print(f"Matches: {matches}")
        print(f"False Positives (FP): {fp}")
        print(f"Misses (FN): {fn}")
        print(f"ID Switches: {idsw}")
        print(f"Total Objects: {objects}")

        # 保存结果
        output_file = "<LOCAL_PATH>"
        with open(output_file, 'w') as f:
            f.write("MOT Evaluation Results:\n")
            for metric, value in summary.items():
                f.write(f"{metric}: {value}\n")
            f.write("\nKey Metrics:\n")
            f.write(f"Number of Frames: {num_frames}\n")
            f.write(f"MOTA: {mota:.2f}%\n")
            f.write(f"MOTP: {motp:.4f}\n")
            f.write(f"IDF1: {idf1:.2f}%\n")
            f.write(f"Matches: {matches}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"Misses (FN): {fn}\n")
            f.write(f"ID Switches: {idsw}\n")
            f.write(f"Total Objects: {objects}\n")
        print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()