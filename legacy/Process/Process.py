import cv2
import numpy as np


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # 转换到 LAB 颜色空间
    l, a, b = cv2.split(lab)  # 拆分通道

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 创建 CLAHE 对象
    l_clahe = clahe.apply(l)  # 只对 L 通道应用 CLAHE

    lab_clahe = cv2.merge((l_clahe, a, b))  # 重新合并通道
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)  # 转回 BGR


def remove_bubbles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)  # 边缘检测
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)  # 扩展边缘

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones_like(gray) * 255  # 创建掩码

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:  # 过滤可能的气泡区域
            cv2.drawContours(mask, [cnt], -1, 0, thickness=cv2.FILLED)

    result = cv2.inpaint(frame, 255 - mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no_bubbles = remove_bubbles(frame)  # 先去除气泡
        frame_clahe = apply_clahe(frame_no_bubbles)  # 再进行 CLAHE 处理

        out.write(frame_clahe)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = "<LOCAL_PATH>"  # 输入视频路径
    output_video = "<LOCAL_PATH>"  # 输出视频路径
    process_video(input_video, output_video)



