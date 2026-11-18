import cv2
import os

# 视频路径
video_path = "<LOCAL_PATH>"

# 输出文件夹路径
output_dir = "<LOCAL_PATH>"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Output directory already exists: {output_dir}")

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件，请检查路径")
    exit()

# 获取视频属性
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total frames: {total_frames}")

# 帧计数器
frame_count = 1

# 逐帧读取并保存
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 生成输出文件名，例如 "frame_0001.jpg"
    output_file = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")

    # 保存帧为 JPG 文件
    cv2.imwrite(output_file, frame)
    print(f"Saved frame {frame_count}/{total_frames} to {output_file}")

    frame_count += 1

# 释放视频对象
cap.release()
print(f"Finished extracting {frame_count - 1} frames to {output_dir}")