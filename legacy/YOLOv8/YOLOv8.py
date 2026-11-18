import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path


model = YOLO('<LOCAL_PATH>')  # 替换为你的 .pt 模型文件路径


output_root_dir = '<LOCAL_PATH>'  # 替换为你希望存储图像的根目录路径


output_dirs = ['fish 1', 'fish 2', 'fish 3', 'fish 4']


if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)


for folder in output_dirs:
    fish_folder = os.path.join(output_root_dir, folder)
    if not os.path.exists(fish_folder):
        os.makedirs(fish_folder)


input_folder = '<LOCAL_PATH>'  # 替换为你的鱼类图像文件夹路径
image_paths = list(Path(input_folder).glob('frame_*.jpg'))  # 只处理 frame_x.jpg 格式的图片


output_txt_path = os.path.join(output_root_dir, 'target_count.txt')


with open(output_txt_path, 'w') as txt_file:

    for image_path in image_paths:

        img = cv2.imread(str(image_path))

        # 使用 YOLOv8 进行预测
        results = model(img)  # 预测图像中的对象

        # 获取预测结果
        predictions = results[0].boxes  # 获取第一个图像的预测结果
        boxes = predictions.xyxy  # 获取边界框 [x1, y1, x2, y2]
        classes = predictions.cls.tolist()  # 类别（鱼的类别）

        # 为每个类别创建一个计数器
        fish_counters = [0, 0, 0, 0]  # 用于计数每个类别中目标的数量（fish 1, fish 2, fish 3, fish 4）

        # 遍历每个预测目标
        for i, box in enumerate(boxes):
            # 获取类别并截取图像
            class_id = int(classes[i])

            # 只保存类别在 0-3 范围内的目标
            if class_id < 4:
                # 提取目标框
                x1, y1, x2, y2 = map(int, box)
                target_img = img[y1:y2, x1:x2]

                # 获取目标的文件夹路径
                target_folder = os.path.join(output_root_dir, output_dirs[class_id])

                # 更新计数器
                fish_counters[class_id] += 1

                # 构建新的文件名，确保命名格式为 frame_x_fish_x(y).jpg
                target_image_path = os.path.join(target_folder, f'{image_path.stem}_fish_{class_id + 1}({fish_counters[class_id]}).jpg')

                # 如果文件已存在，添加一个编号避免覆盖
                count = 1
                while os.path.exists(target_image_path):
                    target_image_path = os.path.join(target_folder, f'{image_path.stem}_fish_{class_id + 1}({fish_counters[class_id]}_{count}).jpg')
                    count += 1

                # 保存截取的图像
                cv2.imwrite(target_image_path, target_img)
                print(f'Saved: {target_image_path}')

        # 将每张图片的目标数目写入到 txt 文件中
        txt_file.write(f'{image_path.stem}: fish 1: {fish_counters[0]}, fish 2: {fish_counters[1]}, fish 3: {fish_counters[2]}, fish 4: {fish_counters[3]}\n')

print("所有鱼类图像已处理完毕。目标数目已记录在 target_count.txt 文件中。")








