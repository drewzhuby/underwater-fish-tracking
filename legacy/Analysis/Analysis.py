import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置 seaborn 主题，但不重置字体
sns.set_theme(style="whitegrid", font='SimHei', rc={'axes.unicode_minus': False})

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# CLAHE增强
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


# 多尺度Retinex增强（修改后）
def apply_multiscale_retinex(image, sigma_list=[15, 80, 250]):
    def single_scale_retinex(img, sigma):
        gaussian = gaussian_filter(img, sigma)
        retinex = np.log1p(img) - np.log1p(gaussian)
        return retinex

    # 对每个颜色通道分别处理
    enhanced_channels = []
    for i in range(3):  # 分别处理 B、G、R 通道
        channel = image[:, :, i].astype(np.float32) + 1.0
        retinex_result = np.zeros_like(channel)

        for sigma in sigma_list:
            retinex_result += single_scale_retinex(channel, sigma)

        retinex_result /= len(sigma_list)
        retinex_result = cv2.normalize(retinex_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced_channels.append(retinex_result)

    # 合并通道
    enhanced_image = cv2.merge(enhanced_channels)
    return enhanced_image


# Gamma增强
def apply_gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


# 暗通道先验（Dark Channel Prior, DCP）
def apply_dark_channel_prior(image, omega=0.95, t0=0.1):
    # 计算暗通道
    dark_channel = cv2.erode(np.min(image, axis=2), np.ones((15, 15), np.uint8))

    # 估计大气光
    atmospheric_light = np.percentile(dark_channel, 99)

    # 计算透射率
    transmission = 1 - omega * (dark_channel / atmospheric_light)
    transmission = np.maximum(transmission, t0)

    # 恢复清晰图像
    restored_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        restored_image[:, :, i] = (image[:, :, i] - atmospheric_light) / transmission + atmospheric_light

    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    return restored_image


# 图像融合方法
def apply_image_fusion(image):
    # 使用 CLAHE 和 Retinex 分别增强图像
    clahe_image = apply_clahe(image)
    retinex_image = apply_multiscale_retinex(image)

    # 对两幅图像进行加权融合
    fused_image = cv2.addWeighted(clahe_image, 0.5, retinex_image, 0.5, 0)
    return fused_image


# 计算指标
def calculate_metrics(original, enhanced):
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    entropy = lambda img: -np.sum(np.histogram(img, bins=256, range=(0, 256))[0] / img.size * np.log2(
        np.histogram(img, bins=256, range=(0, 256))[0] / img.size + 1e-7))
    psnr = cv2.PSNR(original, enhanced)
    ssim_value = ssim(gray_original, gray_enhanced)

    def edge_sharpness(img):
        edges = cv2.Canny(img, 100, 200)
        return np.mean(edges)

    edge_enhanced = edge_sharpness(gray_enhanced)

    return {
        'PSNR': psnr,
        'SSIM': ssim_value,
        'Entropy (Enhanced)': entropy(gray_enhanced),
        'Edge Sharpness (Enhanced)': edge_enhanced
    }


# 批量处理图像
def process_images_in_folder(folder_path):
    methods = ['CLAHE', 'Retinex', 'Gamma', 'DCP', 'Image Fusion']
    metrics_accumulator = {
        method: {'PSNR': 0, 'SSIM': 0, 'Entropy (Enhanced)': 0, 'Edge Sharpness (Enhanced)': 0, 'Time': 0} for method in
        methods}
    image_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is None:
            continue

        image_count += 1

        # Process with each method and measure time
        for method, func in zip(methods, [apply_clahe, apply_multiscale_retinex, apply_gamma_correction,
                                          apply_dark_channel_prior, apply_image_fusion]):
            start_time = time.time()
            if method == 'Gamma':
                enhanced_image = func(image, gamma=1.5)
            else:
                enhanced_image = func(image)
            elapsed_time = time.time() - start_time

            metrics = calculate_metrics(image, enhanced_image)

            # Accumulate metrics
            for key in metrics:
                metrics_accumulator[method][key] += metrics[key]
            metrics_accumulator[method]['Time'] += elapsed_time

    # Calculate averages
    for method in metrics_accumulator:
        for key in metrics_accumulator[method]:
            metrics_accumulator[method][key] = round(metrics_accumulator[method][key] / image_count, 4)

    return metrics_accumulator, image_count


# 绘制直方图
def plot_metrics(metrics):
    methods = list(metrics.keys())
    metrics_keys = ['PSNR', 'SSIM', 'Entropy (Enhanced)', 'Edge Sharpness (Enhanced)', 'Time']
    metrics_titles = ['PSNR (峰值信噪比)', 'SSIM (结构相似性)', 'Entropy (熵)', 'Edge Sharpness (边缘锐度)',
                      'Time (处理时间)']

    plt.figure(figsize=(18, 12))
    sns.set_theme(style="whitegrid")  # 设置Seaborn主题
    palette = sns.color_palette("coolwarm", len(methods))  # 使用渐变配色

    # 显式设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for i, key in enumerate(metrics_keys):
        values = [metrics[method][key] for method in methods]
        plt.subplot(3, 2, i + 1)
        bars = plt.bar(methods, values, color=palette, width=0.5, edgecolor='black')
        plt.title(metrics_titles[i], fontsize=16, fontweight='bold')
        plt.ylabel(key, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)

        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # 标注每个柱子上的数值
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{value:.4f}',
                     ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()


# 文件夹路径
folder_path = '<LOCAL_PATH>'

# 处理图像
metrics, total_images = process_images_in_folder(folder_path)

# 打印结果
print(f"Processed {total_images} images.")
for method, metric_values in metrics.items():
    print(f"\n{method} Averages:")
    for key, value in metric_values.items():
        print(f"  {key}: {value:.4f}")

# 绘制直方图
plot_metrics(metrics)