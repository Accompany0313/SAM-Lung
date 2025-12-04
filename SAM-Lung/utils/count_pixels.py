from PIL import Image
import numpy as np

# 增加最大图像像素限制
Image.MAX_IMAGE_PIXELS = None
def count_white_pixels(mask_path):
    # 读取掩码图像
    mask = Image.open(mask_path).convert('L')  # 转为灰度图

    # 将图像转换为numpy数组
    mask_array = np.array(mask)

    # 统计不是黑色的像素
    white_pixel_count = np.sum(mask_array > 0)

    return white_pixel_count


# 示例用法
mask_path = 'path_to_your_mask.png'  # 替换为掩码文件路径
white_pixels = count_white_pixels(mask_path)
print(f"White pixel count: {white_pixels}")
