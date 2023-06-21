from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 小数点图像的文件路径
image_path = '/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/1.0/raw/train/11776/1.png'

# 打开图像文件
image = Image.open(image_path).convert('L')

# 将图像转换为NumPy数组，进行颜色反转
image_data = np.array(image)
image_data = 255 - image_data  # 颜色反转

# 创建新的图像，显示反转后的图像
image_inverted = Image.fromarray(image_data)
plt.imshow(image_inverted, cmap='gray')
plt.show()