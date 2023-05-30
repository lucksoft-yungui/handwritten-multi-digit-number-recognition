import cv2
import sys
import os

def binarize_image(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image {image_path} not found.")
        return

    # 二值化图像
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # 创建新的文件名
    base, ext = os.path.splitext(image_path)
    new_file_name = base + "_b" + ext

    # 保存二值化后的图像
    cv2.imwrite(new_file_name, binary_img)
    print(f"Binarized image is saved as {new_file_name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python binary.py [image_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    binarize_image(image_path)