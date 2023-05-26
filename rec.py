import argparse
import handwritten_multi_digit_number_recognition.utils as utils
from handwritten_multi_digit_number_recognition import Recognizer

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Predict handwritten multi-digit number")
parser.add_argument('image_path', type=str, help="Path to the image file")

# 解析命令行参数
args = parser.parse_args()

model = Recognizer()

# 使用命令行参数中的图片地址
pred_num = model.predict(args.image_path)

print(f"pred_num:{pred_num}")