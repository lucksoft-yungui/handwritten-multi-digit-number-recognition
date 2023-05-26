import logging

import handwritten_multi_digit_number_recognition.utils as utils
from handwritten_multi_digit_number_recognition import Recognizer

model = Recognizer()

pred_num = model.predict("/kaggle/working/src/images/test_image.png")

print(f"pred_num:{pred_num}")