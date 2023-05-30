from pathlib import Path
from typing import Union

import torch
import torchvision.transforms as transforms
from PIL import Image

from . import utils
from .lit_models import CTCLitModel

MODEL_CKPT_FILENAME = "./artifacts/lit_model.ckpt"

class Recognizer:
    """Model used for production."""

    def __init__(self):
        self.transform = transforms.ToTensor()
        self.model = CTCLitModel.load_from_checkpoint(MODEL_CKPT_FILENAME)
        self.model.freeze()
    
    def resize_image(self, image_pil, target_height=32):
        # 获取原始图像的宽度和高度
        orig_width, orig_height = image_pil.size

        # 计算新的宽度，保持原始的宽高比
        new_width = int(orig_width * target_height / orig_height)

        # 使用PIL的resize方法创建新的图像
        new_image_pil = image_pil.resize((new_width, target_height), Image.ANTIALIAS)

        return new_image_pil

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict the number in the input image.

        Args:
            image: can be a path to the image file or an instance of Pillow image.

        Returns:
            The predicted number. "None" if the model cannot detect any number.
        """
        if isinstance(image, Image.Image):
            image_pil = image
        else:
            image_pil = utils.read_image_pil(image, grayscale=True)

        image_pil_new = self.resize_image(image_pil)
        image_tensor = self.transform(image_pil_new)
        decoded, pred_lengths = self.model(image_tensor.unsqueeze(0))
        # Remove the paddings
        digit_lists = decoded[0][: pred_lengths[0]]
        if len(digit_lists) == 0:
            return "None"
        # Concatenate the digits
        pred_num = "".join(str(i) for i in digit_lists.tolist())
        return pred_num
