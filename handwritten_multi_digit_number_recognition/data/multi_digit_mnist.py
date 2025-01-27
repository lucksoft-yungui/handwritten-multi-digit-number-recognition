from pathlib import Path
from typing import Optional

import h5py
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from .base_data_module import BaseDataModule
from .single_digit_mnist import SingleDigitMNIST
from .utils import BaseDataset, DatasetGenerator

import matplotlib.pyplot as plt
import numpy as np
import os


class MultiDigitMNIST(BaseDataModule):
    """Data module for a synthetic multi-digit MNIST dataset.

    Args:
        num_train: Number of training samples.
        num_val: Number of validation samples.
        num_test: Number of test samples.
        max_length: Maximum number of digits.
        min_overlap: Minimum proportion of an image being overlapped with another image.
        max_overlap: Maximum proportion of an image being overlapped with another image.
        kwargs: Keyward arguments to BaseDataModule.
    """

    def __init__(
        self,
        num_train: int = 1000,
        num_val: int = 200,
        num_test: int = 200,
        max_length: int = 5,
        min_overlap: float = 0.0,
        max_overlap: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert 1 <= max_length
        assert 0 <= min_overlap < max_overlap

        self.num_samples = {"train": num_train, "val": num_val, "test": num_test}
        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap

        self.padding_index = 10
        self.blank_index = 11
        self.single_digit_mnist = SingleDigitMNIST()
        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=(-0.05, 0.05),
                        scale=(0.7, 1.1),
                        shear=(-30, 30),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    ),
                    transforms.ToTensor(),
                ]
            ),
            "val/test": transforms.ToTensor(),
        }

    @property
    def dataset_dirname(self) -> Path:
        """Directory to the dataset."""
        return self.data_dirname() / "multi_digit_mnist" / "processed"

    @property
    def dataset_filename(self) -> Path:
        """Filename of the dataset created by prepare_data."""
        return (
            self.dataset_dirname
            / f"ml_{self.max_length}_o{self.min_overlap:.2f}_{self.max_overlap:.2f}_"
            f"ntr{self.num_samples['train']}_nv{self.num_samples['val']}_"
            f"nte{self.num_samples['test']}.h5"
        )

    def prepare_data(self) -> None:
        """Create a synthetic dataset."""
        if self.dataset_filename.is_file():
            return
        self.single_digit_mnist.prepare_data()
        self.single_digit_mnist.setup()
        self.dataset_dirname.mkdir(parents=True, exist_ok=True)

        counter = 0
        for split in ("train", "val", "test"):
            print(f"Preparing {split} dataset...")
            image_generator = DatasetGenerator(
                self.single_digit_mnist.dataset[split],
                max_length=self.max_length,
                min_overlap=self.min_overlap,
                max_overlap=self.max_overlap,
                padding_index=self.padding_index,
                dot_index=12,
                dot_image_directory="/Users/peiyandong/Documents/code/ai/train-data/hwdb/single/1.0/raw/train/11776"
            )
            images, labels = image_generator.generate(self.num_samples[split])

            # 创建保存图像的目录
            img_dir = "/Users/peiyandong/Documents/code/ai/train-data/data-digit/img"
            os.makedirs(img_dir, exist_ok=True)

            # 确保标签文件的目录存在
            label_dir = "/Users/peiyandong/Documents/code/ai/train-data/data-digit/label"
            os.makedirs(label_dir, exist_ok=True)
            label_file = f"{label_dir}/digit_{split}.txt"

            with open(label_file, 'w') as f:
                for i, (image, label) in enumerate(zip(images, labels)):
                    # 将图像保存为PNG文件
                    img_file = f"{img_dir}/{counter}.png"
                    Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(img_file)

                    # 将标签转换为字符串
                    label_str = ''.join(['.' if digit.item() == 12 else str(
                        digit.item()) for digit in label if digit.item() != self.padding_index])

                    # 写入标签文件
                    f.write(f"{counter}.png {label_str}\n")
                    counter += 1

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            with h5py.File(self.dataset_filename, "r") as f:
                X_train = [Image.fromarray(_) for _ in f["X_train"][:]]
                y_train = torch.IntTensor(f["y_train"])
                X_val = [Image.fromarray(_) for _ in f["X_val"][:]]
                y_val = torch.IntTensor(f["y_val"])
            self.dataset["train"] = BaseDataset(
                X_train, y_train, transform=self.transform["train"]
            )
            self.dataset["val"] = BaseDataset(
                X_val, y_val, transform=self.transform["val/test"]
            )

        if stage in ("test", None):
            with h5py.File(self.dataset_filename, "r") as f:
                X_test = f["X_test"][:]
                y_test = torch.IntTensor(f["y_test"][:])
            self.dataset["test"] = BaseDataset(
                X_test, y_test, transform=self.transform["val/test"]
            )
