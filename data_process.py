from argparse import Namespace
from typing import List
from pathlib import Path

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from handwritten_multi_digit_number_recognition.data import MultiDigitMNIST
from handwritten_multi_digit_number_recognition.lit_models import CTCLitModel

@hydra.main(config_path="./", config_name="config")
def main(cfg: DictConfig):
    datamodule = MultiDigitMNIST(num_train=10000,num_test=1000,num_val=500,max_length=6)
    datamodule.prepare_data()

if __name__ == "__main__":
    main()



    