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
    datamodule = MultiDigitMNIST(**cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    cfg.lit_model.padding_index = datamodule.padding_index
    cfg.lit_model.blank_index = datamodule.blank_index
    lit_model = CTCLitModel(**cfg.lit_model)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))


    checkpoint_path = "/kaggle/input/hwdigitrectrain/lit_model.ckpt"
    if Path(checkpoint_path).is_file():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer = Trainer(resume_from_checkpoint=checkpoint_path, **cfg.trainer, callbacks=callbacks)
    else:
        print("No checkpoint found. Starting from scratch.")
        trainer = Trainer(**cfg.trainer, callbacks=callbacks)
    
    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()