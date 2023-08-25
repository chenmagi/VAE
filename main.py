from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml
import argparse
from pathlib import Path
import os

import pytorch_lightning as pl
from module import VAEModule
from vae import VanillaVAE
from dataset import VAEDataset


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to the config file',
                    default='vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'])

model = VanillaVAE(**config['model_params'])
pl_module = VAEModule(model, config['exp_params'])
data = VAEDataset(**config["data_params"], pin_memory=True)
data.setup()

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                     monitor="val_loss",
                                     save_last=True),
                 ],
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
try:
    if config['fit_params']['resume'] is not None:
        print('-->Resum from checkpoint {}'.format(config['fit_params']['resume']))
        runner.fit(pl_module, datamodule=data, ckpt_path=config['fit_params']['resume'])
except KeyError: 
        print('-->Start from scratch')
        runner.fit(pl_module, datamodule=data)
     
