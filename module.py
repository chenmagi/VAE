from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from vae import VanillaVAE
import torchvision.utils as vutils
import os
from torch import optim


class VAEModule(pl.LightningModule):
    def __init__(self,model, params: dict):
        super().__init__()
        self.model = model
        self.params = params
        self.curr_device = None
        
    
    def forward(self, input:torch.tensor, **kwargs):
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        self.curr_device = data.device
        results = self.forward(data, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,   
                                              batch_idx = batch_idx)  
        self.log_dict({key: val.item() for key, val in train_loss.items()}, prog_bar=True, sync_dist=True)
        
        return train_loss['loss'] 
    
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        self.curr_device = data.device
        
        results = self.forward(data, labels = labels)
        val_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,   
                                              batch_idx = batch_idx)
        
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        img_path= os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png")
        vutils.save_image(recons.data,img_path,normalize=True,nrow=12)
        
        

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass    
         
    def configure_optimizers(self):
        optims=[]
        scheds=[]
        optimizer = torch.optim.Adam(self.model.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)
                return optims, scheds
        except:
            return optims