import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Sequence, Union, Any, Callable
import torchvision


class VAEDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_path:str,
        train_batch_size:int=8,
        val_batch_size:int=8,
        patch_size:Union[int, Sequence[int]]=(256, 256),
        num_workers:int=0,
        pin_memory:bool=False,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage:Optional[str]=None) -> None:
        
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.patch_size,self.patch_size)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.1307,),(0.3081,))
        ])
        
        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.patch_size,self.patch_size)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.1307,),(0.3081,))
        ])
        
        self.train_dataset = torchvision.datasets.CelebA(
            root=self.data_path,
            #train=True,
            transform=train_transforms,
            download=True
        )
        
        self.val_dataset = torchvision.datasets.CelebA(
            root=self.data_path,
            #train=False,
            transform=val_transforms,
            download=True
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,  
            pin_memory=self.pin_memory
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  
            pin_memory=self.pin_memory
        )    
    
    def test_dataloader(self) -> Union[DataLoader,List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,  
            pin_memory=self.pin_memory
        )    
        