import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional
from torchvision import transforms
import lightning as L
from torch.utils.data import DataLoader, random_split
import logging

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    Custom Dataset for Image Loading.

    This dataset class is designed for loading images from specified directories. 
    It supports various image formats (jpg, png, jpeg) and allows for optional 
    image transformations.

    Args:
        img_dirs (list, str): List of directory paths or a single directory path 
                             containing the images. Defaults to ["path/to/dir"].
        transform (torchvision.transforms.Compose, optional):  A torchvision transforms pipeline 
                                                               to be applied to each image. 
                                                               Defaults to None.

    Example:
        >>> dataset = CustomDataset(img_dirs=['train_images'], transform=transforms.ToTensor())
        >>> image = dataset[0]
    """
    def __init__(self, img_dirs: list = ["path/to/dir"], transform: Optional[transforms.Compose] = None):
        logger.info("Initializing CustomDataset...")
        self.img_dirs = img_dirs
        self.transform = transform
    
        self.images = []
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]  

        for img_dir in img_dirs:
            for img_name in os.listdir(img_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')): 
                    self.images.append(os.path.join(img_dir, img_name))
        logger.info(f"Found {len(self.images)} images in directories: {img_dirs}")
        logger.info("CustomDataset initialized.")

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)
        
    def __getitem__(self, idx):
        """
        Retrieve an image from the dataset at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            PIL.Image.Image: The image at the given index, transformed if a transform is provided.
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') 

        if self.transform:
            image = self.transform(image) 
            
        return image
    


class CustomDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Custom Image Datasets.

    This DataModule streamlines the process of loading and preparing custom image datasets 
    for training and validation in PyTorch Lightning. It supports loading data from 
    specified training and optional validation directories, and handles data splitting 
    if a separate validation directory is not provided.

    Args:
        train_data_dir (str): Path to the directory containing training images.
        transform (torchvision.transforms.Compose):  A torchvision transforms pipeline 
                                                     to be applied to the images.
        val_data_dir (str, optional): Path to the directory containing validation images. 
                                       If None, a validation set is created by splitting the 
                                       training data. Defaults to None.
        batch_size (int, optional): Number of samples per batch in the DataLoader. Defaults to 32.
        num_workers (int, optional): Number of CPU workers to use for data loading. Defaults to 4.
        train_val_split (float, optional):  Fraction of the training set to be used for training 
                                            when `val_data_dir` is None. Should be between 0 and 1. 
                                            Defaults to 0.99.

    Example:
        >>> data_module = CustomDataModule(
        ...     train_data_dir='train_images',
        ...     val_data_dir='val_images',
        ...     transform=train_transforms,
        ...     batch_size=64
        ... )
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader()
        >>> val_loader = data_module.val_dataloader()
    """
    def __init__(
        self,
        train_data_dir: str,
        transform,
        val_data_dir: Optional[str] = None,  
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.99
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir  
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        """
        Setup the data module for training and validation.

        This method is called by PyTorch Lightning to prepare the dataloaders. 
        It initializes the training and validation datasets, handling the case where 
        a separate validation directory is provided or when the validation set needs to be 
        split from the training set.

        Args:
            stage (str, optional):  Stage of training (fit, validate, test, predict). 
                                    Defaults to None.
        """
        logger.info("Setting up data module...")
        logger.info(f"Train data directories: {self.train_data_dir}")
        logger.info(f"Validation data directories: {self.val_data_dir}")
        train_dataset = CustomDataset(img_dirs=self.train_data_dir, transform=self.transform) 
        
        if self.val_data_dir is None:  
            train_size = int(self.train_val_split * len(train_dataset))
            val_size = len(train_dataset) - train_size
            logger.info(f"Splitting dataset into train ({train_size}) and val ({val_size}) sets with ratio: {self.train_val_split}")

            self.train_dataset, self.val_dataset = random_split(
                train_dataset, 
                [train_size, val_size]
            )
        else:
            logger.info(f"Using provided validation data directory: {self.val_data_dir}")
            self.val_dataset = CustomDataset(img_dirs=self.val_data_dir, transform=self.transform)  
            self.train_dataset = train_dataset  
        logger.info("Data module setup finished.")

    def train_dataloader(self):
        """
        Create and return the training DataLoader.

        Returns:
            torch.utils.data.DataLoader: Training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True, 
        )

    def val_dataloader(self):
        """
        Create and return the validation DataLoader.

        Returns:
            torch.utils.data.DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False, 
        )
