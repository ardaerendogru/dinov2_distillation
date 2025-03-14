import warnings
import sys
import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import argparse
sys.path.append('../')
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from models.backbones import DINOv2ViT
from models import ModelWrapper
from train.distillation_module import DistillationModule
from datasets.CustomDataset import CustomDataModule
from datasets.augmentations import DataAugmentationDINO
import wandb
from utils import get_logger
logger = get_logger()

os.environ["NCCL_P2P_DISABLE"] = "1"


class DistillationTrainer:
    """
    Orchestrates the knowledge distillation training pipeline.

    This class is responsible for setting up and managing the entire distillation process,
    from configuration loading and data preparation to model training and checkpointing.
    It integrates various components such as data augmentation, data loading, model creation,
    loss function setup, and the PyTorch Lightning Trainer to facilitate efficient and configurable
    knowledge distillation.

    Attributes:
        cfg (Dict[str, Any]):
            Configuration dictionary loaded from a YAML file, defining all aspects of the training,
            including model settings, data paths, hyperparameters, and logging configurations.
        transform (DataAugmentationDINO):
            Data augmentation pipeline based on DINOv2 augmentations, applied to the input images
            to enhance the diversity of the training data and improve model generalization.
        data_module (CustomDataModule):
            PyTorch Lightning DataModule responsible for handling data loading and preprocessing.
            It manages training and validation datasets, dataloaders, and data transformations.
        teacher (torch.nn.Module):
            Pre-trained teacher model (typically DINOv2 ViT) from which knowledge is distilled.
            This model is frozen during training and provides target feature representations.
        student (torch.nn.Module):
            Student model being trained to mimic the teacher's behavior. This model is typically smaller
            and more efficient than the teacher and is optimized during the distillation process.
        distillation_module (DistillationModule):
            Core PyTorch Lightning Module that encapsulates the distillation logic, including the student and
            teacher models, loss functions, and the definition of training and validation steps.
        trainer (L.Trainer):
            PyTorch Lightning Trainer instance that automates the training loop, manages hardware acceleration,
            logging, checkpointing, and other training utilities based on the provided configuration.
        checkpoint_path (Optional[str]):
            Path to a checkpoint file from which to resume training. If provided, the trainer will load
            the model and training state from this checkpoint and continue training from where it left off.
            Defaults to None, indicating training from scratch.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DistillationTrainer.

        Sets up the trainer by processing the configuration, creating data transformations,
        initializing data modules, building teacher and student models, setting up the distillation module,
        configuring the PyTorch Lightning Trainer, and determining the checkpoint path for resuming training.

        Args:
            config (Dict[str, Any]):
                Configuration dictionary loaded from a YAML file, containing all training parameters.
        """
        logger.info("Starting DistillationTrainer initialization...")
        self.cfg = self._handle_config(config)
        self.transform = self._create_transform()
        self.data_module = self._create_data_module()
        self.teacher, self.student = self._create_models()
        self.distillation_module = self._create_distillation_module()
        self.trainer = self._create_trainer()
        self.checkpoint_path = self.cfg.train.get('resume_from_checkpoint', None)
        logger.info("DistillationTrainer initialized.")

    def _handle_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes and validates the configuration dictionary, deriving necessary parameters.

        This method performs configuration processing, including setting teacher model dimensions,
        configuring loss function parameters based on the model names and configurations,
        and handling any model-specific configurations for student and teacher models.

        Args:
            config (Dict[str, Any]):
                Raw configuration dictionary loaded from YAML.

        Returns:
            Dict[str, Any]:
                Processed configuration dictionary with derived parameters and validated settings.
        """
        logger.info("Starting config handling...")
        teacher_dims = {
            'dinov2_vits14' : 384,
            'dinov2_vitb14' : 768,
            'dinov2_vitl14' : 1024,
            'dinov2_vitg14' : 1536
        }
        config.teacher.out_dim = teacher_dims[config.teacher.model_name]
        config.teacher.teacher_key = config.teacher.get('teacher_key', 'feature_map')
        config.teacher.n_patches = [(config.data_transform.global_crops_size[0]//14), (config.data_transform.global_crops_size[1]//14) ]

        for loss in config.loss.losses:
            if loss.type == 'scalekd':
                loss.kwargs.teacher_dims = config.teacher.out_dim
                loss.kwargs.teacher_dims = config.teacher.out_dim
                loss.kwargs.pos_dims = config.teacher.out_dim
                loss.kwargs.pos_hw = [int(config.teacher.n_patches[0]),int(config.teacher.n_patches[1])]
                loss.kwargs.query_hw = [int(config.teacher.n_patches[0]),int(config.teacher.n_patches[1])]

        logger.info("Config handled and updated.")
        return config

    def _create_transform(self) -> DataAugmentationDINO:
        """
        Creates the data augmentation pipeline using DINO-style augmentations.

        Utilizes the `DataAugmentationDINO` class to generate a transformation pipeline
        that includes global crops and other augmentations suitable for self-supervised and
        distillation training, enhancing the robustness and generalization of the student model.

        Returns:
            DataAugmentationDINO:
                Initialized data augmentation pipeline.
        """
        logger.info("Creating data transform...")
        transform = DataAugmentationDINO(
            global_crops_scale=tuple(self.cfg['data_transform']['global_crops_scale']),
            global_crops_size=tuple(self.cfg['data_transform']['global_crops_size']),
        )
        logger.info(f"Data transform created: {transform}")
        return transform

    def _create_data_module(self) -> CustomDataModule:
        """
        Creates the PyTorch Lightning DataModule for handling data loading.

        Initializes a `CustomDataModule` with configurations from `self.cfg['data_loader']`,
        including training and validation data directories, data transformation pipeline,
        batch size, and number of workers for data loading.

        Returns:
            CustomDataModule:
                Initialized data loading module.
        """
        logger.info("Creating data module...")
        data_module = CustomDataModule(
            train_data_dir=self.cfg['data_loader'].get('train_dir', ['/home/arda/data/train2017']),
            val_data_dir = self.cfg['data_loader'].get('val_dir', None),
            transform=self.transform,
            batch_size=self.cfg['data_loader']['batch_size'],
            num_workers=self.cfg['data_loader']['num_workers']
        )
        logger.info(f"Data module created: {data_module}")
        return data_module

    def _create_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Creates and initializes the teacher and student models.

        Instantiates the teacher model using `DINOv2ViT` and the student model using `ModelWrapper`.
        The configurations for both models are loaded from `self.cfg`, and the student model's
        feature channels are dynamically adjusted based on the configuration.

        Returns:
            Tuple[torch.nn.Module, torch.nn.Module]:
                A tuple containing the initialized teacher and student models.
        """
        logger.info("Creating teacher and student models...")
        teacher = DINOv2ViT(
            model_name=self.cfg['teacher']['model_name'],
        )
        student = ModelWrapper(
            model_name=self.cfg['student']['model_name'],
            n_patches=self.cfg.teacher.n_patches,
            target_feature=self.cfg['student']['student_keys'],
        )
        for loss in self.cfg.loss.losses:
            if loss.type == 'scalekd':
                loss.kwargs.student_dims = int(student.feature_channels[loss.kwargs.name.split('_')[1]] )

        logger.info(f"Teacher model created: {self.cfg['teacher']['model_name']}")
        logger.info(f"Student model created: {self.cfg['student']['model_name']}")
        return teacher, student

    def _create_distillation_module(self) -> DistillationModule:
        """
        Creates the DistillationModule, which encapsulates the distillation setup.

        Initializes the `DistillationModule` with the student model, teacher model, and the
        entire configuration dictionary. This module manages the distillation loss computation
        and the training/validation steps.

        Returns:
            DistillationModule:
                Initialized distillation module.
        """
        logger.info("Creating distillation module...")
        distillation_module = DistillationModule(
            student=self.student,
            teacher=self.teacher,
            cfg=self.cfg
        )
        logger.info(f"Distillation module created.")
        return distillation_module

    def _create_trainer(self) -> L.Trainer:
        """
        Configures and creates the PyTorch Lightning Trainer.

        Sets up the `L.Trainer` with configurations from `self.cfg['train']` and `self.cfg['checkpoints']`,
        including logging, checkpointing, hardware acceleration, and distributed training strategies.
        It also initializes TensorBoardLogger and WandbLogger for experiment tracking.

        Returns:
            L.Trainer:
                Configured PyTorch Lightning Trainer instance.
        """
        logger.info("Creating PyTorch Lightning Trainer...")
        experiment_dir = f"logs/{self.cfg.student.model_name}"

        wandb_config = OmegaConf.to_container(self.cfg, resolve=True)
        wandb.init(
            config=wandb_config, 
            project=self.cfg.wandb.project,
            name= f'{self.cfg.student.model_name}_{self.cfg.teacher.model_name}' ,
            tags=self.cfg.wandb.tags,
            notes=self.cfg.wandb.notes,
            sync_tensorboard=True
        )
        wandb.define_metric("global_step")

        tb_logger = TensorBoardLogger(experiment_dir, name="distillation",default_hp_metric=False )
        tb_logger.log_hyperparams(self.cfg)

        # Set up checkpoint callback to save in the same experiment directory
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            filename=f"{{epoch}}-{{{self.cfg.checkpoints.monitor}:.4f}}",
            monitor=self.cfg.checkpoints.monitor,
            mode=self.cfg.checkpoints.mode,
            save_top_k=self.cfg.checkpoints.save_top_k,
            save_last=True
        )

        trainer = L.Trainer(
            default_root_dir='/storage/disk0/arda/tmp',  # Set default root dir
            max_epochs=self.cfg['train']['max_epochs'],
            accelerator=self.cfg.train.accelerator,
            devices=self.cfg.train.devices,
            num_nodes=self.cfg.train.num_nodes,
            strategy=self.cfg.train.strategy,
            precision=self.cfg.get('precision', 16),
            callbacks=[checkpoint_callback],
            logger=tb_logger,
            num_sanity_val_steps=0,
            gradient_clip_val=1.0,  # Example: Clip gradients to a maximum norm of 1.0
            gradient_clip_algorithm="norm", # Optional: "norm" (default) or "value"
            accumulate_grad_batches=self.cfg.train.get('accumulate_grad_batches', 1)  # Add gradient accumulation
        )
        logger.info(f"Trainer created: {trainer}")
        return trainer

    def train(self):
        """
        Executes the distillation training process.

        Starts the training process using the configured PyTorch Lightning Trainer and DistillationModule.
        It handles both starting a fresh training run and resuming from a checkpoint if `self.checkpoint_path` is set.
        """
        logger.info("Starting training process...")
        if self.checkpoint_path:
            logger.info(f"Resuming training from checkpoint: {self.checkpoint_path}")
            self.trainer.fit(self.distillation_module, self.data_module, ckpt_path=self.checkpoint_path)
        else:
            logger.info("Starting training from scratch.")
            self.trainer.fit(self.distillation_module, self.data_module)
        logger.info("Training process finished.")


def setup_environment():
    """
    Configures the global training environment.

    This function sets up the environment by configuring warning filters to ignore specific
    warnings related to deprecated features and user warnings, and by setting the precision
    for matrix multiplications to 'high' to leverage Tensor Cores for improved performance.
    """
    # Configure warnings
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set precision for Tensor Cores
    torch.set_float32_matmul_precision('high')


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the training script.

    Defines and parses command-line arguments, specifically the path to the configuration YAML file.
    Returns an `argparse.Namespace` object containing the parsed arguments, which can be accessed
    by attribute name.

    Returns:
        argparse.Namespace:
            Parsed arguments, including the path to the configuration file.
    """
    parser = argparse.ArgumentParser(description='Training script for distillation')
    parser.add_argument(
        '--config',
        type=str,
        default='./config/config.yaml',
        help='Path to the config file'
    )
    return parser.parse_args()


def main():
    """
    Main function to execute the distillation training script.

    This function serves as the entry point for the training script. It performs the following steps:
    1. Parses command-line arguments to load the configuration file path.
    2. Sets up the global training environment (warning filters, precision settings).
    3. Loads the training configuration from the specified YAML file.
    4. Initializes the `DistillationTrainer` with the loaded configuration.
    5. Starts the training process by calling the `train` method of the trainer.
    """
    # Parse command line arguments
    args = parse_args()

    # Setup environment
    setup_environment()

    # Load configuration from YAML
    with open(args.config, "r") as f:
        config = OmegaConf.load(f)

    trainer = DistillationTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()