import lightning as L
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.nn as nn
from losses import ScaleKD
from utils import get_logger
import numpy as np
logger = get_logger()

LOSS_REGISTRY = {
    'scalekd': ScaleKD,
}

class DistillationModule(L.LightningModule):
    """
    Core Lightning module for handling the distillation training process.

    This module orchestrates the training of a student model by distilling knowledge from a pre-trained teacher model.
    It encompasses model initialization, loss function configuration, optimization setup, and the training/validation loop.

    Attributes:
        cfg (dict):
            Configuration dictionary loaded from a YAML file, defining hyperparameters, model settings,
            loss configurations, and optimization parameters.
        student (nn.Module):
            The student model being trained to mimic the teacher's behavior. This model is typically smaller
            and more efficient than the teacher.
        teacher (nn.Module):
            The pre-trained teacher model from which knowledge is distilled. This model is frozen during training
            and serves as the source of rich feature representations.
        losses (nn.ModuleDict):
            A dictionary of loss functions used for distillation. Each loss function is registered as a module
            and is configurable via the `cfg.loss` section. Common losses include ScaleKD, MSE, etc.
        loss_weights (dict):
            A dictionary specifying the weights for each loss function defined in `losses`. These weights
            control the contribution of each loss to the total training objective.
        loss_mse (nn.Module):
            Mean Squared Error (MSE) loss module, potentially used as a component in certain distillation losses
            or for auxiliary tasks. Initialized as `nn.MSELoss(reduction='sum')`.
    """

    def __init__(
        self,
        student,
        teacher,
        cfg
    ):
        """
        Initializes the DistillationModule.

        Constructs the distillation training framework by setting up the student and teacher models,
        configuring loss functions, and preparing for the training process.

        Args:
            student (nn.Module): The student model instance.
            teacher (nn.Module): The teacher model instance.
            cfg (dict): The configuration dictionary containing training settings.
        """
        super().__init__()
        logger.info("Starting DistillationModule initialization...")
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self._initialize_models(student, teacher)
        self._initialize_loss()
        logger.info("DistillationModule initialized.")

    def _initialize_models(self, student, teacher):
        """
        Initializes and configures the student and teacher models.

        Sets up the student and teacher models within the module, registers them for Lightning management,
        freezes the teacher model to prevent its weights from being updated during training, and loads
        a student checkpoint if a path is specified in the configuration.

        Args:
            student (nn.Module): The student model to be initialized.
            teacher (nn.Module): The teacher model to be initialized.
        """
        logger.info("Initializing models...")
        self.student = student
        self.teacher = teacher
        self.register_module('student', self.student)
        self.register_module('teacher', self.teacher)

        
        self._freeze_teacher()
        
        
        if self.cfg.student.get('checkpoint_path', None):
            self._load_student_checkpoint(self.cfg.student.checkpoint_path)
        else:
            self._load_student_checkpoint(self.student.model.ckpt_dir)

        logger.info("Models initialized.")

    def _freeze_teacher(self):
        """
        Freezes the parameters of the teacher model.

        Sets the teacher model to evaluation mode (`eval()`) and disables gradient computation for all
        teacher model parameters (`requires_grad = False`). This ensures that the teacher model remains
        constant throughout the distillation process, acting as a fixed knowledge source.
        """
        logger.info("Freezing teacher model...")
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        logger.info("Teacher model frozen.")

        
    def _initialize_loss(self):
        """
        Initializes and registers the loss functions based on the configuration.

        Parses the loss configuration from `self.cfg.loss['losses']`, instantiates the specified loss functions
        using `LOSS_REGISTRY`, and registers them as a `nn.ModuleDict` under `self.losses`. Also, sets up
        `self.loss_weights` to store the weights associated with each loss function, as defined in the config.
        """
        logger.info("Initializing loss functions...")
        self.losses = nn.ModuleDict()  
        self.loss_weights = {}
        
        for loss_spec in self.cfg.loss['losses']:
            loss_type = loss_spec['type']
            weight = loss_spec['weight']
            kwargs = loss_spec['kwargs']
            
            
            name = kwargs.get('name', loss_type)
            
            
            loss_fn = LOSS_REGISTRY[loss_type](**kwargs)
            self.losses[name] = loss_fn  
            self.loss_weights[name] = weight
        self.register_module('losses', self.losses)
        logger.info(f"Loss functions initialized: {list(self.losses.keys())}")

    def _forward_specific_stage(self, feat, layer):
        """
        Processes features through specific blocks of the DinoV2 teacher model, optimized for ResNet student.

        This method is tailored for distillation from a DinoV2 teacher (Vision Transformer) to a ResNet student (CNN).
        It selectively forwards features through a subset of DinoV2 blocks that are semantically aligned with
        different stages of a ResNet architecture (res2, res3, res4, res5). This focuses on spatial feature
        transfer and avoids transformer-specific components that might not be beneficial for CNN students.

        Args:
            feat (torch.Tensor):
                Input feature tensor, expected to be in the format [B, Patches, embedding_dim].
            layer (str):
                Identifier for the ResNet layer stage ('res2', 'res3', 'res4', 'res5'). This determines
                which blocks of the DinoV2 teacher are used for feature processing, based on a predefined mapping.

        Returns:
            torch.Tensor:
                Processed feature tensor after passing through the selected DinoV2 blocks. The output shape
                will depend on the DinoV2 block configuration and the input feature shape.
        """
        
        
        layers = {
            'res2': 0.25,  
            'res3': 0.50,  
            'res4': 0.75   
        }

        
        n_total_blocks = len(self.teacher.model.blocks)
        start_block = int(n_total_blocks * layers[layer])
        end_block = int(n_total_blocks/4) - 1


        if layer == 'res4':
            end_block = n_total_blocks - 1
        for i in range(start_block, end_block):
            feat = self.teacher.model.blocks[i](feat)
        return feat

    def _compute_losses(self, features):
        """
        Computes the composite distillation loss from student and teacher features.

        Calculates the total distillation loss by iterating through configured loss functions, applying them to
        the extracted student and teacher features, and weighting their contributions according to `self.loss_weights`.
        This method supports multiple loss functions and aggregates them into a single training objective.

        Args:
            features (dict):
                A dictionary containing feature maps from the student and teacher models.
                Expected keys are 'student' and 'teacher', with each value being a dictionary of feature level outputs
                (e.g., {'res2': student_feat_res2, 'res3': student_feat_res3, ...}).

        Returns:
            dict:
                A dictionary containing individual loss values and the total aggregated loss.
                Keys in the dictionary correspond to loss names (e.g., 'scalekd_res4_total_loss', 'scalekd_res4_spatial_loss')
                and 'loss' for the total loss. These values are detached from the computation graph for logging purposes.
        """
        total_loss = 0
        loss_dict = {}

        
        spatial_query = None
        frequency_query = None

        scale_kd_losses = sorted([layer for layer in self.losses.keys()])

        for scale_kd_layer in scale_kd_losses:
            layer_name = scale_kd_layer.split('_')[1]

            if 'res5' in scale_kd_layer:
                weight = self.loss_weights[scale_kd_layer]
                loss_fn = self.losses[scale_kd_layer]
                loss = loss_fn(features['student'][layer_name], features['teacher'], 
                                            query_s=spatial_query, 
                                            query_f=frequency_query)
                loss_dict[f'{scale_kd_layer}_total_loss'] = loss['loss'] * weight
                loss_dict[f'{scale_kd_layer}_frequency_loss'] =  loss['frequency_loss'] * weight
                loss_dict[f'{scale_kd_layer}_spatial_loss'] =  loss['spatial_loss'] * weight
                loss_dict[f'{scale_kd_layer}_spatial_similarity'] = loss['spatial_similarity'] 
                loss_dict[f'{scale_kd_layer}_frequency_similarity'] = loss['frequency_similarity'] 
                total_loss += loss['loss'] * weight
                break


            B, C, H, W = features['student'][layer_name].shape
            loss_fn = self.losses[scale_kd_layer]
            weight = self.loss_weights[scale_kd_layer]
            feat_S_spat = loss_fn.project_feat_spat(features['student'][layer_name], query=spatial_query)
            feat_S_freq = loss_fn.project_feat_freq(features['student'][layer_name], query=frequency_query)
            feat_S_spat = self._forward_specific_stage(feat_S_spat, layer_name) 
            feat_S_freq = self._forward_specific_stage(feat_S_freq, layer_name)
            spatial_query = feat_S_spat
            frequency_query = feat_S_freq
            spatial_loss, spatial_similarity = loss_fn.get_spat_loss(feat_S_spat, features['teacher'])
            frequency_loss, frequency_similarity = loss_fn.get_spat_loss(feat_S_freq, features['teacher'])
            loss_dict[f'{scale_kd_layer}_total_loss'] = (spatial_loss + frequency_loss )* weight
            loss_dict[f'{scale_kd_layer}_frequency_loss'] =  frequency_loss * weight
            loss_dict[f'{scale_kd_layer}_spatial_loss'] =  spatial_loss * weight
            loss_dict[f'{scale_kd_layer}_spatial_similarity'] =  spatial_similarity
            loss_dict[f'{scale_kd_layer}_frequency_similarity'] = frequency_similarity           
            total_loss += (spatial_loss + frequency_loss )* weight
        
        loss_dict['loss'] = total_loss
        return loss_dict
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        This method is called for each batch during training. It orchestrates the feature extraction from both
        student and teacher models, computes the distillation loss, logs training metrics, and returns the total loss
        for optimization.

        Args:
            batch (torch.Tensor):
                Input batch of data from the training dataloader. The structure of the batch is defined by the
                dataloader and typically contains input images.
            batch_idx (int):
                The index of the current batch within the training epoch.

        Returns:
            torch.Tensor:
                The total computed loss for the current training step. This loss is used by the optimizer to update
                the student model's parameters.
        """
        
        
        features = self._extract_features(batch)

        
        losses = self._compute_losses(features)
        
        self._log_training_metrics(losses)

        return losses['loss']


    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        This method is called for each batch during validation. It mirrors the `training_step` in terms of feature
        extraction and loss computation but is used to evaluate the student model's performance on the validation set.
        Crucially, it operates in evaluation mode (no gradient computation or student parameter updates) and typically
        uses an EMA (Exponential Moving Average) version of the student model if configured.

        Args:
            batch (torch.Tensor):
                Input batch of data from the validation dataloader.
            batch_idx (int):
                The index of the current batch within the validation epoch.

        Returns:
            torch.Tensor:
                The total computed validation loss for the current validation step. This loss is used for monitoring
                model performance and potentially for early stopping or learning rate scheduling.
        """
        
        features = self._extract_features(batch)
        
        
        losses = self._compute_losses(features)
        
        
        self._log_validation_metrics(losses)

        
        return losses['loss']

    def _extract_features(self, batch):
        """
        Extracts feature maps from the student and teacher models for a given input batch.

        Passes the input batch through both the student and teacher models to obtain their respective feature maps.
        The teacher model's features are extracted in a no-gradient context (`torch.no_grad()`) as it is frozen.
        The specific feature levels to extract are determined by the model configurations in `self.cfg`.

        Args:
            batch (torch.Tensor):
                Input batch of data, typically image tensors.

        Returns:
            dict:
                A dictionary containing the extracted feature maps from both the student and teacher models.
                The dictionary has keys 'student' and 'teacher', each mapping to a dictionary of feature level
                outputs (e.g., {'res2': student_features_res2, 'res3': student_features_res3, ...}).
        """
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            teacher_features = teacher_output[self.cfg.teacher.teacher_key]

        student_output = self.student(batch)
        return {
            'student': student_output,
            'teacher': teacher_features
        }

    def _log_training_metrics(self, losses):
        """
        Logs training metrics to the Lightning logger.

        Iterates through the computed losses and logs each loss value under the 'train_' prefix. Also logs the
        current learning rate of the optimizer. Metrics are logged for each training step and aggregated across epochs.

        Args:
            losses (dict):
                Dictionary of computed loss values, typically returned by `self._compute_losses`.
            features (dict):
                Dictionary of extracted features (not directly used for logging in this method, but included
                for potential future extensions).
        """
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, sync_dist=True)
        
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, sync_dist=True, prog_bar=True)
        

    def _log_validation_metrics(self, losses):
        """
        Logs validation metrics to the Lightning logger.

        Similar to `_log_training_metrics`, but logs metrics with a 'val_' prefix, indicating validation metrics.
        This method is called at the end of each validation step to record the performance of the student model
        on the validation dataset.

        Args:
            losses (dict):
                Dictionary of computed loss values from the validation step.
            features (dict):
                Dictionary of extracted features (not directly used for logging in this method, but included
                for potential future extensions).
        """
        
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, sync_dist=True)


    def _load_student_checkpoint(self, checkpoint_path):
        """
        Loads a checkpoint into the student model.
        
        Args:
            checkpoint_path (str): Path to checkpoint file (.pkl or .pth).
            
        Raises:
            KeyError: If model name is not recognized.
            ValueError: If file format is not supported.
        """
        logger.info(f"Loading student checkpoint from: {checkpoint_path}...")
        
        if checkpoint_path.endswith('.pkl'):
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                checkpoint = checkpoint['model']
        elif checkpoint_path.endswith('.pth'):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        
        for k, v in checkpoint.items():
            if isinstance(v, np.ndarray):
                checkpoint[k] = torch.from_numpy(v)
        student_type = self.cfg.student.model_name.split('_')[0].lower()

        if student_type == 'mobilenet':
            version = self.cfg.student.model_name.split('_')[1].lower()
            student_type = student_type + '_' + version

        if 'stdc' == student_type:
            checkpoint = {f"model.model.{k.replace('cp.backbone.', '')}": v for k, v in checkpoint.items()}
            result = self.student.load_state_dict(checkpoint, strict=False)
        elif 'mit' == student_type or 'darknet' == student_type or 'mobilenet_v2' == student_type or student_type == 'presnet':
            checkpoint = {f"model.model.{k.replace('backbone.', '')}": v for k, v in checkpoint.items()}
            result = self.student.load_state_dict(checkpoint, strict=False)
        elif student_type == 'mobilenet_v3':
            checkpoint = {f"model.model.{k.replace('backbone.', '')}": v for k, v in checkpoint.items() if 'classifier' not in k}
            result = self.student.load_state_dict(checkpoint, strict=False)
        else:
            checkpoint = {f"model.model.{k}": v for k, v in checkpoint.items()}
            result = self.student.load_state_dict(checkpoint, strict=False)


        logger.info(f"Missing Keys:")
        for key in result.missing_keys:
            logger.info(f"  {key}")
        logger.info(f"Unexpected Keys:")
        for key in result.unexpected_keys:
            logger.info(f"  {key}")
        matched_keys = [key for key in self.student.state_dict().keys() if key in checkpoint.keys()]
        logger.info(f"Matched Keys:")
        for key in matched_keys:
            logger.info(f"  {key}")

        logger.info(f"Student checkpoint loaded from: {checkpoint_path}. Result: Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Sets up the optimizer (e.g., AdamW, SGD) as specified in `self.cfg['optimizer']` and, optionally, a
        learning rate scheduler as defined in `self.cfg['optimizer']['scheduler']`. Parameters to be optimized
        include those from the student model and any trainable parameters from the loss functions.

        Returns:
            Union[torch.optim.Optimizer, dict]:
                Configured optimizer or a dictionary containing the optimizer and scheduler configurations.
                If a scheduler is configured, returns a dictionary with keys 'optimizer' and 'lr_scheduler'.
                Otherwise, returns just the optimizer. The scheduler configuration includes settings for monitoring
                a metric, update interval, and frequency, as specified in the config.
        """
        logger.info("Configuring optimizers...")
        
        param_groups = []
        
        
        student_params = list(self.student.parameters())
        param_groups.append({
            'params': student_params,
            'name': 'student'
        })
        
        
        for loss_name, loss_module in self.losses.items():
            loss_params = list(loss_module.parameters())
            if loss_params:  
                param_groups.append({
                    'params': loss_params,
                    'name': f'loss_{loss_name}'
                })
                

        optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])(
            param_groups,
            **self.cfg['optimizer'].get('kwargs', {})
        )
        logger.info(f"Optimizer configured: {self.cfg['optimizer']['type']}")
        
        
        if 'scheduler' in self.cfg['optimizer']:
            scheduler = getattr(torch.optim.lr_scheduler, 
                            self.cfg['optimizer']['scheduler']['type'])(
                optimizer,
                **self.cfg['optimizer']['scheduler'].get('kwargs', {})
            )
            logger.info(f"Scheduler configured: {self.cfg['optimizer']['scheduler']['type']}")

            return {
                "optimizer": optimizer,
                
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg['optimizer']['scheduler'].get('monitor', 'val_loss'),
                    "interval": self.cfg['optimizer']['scheduler'].get('interval', 'epoch'),
                    "frequency": self.cfg['optimizer']['scheduler'].get('frequency', 1)
                }
            }
        
        return optimizer

