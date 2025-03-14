import torch.nn as nn

import torch.nn.functional as F
import torch

import warnings
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])



class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)
        self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(
                    self.running_mean
                )
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(
            self.num_features, self.eps
        )

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    @classmethod
    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
        """
        Convert all FrozenBatchNorm2d to BatchNorm2d

        Args:
            module (torch.nn.Module):

        Returns:
            If module is FrozenBatchNorm2d, returns a new module.
            Otherwise, in-place convert module and return it.

        This is needed for quantization:
            https://fb.workplace.com/groups/1043663463248667/permalink/1296330057982005/
        """

        res = module
        if isinstance(module, FrozenBatchNorm2d):
            res = torch.nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def check_if_dynamo_compiling():
    if TORCH_VERSION >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False
    

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        act (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').

    Returns:
        Tensor: The output tensor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        dilation=1,
        norm="BN",
        activation=None,
        **kwargs,
    ):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], f"stride must in [1, 2]. " f"But received {stride}."
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    bias="",
                    norm=get_norm(norm, hidden_dim),
                    activation=activation,
                    **kwargs,
                )
            )
        layers.extend(
            [
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim,
                    bias="",
                    norm=get_norm(norm, hidden_dim),
                    activation=activation,
                    **kwargs,
                ),
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias="",
                    norm=get_norm(norm, out_channels),
                    activation=activation,
                    **kwargs,
                ),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class D2MobileNetV2(nn.Module):
    """MobileNetV2 backbone.

    This backbone is the implementation of
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        strides (Sequence[int], optional): Strides of the first block of each
            layer. If not specified, default config in ``arch_setting`` will
            be used.
        dilations (Sequence[int]): Dilation of each layer.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [
        [1, 16, 1],
        [6, 24, 2],
        [6, 32, 3],
        [6, 64, 4],
        [6, 96, 3],
        [6, 160, 3],
        [6, 320, 1],
    ]

    def __init__(
        self,
        widen_factor=1.0,
        strides=(1, 2, 2, 2, 1, 2, 1),
        dilations=(1, 1, 1, 1, 1, 1, 1),
        frozen_stages=-1,
        norm="BN",
        out_features=("res2", "res3", "res4", "res5"),
    ):
        super().__init__()
        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)

        if frozen_stages not in range(-1, 7):
            raise ValueError(
                "frozen_stages must be in range(-1, 7). "
                f"But received {frozen_stages}"
            )
        self.frozen_stages = frozen_stages
        self.norm = norm
        self.act = nn.functional.relu6

        self.in_channels = int(32 * widen_factor)

        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self._out_features = out_features

        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias="",
            norm=get_norm(norm, self.in_channels),
            activation=self.act,
        )

        self.layers = []
        self.layer_to_res = {
            "layer2": "res2",
            "layer3": "res3",
            "layer5": "res4",
            "layer7": "res5",
        }
        tot_stride = 1

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            tot_stride = tot_stride * stride
            dilation = self.dilations[i]
            out_channels = int(channel * widen_factor)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio,
            )
            layer_name = f"layer{i + 1}"
            if layer_name in self.layer_to_res:
                res_block = self.layer_to_res[layer_name]
                self._out_feature_strides[res_block] = tot_stride
                self._out_feature_channels[res_block] = out_channels
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride, dilation, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                    norm=self.norm,
                    activation=self.act,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        outs = {}
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if layer_name in self.layer_to_res:
                res_block = self.layer_to_res[layer_name]
                if res_block in self._out_features:
                    outs[res_block] = x

        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
