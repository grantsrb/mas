import torch.nn as nn
import torch.nn.functional as F
import dl_utils.torch_modules as tmods
import torch

class VisionModule(tmods.CoreModule):
    """
    This is the base class for vision modules.
    """
    def __init__(self,
                 inpt_shape,
                 outp_size,
                 bnorm=True,
                 lnorm=False,
                 drop_p=0,
                 actv_fxn="ReLU",
                 depths=[12, 32, 48],
                 kernels=3,
                 strides=[4, 1, 1],
                 paddings=0,
                 groups=1,
                 *args, **kwargs):
        """
        Args: 
            inpt_shape: tuple or listlike (..., C, H, W)
                the shape of the input
            outp_size: int
                the size of the final output vector
            bnorm: bool
                if true, the model uses batch normalization
            lnorm: bool
                if true, the model uses layer normalization on the h
                and c recurrent vectors after the recurrent cell
            drop_p: float
                the probability of zeroing a neuron within the dense
                layers of the network.
            actv_fxn: str
                the name of the activation function for each layer
            depths: tuple of ints
                the depth of each layer of the conv net
            kernels: tuple of ints
                the kernel size of each layer of the conv net
            strides: tuple of ints
                the stride of each layer of the conv net
            paddings: tuple of ints
                the padding of each layer of the conv net
            groups: int or tuple of ints
                the number of convolutional groups at each layer of the
                fully connected
                ouput networks
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.outp_size = outp_size
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.drop_p = drop_p
        self.actv_fxn = actv_fxn
        self.depths = [self.inpt_shape[-3], *depths]
        self.kernels = kernels
        if isinstance(kernels, int):
            self.kernels = [kernels for i in range(len(depths))]
        self.strides = strides
        if isinstance(strides, int):
            self.strides = [strides for i in range(len(depths))]
        self.paddings = paddings
        if isinstance(paddings, int):
            self.paddings = [paddings for i in range(len(depths))]
        self.groups = groups
        if isinstance(groups, int):
            self.groups = [groups for i in range(len(depths))]

class RawVision(VisionModule):
    """
    This vision module feeds the visual input directly, without
    preprocessing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = [ *self.inpt_shape[-3:] ]
        self.flat_size = int(np.prod(self.inpt_shape[-3:]))
        self.features = tmods.NullOp()

    def step(self, x, *args, **kwargs):
        return x

    def forward(self, x, *args, **kwargs):
        return x.reshape(len(x), -1)

class CNN(VisionModule):
    """
    A simple convolutional network
        conv2d
        bnorm/lnorm
        relu
        dropout
        repeat xN
        linear
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape]
        groups = 1
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i],
                    self.depths[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                    groups=max(int(groups*(i>0)), 1)
                )
            )
            # RELU
            modules.append(globals()[self.actv_fxn]())
            # Batch Norm
            if self.bnorm:
                modules.append(nn.BatchNorm2d(self.depths[i+1]))
            # Track Activation Shape Change
            shape = update_shape(
                shape, 
                depth=self.depths[i+1],
                kernel=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )
            self.shapes.append(shape)
        self.features = nn.Sequential(*modules)

        self.flat_size = int(np.prod(shape))
        self.projection = nn.Linear(self.flat_size, self.outp_size)

    def forward(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            pred: torch Float Tensor (B, K)
        """
        fx = self.features(x)
        fx = fx.reshape(len(fx), -1)
        return self.projection(fx)


class ResBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 resizer=None,
                 bnorm=True,
                 lnorm=False,
                 leading_norm=True,
                 noise=0):
        super().__init__()
        modules = []
        if leading_norm:
            if bnorm:
                modules.append(nn.BatchNorm2d(inplanes))
            if lnorm:
                modules.append(nn.LayerNorm(inplanes))
            if noise:
                modules.append(tmods.GaussianNoise(noise))
        modules.append( nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
        ))
        if bnorm:
            modules.append(nn.BatchNorm2d(planes))
        if lnorm:
            modules.append(nn.LayerNorm(planes))
        if noise:
            modules.append(tmods.GaussianNoise(noise))
        modules.append(nn.GELU())
        modules.append(nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ))
        if bnorm:
            modules.append(nn.BatchNorm2d(planes))
        if lnorm:
            modules.append(nn.LayerNorm(planes))
        if noise:
            modules.append(tmods.GaussianNoise(noise))
        self.fxns = nn.Sequential(*modules)
        self.resizer = resizer
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.fxns(x)
        if self.resizer is not None:
            identity = self.resizer(x)
        out += identity
        out = torch.nn.functional.gelu(out)
        return out
    
def _make_layer(block,
                inplanes,
                planes,
                n_blocks,
                stride=1,
                bnorm=True,
                lnorm=False,
                leading_norm=True,
                noise=0):
    resizer = None
    if stride != 1 or inplanes != planes:
        modules = [nn.Conv2d(
            inplanes, planes, 1, stride, bias=False
        )]
        if bnorm: modules.append(nn.BatchNorm2d(planes))
        if lnorm: modules.append(nn.LayerNorm(planes))
        if noise: modules.append(tmods.GaussianNoise(noise))
        resizer = nn.Sequential(*modules)
    layers = []
    layers.append(block(
        inplanes=inplanes,
        planes=planes,
        stride=stride,
        resizer=resizer,
        bnorm=bnorm,
        lnorm=lnorm,
        leading_norm=leading_norm,
        noise=noise,
    ))
    inplanes = planes
    for _ in range(1, n_blocks):
        layers.append(block(
            inplanes,
            planes,
            stride=1,
            bnorm=bnorm,
            lnorm=lnorm,
            leading_norm=leading_norm,
            noise=noise
        ))
    return nn.Sequential(*layers)

class ResLikeCNN(VisionModule):
    def __init__(self,
                 layer_counts=[2,2,2],
                 leading_norm=True,
                 noise=0,
                 ksize0=7,
                 *args, **kwargs):
        """
        layer_counts: list of ints
            denotes the number of res blocks for each channel change
        ksize0: int
            the first kernel size
        """
        super().__init__(*args, **kwargs)

        depths = self.depths
        self.in_conv = [nn.Conv2d(
            depths[0],
            depths[1],
            kernel_size=ksize0,
            stride=2,
            padding=3,
            bias=False
        )]
        if self.bnorm:
            self.in_conv.append(nn.BatchNorm2d(depths[1]))
        if self.lnorm:
            self.in_conv.append(nn.LayerNorm(depths[1]))
        self.in_conv.append(nn.GELU())
        self.in_conv = nn.Sequential(*self.in_conv)

        self.blocks = nn.ModuleList([])
        depths.append(depths[-1])
        for i, n_layers in enumerate(layer_counts):
            self.blocks.append(_make_layer(
                ResBlock,
                depths[i+1],
                depths[i+2],
                n_layers,
                stride=2 if i!=0 else 1,
                noise=noise,
                bnorm=self.bnorm,
            ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat_size = self.depths[-1]

        self.projection = nn.Linear(self.flat_size, self.outp_size)

    def features(self, x):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.avgpool(x)

    def forward(self, x):
        fx = self.features(x)
        fx = fx.reshape(len(fx), -1)
        return self.projection(fx)

