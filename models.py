import torch.nn as nn
import torch

''' Split-Brain Code '''

N_HID = 1048


class SimpleAE(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimpleAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 12, 4, stride=2, padding=1),  # [batch, 12, 48, 48]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),                # [batch, 24, 24, 24]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 5),                                     # [batch, 48, 20, 20]
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, 5),                                     # [batch, 48, 16, 16]
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, self.out_channels, 5)                       # [batch, out, 12, 12]
        )

    def forward(self, x):
        encoded = self.encoder(x.view(x.shape[0], self.in_channels, 96, 96))
        return encoded


class AlexNetAE(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(AlexNetAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Our Input size: [batch, 3, 96, 96] (NYU-Dataset)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 96, kernel_size=10, stride=2, padding=1),    # [batch, 96, 45, 45]
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=1),           # [batch, 96, 23, 23]
            nn.Conv2d(96, 256, kernel_size=5, dilation=1, padding=2),               # [batch, 256, 23, 23]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=1),           # [batch, 256, 12, 12]
            nn.Conv2d(256, 384, kernel_size=3, dilation=1, padding=1),              # [batch, 384, 12, 12]
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, dilation=1, padding=1),              # [batch, 384, 12, 12]
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, dilation=1, padding=1),              # [batch, 256, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, dilation=1, padding=1),           # [batch, 256, 12, 12]
            nn.Conv2d(256, 4096, kernel_size=7, dilation=2, padding=6),             # [batch, 4096, 12, 12]
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1, dilation=1),                       # [batch, 4096, 12, 12]
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, self.out_channels, kernel_size=1, dilation=1),          # [batch, out, 12, 12]
        )

    def forward(self, x):
        encoded = self.encoder(x.view(x.shape[0], self.in_channels, 96, 96))
        return encoded


class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )


    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNetAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GoogLeNetAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 192, kernel_size=4, padding=2),     #[batch, 192, 97, 97]
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            Inception(192, 64, 96, 128, 16, 32, 32),                        #[batch, 256, 97, 97]
            Inception(256, 128, 128, 192, 32, 96, 64),                      #[batch, 480, 97, 97]
            nn.MaxPool2d(3, stride=2, padding=1),                           # ...
            Inception(480, 192,  96, 208, 16,  48,  64),
            Inception(512, 160, 112, 224, 24,  64,  64),
            Inception(512, 128, 128, 256, 24,  64,  64),
            Inception(512, 112, 144, 288, 32,  64,  64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(8, stride=2),                                      # [batch, 1024, 21, 21]
            nn.Conv2d(1024, self.out_channels, kernel_size=3, padding=2, stride=2) # [batch, out, 12, 12]  Added
        )

    def forward(self, x):
        encoded = self.encoder(x.view(x.shape[0], self.in_channels, 96, 96))
        return encoded

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

''' ResNet Component '''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

''' ResNet Component'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetAE(nn.Module):
    # ResNet 18: BasicBlock, [2, 2, 2, 2]
    # ResNet 50: Bottleneck, [3, 4, 6, 3]
    def __init__(self, in_channels, out_channels, block=BasicBlock, layers=[2,2,2,2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=1,
                                           dilate=replace_stride_with_dilation[0]),
            self._make_layer(block, 256, layers[2], stride=1, dilate=replace_stride_with_dilation[1]),
            self._make_layer(block, self.out_channels, layers[3], stride=1, dilate=replace_stride_with_dilation[2]),
            nn.AdaptiveAvgPool2d((12, 12)),                                                  # [batch, out, 12, 12]
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        encoded = self.encoder(x.view(x.shape[0], self.in_channels, 96, 96))
        return encoded

#########################################################################################################################
#########################################################################################################################
'''Split Brain Models for Pretraining and Finetuning'''


class SplitBrain(nn.Module):
    def __init__(self, encoder="alex", num_ch2=25, num_ch1=100):
        super(SplitBrain, self).__init__()
        if encoder == "alex":
            self.ch2_net = AlexNetAE(in_channels=2, out_channels=num_ch1)
            self.ch1_net = AlexNetAE(in_channels=1, out_channels=num_ch2**2)
        elif encoder == "resnet":
            self.ch2_net = ResNetAE(in_channels=2, out_channels=num_ch1)
            self.ch1_net = ResNetAE(in_channels=1, out_channels=num_ch2**2)
        elif encoder == "googl":
            self.ch2_net = GoogLeNetAE(in_channels=2, out_channels=num_ch1)
            self.ch1_net = GoogLeNetAE(in_channels=1, out_channels=num_ch2**2)
        elif encoder == "simple":
            self.ch2_net = SimpleAE(in_channels=2, out_channels=num_ch1)
            self.ch1_net = SimpleAE(in_channels=1, out_channels=num_ch2**2)
        print("Split Brain Parameters- AB Net: ", sum(p.numel() for p in self.ch2_net.parameters() if p.requires_grad))
        print("Split Brain Parameters- ch1 Net: ", sum(p.numel() for p in self.ch1_net.parameters() if p.requires_grad))

    def forward(self, x):
        ch2, ch1 = x
        ch2_hat = self.ch1_net(ch1)
        ch1_hat = self.ch2_net(ch2)
        return ch2_hat, ch1_hat

''' Train a classifier consisting of the AlexNet plus a resizing linear layer'''
class SBNetClassifier(nn.Module):
    def __init__(self, encoder="alex", classifier="mlp", num_ch2=25, num_ch1=100):
        super(SBNetClassifier, self).__init__()
        self.sp = SplitBrain(encoder=encoder, num_ch2=num_ch2, num_ch1=num_ch1)
        n_in = num_ch2**2+num_ch1
        if encoder == "alex":
            n_in *= 11**2
        elif encoder == "googl":
            n_in *= 7**2
        elif encoder == "simple":
            n_in *= 6**2
        if classifier == "mlp":
            self.classifier = MLPClassifier(n_in,1000)
        elif classifier == "shallow":
            self.classifier = ShallowClassifier(n_in,1000)

        print("Total Finetuning Params: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        ch2, ch1 = x
        encoded_ch1 = self.sp.ch2_net(ch2.view(ch2.shape[0], self.sp.ch2_net.in_channels, 96, 96))
        encoded_ch2 = self.sp.ch1_net(ch1.view(ch1.shape[0], self.sp.ch1_net.in_channels, 96, 96))
        full_emb = torch.cat((encoded_ch2, encoded_ch1), 1)
        linear = self.classifier(full_emb.view(full_emb.shape[0], -1))
        return linear


#########################################################################################################################
#########################################################################################################################
'''Classifiers'''

class MLPClassifier(nn.Module):

    def __init__(self, n_in, n_out):

        super(MLPClassifier, self).__init__()

        self.out_channels = n_out
        self.classifier = nn.Sequential(
            nn.Linear(n_in, N_HID),
            nn.BatchNorm1d(N_HID),
            nn.ReLU(inplace=True),
            nn.Linear(N_HID, self.out_channels),
        )
        print("Classifier Params: ", sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))


    def forward(self, x):
        classification = self.classifier(x)
        return classification

class ShallowClassifier(nn.Module):

    def __init__(self, n_in, n_out):

        super(ShallowClassifier, self).__init__()

        self.out_channels = n_out
        self.classifier = nn.Linear(n_in, self.out_channels)

        print("Classifier Params: ", sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))

    def forward(self, x):
        classification = self.classifier(x)
        return classification

