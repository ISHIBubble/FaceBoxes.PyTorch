import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block from MobileNetV2.
    
    Structure: narrow → wide → narrow
    1. Expansion: 1×1 conv to expand channels by expansion factor
    2. Depthwise: 3×3 depthwise separable conv
    3. Projection: 1×1 conv to project back (LINEAR - no activation!)
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for depthwise conv (1 or 2)
        expand_ratio: Expansion factor for hidden dimension
    """
    
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                # 1×1 pointwise conv to expand
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-5),
                nn.Hardswish(inplace=True),  # Hardswish for quantization friendliness
            ])
        
        # Depthwise phase
        layers.extend([
            # 3×3 depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-5),
            nn.Hardswish(inplace=True),
        ])
        
        # Projection phase (LINEAR - no activation!)
        layers.extend([
            # 1×1 pointwise conv to project
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            # NO ACTIVATION HERE - this is the "linear bottleneck"
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)  # Residual connection
        else:
            return self.conv(x)


class DSConv2d(nn.Module):
    """Depthwise Separable Convolution with Hardswish"""
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.hardswish(x, inplace=True)


class DSCRelu(nn.Module):
    """Depthwise Separable Convolution with CRelu"""
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DSCRelu, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = F.hardswish(x, inplace=True)
        return x


class InceptionWithInvertedResidual(nn.Module):
    """
    Modified Inception module using Inverted Residuals.
    Replaces standard convolutions with inverted residual blocks.
    """
    
    def __init__(self, in_channels=128, expand_ratio=4):
        super(InceptionWithInvertedResidual, self).__init__()
        
        # Branch 1: 1×1 path (use small expansion since it's already 1×1)
        self.branch1x1 = InvertedResidual(in_channels, 32, stride=1, expand_ratio=2)
        
        # Branch 2: pooling + 1×1 path
        self.branch1x1_2 = InvertedResidual(in_channels, 32, stride=1, expand_ratio=2)
        
        # Branch 3: 1×1 reduce + 3×3
        self.branch3x3 = nn.Sequential(
            InvertedResidual(in_channels, 24, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(24, 32, stride=1, expand_ratio=expand_ratio),
        )
        
        # Branch 4: 1×1 reduce + 3×3 + 3×3
        self.branch3x3_double = nn.Sequential(
            InvertedResidual(in_channels, 24, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(24, 32, stride=1, expand_ratio=expand_ratio),
            InvertedResidual(32, 32, stride=1, expand_ratio=expand_ratio),
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)
        
        branch3x3 = self.branch3x3(x)
        
        branch3x3_double = self.branch3x3_double(x)
        
        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_double]
        return torch.cat(outputs, 1)  # Output: 32+32+32+32 = 128 channels


class FaceBoxes(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        # Initial convolutions (keep DSCRelu for efficient early processing)
        self.conv1 = DSCRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = DSCRelu(48, 64, kernel_size=5, stride=2, padding=2)
        
        # Inception modules with inverted residuals
        self.inception1 = InceptionWithInvertedResidual(128, expand_ratio=3)
        self.inception2 = InceptionWithInvertedResidual(128, expand_ratio=3)
        self.inception3 = InceptionWithInvertedResidual(128, expand_ratio=3)

        # Transition layers using inverted residuals
        # conv3: 128 → 256 with stride 2
        self.conv3 = nn.Sequential(
            InvertedResidual(128, 128, stride=1, expand_ratio=4),
            InvertedResidual(128, 256, stride=2, expand_ratio=4),
        )

        # conv4: 256 → 256 with stride 2
        self.conv4 = nn.Sequential(
            InvertedResidual(256, 128, stride=1, expand_ratio=4),
            InvertedResidual(128, 256, stride=2, expand_ratio=4),
        )

        self.loc, self.conf = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        
        # Detection heads (keep lightweight)
        loc_layers += [self._make_detection_layer(128, 21 * 4)]
        conf_layers += [self._make_detection_layer(128, 21 * num_classes)]
        
        loc_layers += [self._make_detection_layer(256, 1 * 4)]
        conf_layers += [self._make_detection_layer(256, 1 * num_classes)]
        
        loc_layers += [self._make_detection_layer(256, 1 * 4)]
        conf_layers += [self._make_detection_layer(256, 1 * num_classes)]
        
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _make_detection_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, 
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        detection_sources = list()
        loc = list()
        conf = list()

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        detection_sources.append(x)

        x = self.conv3(x)
        detection_sources.append(x)

        x = self.conv4(x)
        detection_sources.append(x)

        for (x, l, c) in zip(detection_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (loc.view(loc.size(0), -1, 4),
                      self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes))

        return output