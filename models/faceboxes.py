import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = DSConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = DSConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = DSConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = DSConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = DSConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = DSConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = DSConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)

        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)

        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        return torch.cat(outputs, 1)


class DSCRelu(nn.Module):

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
        x = F.relu(x, inplace=True)
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    Computes channel-wise attention weights using both average and max pooling,
    followed by a shared MLP (implemented as 1x1 convolutions for efficiency).
    
    Uses 1D convolution variant to reduce channel information loss as recommended
    for multi-scale feature processing.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP using 1x1 convolutions (more efficient than FC layers)
        # Reduction ratio controls the bottleneck size
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global average pooling path
        avg_out = self.shared_mlp(self.avg_pool(x))
        # Global max pooling path
        max_out = self.shared_mlp(self.max_pool(x))
        # Combine and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    Computes spatial attention weights by concatenating channel-wise average
    and max pooled features, followed by a 7x7 convolution.
    
    The 7x7 kernel size is recommended in the original CBAM paper for
    capturing larger spatial context.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=2,  # avg + max pooled features
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average pooling: (B, C, H, W) -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max pooling: (B, C, H, W) -> (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension: (B, 2, H, W)
        spatial_descriptor = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid
        spatial_attention = self.sigmoid(self.conv(spatial_descriptor))
        return x * spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies:
    1. Channel Attention - "what" to focus on
    2. Spatial Attention - "where" to focus
    
    This sequential refinement is particularly effective for face detection
    as faces have strong spatial structure and certain channels encode
    critical facial features (edges, textures, symmetry).
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention MLP (default: 16)
        spatial_kernel_size: Kernel size for spatial attention conv (default: 7)
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x):
        # Sequential application: channel first, then spatial
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FaceBoxes(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        # RDCL (Rapidly Digested Convolutional Layers)
        self.conv1 = DSCRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = DSCRelu(48, 64, kernel_size=5, stride=2, padding=2)

        # MSCL (Multiple Scale Convolutional Layers) - Inception modules
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()
        
        # CBAM after Inception blocks (128 channels)
        # Using reduction_ratio=8 for the smaller channel count to maintain expressiveness
        self.cbam_inception = CBAM(in_channels=128, reduction_ratio=8, spatial_kernel_size=7)

        # Additional conv layers for multi-scale detection
        self.conv3_1 = DSConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = DSConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # CBAM after conv3_2 (256 channels)
        self.cbam_conv3 = CBAM(in_channels=256, reduction_ratio=16, spatial_kernel_size=7)

        self.conv4_1 = DSConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = DSConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # CBAM after conv4_2 (256 channels)
        self.cbam_conv4 = CBAM(in_channels=256, reduction_ratio=16, spatial_kernel_size=7)
        
        # Detection heads
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
        
        # First detection layer (128 channels → 21 anchors)
        loc_layers += [self._make_detection_layer(128, 21 * 4)]
        conf_layers += [self._make_detection_layer(128, 21 * num_classes)]
        
        # Second detection layer (256 channels → 1 anchor)
        loc_layers += [self._make_detection_layer(256, 1 * 4)]
        conf_layers += [self._make_detection_layer(256, 1 * num_classes)]
        
        # Third detection layer (256 channels → 1 anchor)
        loc_layers += [self._make_detection_layer(256, 1 * 4)]
        conf_layers += [self._make_detection_layer(256, 1 * num_classes)]
        
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def _make_detection_layer(self, in_channels, out_channels):
        """
        Creates a depthwise separable convolution for detection heads.
        Uses 3x3 depthwise followed by 1x1 pointwise to maintain receptive field.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, 
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        detection_sources = list()
        loc = list()
        conf = list()

        # RDCL
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # MSCL - Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        # Apply CBAM after Inception blocks (first detection scale)
        x = self.cbam_inception(x)
        detection_sources.append(x)

        # Second detection scale
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.cbam_conv3(x)  # Apply CBAM
        detection_sources.append(x)

        # Third detection scale
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.cbam_conv4(x)  # Apply CBAM
        detection_sources.append(x)

        # Detection heads
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