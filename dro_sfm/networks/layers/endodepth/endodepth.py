import torch
import torch.nn as nn
from typing import Dict, List
import torch.nn.functional as F
from dro_sfm.networks.layers.endodepth.resnet_encoder import ResnetEncoder
from dro_sfm.networks.layers.endodepth.layers import convbn
from collections import OrderedDict

class SpatialAttention(nn.Module):
    """
    Ozyoruk, K. B. et al. EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos. Med. Image Anal. 71, (2021).
    """

    def __init__(self, input_plane: int = 32, kernel_size: int = 3):

        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(input_plane, 4, kernel_size, padding=padding, bias=False)
        self.maxpool = nn.MaxPool2d(4, stride=4)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(4, 4, kernel_size, padding=padding, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv3 = nn.Conv2d(4, input_plane, kernel_size, padding=padding, bias=False)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, input: torch.Tensor):
        # input layer
        x = self.conv1(input)
        x = self.maxpool(x)
        # attention score
        s = torch.matmul(x, x.transpose(-2, -1).contiguous())
        s = self.relu(s)
        s = self.conv2(s)
        # attention weight
        w = self.softmax(s)
        z = torch.matmul(w, x)
        # output layer
        z = self.conv3(z)
        z = self.upsample(z)
        # residual output
        return z + input


class ResnetAttentionEncoder(ResnetEncoder):

    def __init__(self, out_layer=None, *args, **kwargs):
        super(ResnetAttentionEncoder, self).__init__(*args, **kwargs)
        if hasattr(kwargs, "kernel_size"):
            kernel = kwargs['kernel_size']
        else:
            kernel = 3
        self.SAB = SpatialAttention(self.num_ch_enc[1], kernel_size=kernel)
        self.out = nn.Identity() if out_layer is None else out_layer

    def forward(self, input_image):
        features = []
        x = input_image
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num = len(x)
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        features.append(self.SAB(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[0])))
        features.append(self.encoder.layer2(features[1]))
        features.append(self.encoder.layer3(features[2]))
        features.append(self.out(self.encoder.layer4(features[3])))

        if is_list:
            features[-1] = torch.split(features[-1], [batch_dim] * num, dim=0)
        return features



class DepthDecoder(nn.Module):
    def __init__(self,num_ch_enc,scales=[0, 1, 2, 3],
                 num_output_channels=1,
                 use_skips=True):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.upsample_mode = 'nearest'

        if isinstance(use_skips, bool):
            self.use_skips = [0, 1, 2, 3, 4]
        else:
            self.use_skips = use_skips

        self.num_ch_enc = num_ch_enc
        num_ch_dec = (16, 32, 64, 128, 256)
        self.scales = scales
        self.decoder = nn.ModuleList()
        for l in range(4, -1, -1):
            layer = nn.ModuleDict()
            # upconv_0
            num_ch_in = num_ch_enc[-1] if l == 4 else num_ch_dec[l + 1]
            num_ch_out = num_ch_dec[l]
            layer["upconv_0"] = convbn(int(num_ch_in), num_ch_out, kernel=3)

            # upconv_1
            if l in self.use_skips:
                num_ch_in = num_ch_dec[l]
                if l > 0:
                    num_ch_in += num_ch_enc[l-1]
                num_ch_out = num_ch_dec[l]
                layer["upconv_1"] = convbn(int(num_ch_in), num_ch_out, kernel=3)
            else:
                layer["upconv_1"] = nn.Identity()

            # Disparity conv
            if l in scales:
                layer["dispconv"] = nn.Conv2d(
                    num_ch_dec[l], self.num_output_channels, kernel_size=3,
                    stride=1, padding=1, bias=False, padding_mode='reflect')
            else:
                layer["dispconv"] = nn.Identity()
            # Add layer to decoder
            self.decoder.append(layer)


    def forward(self, input_features):
        outputs : torch.tensor = None
        feature_n = len(input_features)
        x = input_features[feature_n - 1]
        for j, layer in enumerate(self.decoder):
            i: int = feature_n-1 - j
            x = layer["upconv_0"](x)
            x = F.interpolate(x, scale_factor=2.0, mode=self.upsample_mode)  # upsample
            if i-1 >=0 and i-1 < feature_n:
                x = torch.cat((x, input_features[i - 1]), 1)
            x = layer["upconv_1"](x)
            if i in self.scales:
                outputs = torch.sigmoid(layer["dispconv"](x))
        return outputs

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1]*2, 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        pose = 0.01 * out.view(-1, 6)

        return pose