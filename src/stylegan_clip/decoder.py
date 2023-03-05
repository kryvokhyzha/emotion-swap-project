import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import src.stylegan_clip.training.networks
import src.stylegan_clip.dnnlib.util

from src.stylegan_clip.torch_utils import legacy, misc
from src.stylegan_clip.training.networks import Generator


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # self.stylegan = Generator(
        #     c_dim=0, z_dim=512, w_dim=512, img_resolution=(4, 4, 6), img_channels=3, mapping_kwargs={'num_layers': 8},
        #     synthesis_kwargs={'num_fp16_res': 4, 'conv_clamp': 256, 'channel_base': int(0.5 * 32768), 'channel_max': 512},
        # ).requires_grad_(False).eval()

        self.stylegan = Generator(
            c_dim=0, z_dim=512, w_dim=512, img_resolution=(4, 4, 6), img_channels=3, mapping_kwargs={'num_layers': 8},
            synthesis_kwargs={
                'num_fp16_res': 0, 'conv_clamp': None,
                'channel_base': int(0.5 * 32768),
                'channel_max': 512
            },
        ).requires_grad_(False).eval()

        resume_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/' \
            'pretrained/transfer-learning-source-nets/' \
            'ffhq-res256-mirror-paper256-noaug.pkl'

        with src.stylegan_clip.dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
            misc.copy_params_and_buffers(resume_data['G_ema'], self.stylegan, require_all=True)

        with open('stylegan_clip/checkpoints/segmentation.pkl', 'rb') as f:
            model = pickle.load(f)

        latent_average = self.stylegan.mapping.w_avg.clone()
        latent_average = latent_average.view(1, 1, 512)

        self.register_buffer('latent_average', latent_average)
        self.register_buffer('weight', torch.from_numpy(model['w']).view(4, 256, 1, 1))
        self.register_buffer('bias', torch.from_numpy(model['b']))

    def forward(self, latents):
        """
        Arguments:
            latents: a float tensor with shape [b, 14, 512].
        Returns:
            result: a float tensor with shape [b, 3, h, w].
            is_skin: a float tensor with shape [b, 1, h, w].
        """

        result, features = self.stylegan.synthesis(latents)
        # they have shape [b, 3, h, w] and [b, 256, h / 4, w / 4]

        mask = F.conv2d(features[1].float(), self.weight, self.bias).softmax(1)
        is_skin = F.interpolate(mask, 256, mode='bilinear', align_corners=False)[:, 1:2]

        return result, is_skin, features
