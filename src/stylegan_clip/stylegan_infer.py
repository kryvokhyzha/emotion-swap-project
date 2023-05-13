from stylegan_clip.mapper import Mapper
from stylegan_clip.decoder import Decoder
from kornia.filters import GaussianBlur2d
import os
import torch
import copy
from PIL import Image
from torchvision.utils import make_grid


class Model:

    def __init__(self, path_to_checkpoint, device):
        self.device = device

        self.decoder = Decoder().to(device).requires_grad_(False).eval()

        self.stylegan = copy.deepcopy(self.decoder.stylegan)
        self.stylegan.requires_grad_(False).eval()

        self.mapper = Mapper(depth=256)
        self.mapper.to(device).requires_grad_(False).eval()

        self.blur = GaussianBlur2d(kernel_size=(11, 11), sigma=(5, 5))

        self.load_checkpoint(path_to_checkpoint, device)

    def inference(self, batch_size, noise=None):
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(batch_size, 512, device=self.device)
            latents = self.stylegan.mapping(z=noise, c=None, truncation_psi=0.75)
            identity, _ = self.stylegan.synthesis(latents)
            identity = identity.clamp(-1.0, 1.0).float()
            deltas = self.mapper(latents)
            result, _, _ = self.decoder(latents + deltas)
        return noise, identity, result.clamp(-1.0, 1.0).float()

    def load_checkpoint(self, path_to_checkpoint, device):
        self.decoder.load_state_dict(torch.load(os.path.join(path_to_checkpoint, 'decoder.pth'), map_location=device))
        self.mapper.load_state_dict(torch.load(os.path.join(path_to_checkpoint, 'mapper.pth'), map_location=device))


def main(path_to_checkpoint, batch_size, device='cuda:0'):
    model = Model(path_to_checkpoint, device)
    noise, identity, result = model.inference(batch_size)

    images = torch.cat([
        identity.cpu(),
        result.clamp(-1.0, 1.0).cpu(),
    ]).mul_(0.5).add_(0.5)
    grid = make_grid(images, nrow=batch_size, padding=0)
    grid = grid.permute(1, 2, 0).mul(255.).byte()
    Image.fromarray(grid.numpy()).save(f"images.jpg")
    print()


if __name__ == '__main__':
    main(
        path_to_checkpoint='checkpoints/stylegan/sad',
        batch_size=2,
    )


    
