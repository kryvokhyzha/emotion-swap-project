import matplotlib
import sys
import yaml

import imageio
import numpy as np
from skimage.transform import resize

from imageio import imsave
import torch
from src.fomm.sync_batchnorm import DataParallelWithCallback

from src.fomm.modules.generator import OcclusionAwareGenerator
from src.fomm.modules.keypoint_detector import KPDetector

matplotlib.use('Agg')

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params']).to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params']).to(device)

    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval().requires_grad_(False)
    kp_detector.eval().requires_grad_(False)

    return generator, kp_detector


def make_animation(source_image, generator, kp_detector, device):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        kp_source, _ = kp_detector(source)  # x10 affine
        out = generator(source, kp_source=kp_source, kp_driving=kp_source)
        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def main(path):
    source_image = imageio.imread(path)

    source_image = resize(source_image, (256, 256))[..., :3]
    generator, kp_detector = load_checkpoints(config_path='config/vox-adv-256.yaml',
                                              checkpoint_path='checkpoints/vox-adv-cpk.pth.tar', device='cuda:0')

    predictions = make_animation(source_image, generator, kp_detector, device='cuda:0')
    imsave('data/van_rec.jpg', predictions[0])
    print()


if __name__ == "__main__":
    main('data/van_crop.jpg')
