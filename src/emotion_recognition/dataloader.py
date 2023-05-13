import torch
import random

from os.path import join
from stylegan_clip import Model
from helper.gradient_filter import GradientMagnitudeFilter


class EmotionRecognitionDataloader:

    def __init__(self, emotion_list, path_to_stylegan_checkpoints, device):
        self.gradient_filter = GradientMagnitudeFilter(device=device).eval().requires_grad_(False).to(device)
        self.model = Model(path_to_checkpoint=join(path_to_stylegan_checkpoints, 'neutral'), device=device)
        self.device = device
        self.emotion_list = emotion_list
        self.path_to_stylegan_checkpoints = path_to_stylegan_checkpoints

    def prepare_img(self, data):
        grad_magnitude = self.gradient_filter(data)
        _shapes = grad_magnitude.shape

        grad_magnitude = grad_magnitude.view(grad_magnitude.size(0), -1)

        _min = grad_magnitude.min(dim=1, keepdim=True)[0]
        _max = grad_magnitude.max(dim=1, keepdim=True)[0]

        grad_magnitude = (grad_magnitude - _min) / (_max - _min)

        grad_magnitude = grad_magnitude.view(*_shapes)

        return grad_magnitude

    def get_batch(self, batch_size):
        targets_b, emotions_b = [], []
        for b in range(batch_size):
            noise = torch.randn(1, 512, device=self.device)
            rand_emotion_idx = random.choice(self.emotion_list)

            self.model.load_checkpoint(join(self.path_to_stylegan_checkpoints, rand_emotion_idx), self.device)

            data = self.model.inference(1, noise)[1:]
            data = self.prepare_img(data[1].add(1).div(2))

            targets_b.append(data)
            target_emotion = torch.zeros([1, len(self.emotion_list)]).to(self.device)
            target_emotion[:, self.emotion_list.index(rand_emotion_idx)] = 1
            emotions_b.append(target_emotion)
        return torch.cat(targets_b), torch.cat(emotions_b)
