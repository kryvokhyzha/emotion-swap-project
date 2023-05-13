import torch
import random
import numpy as np

from os.path import join
from stylegan_clip import Model


class EmotionSwapDataloader:
    def __init__(
            self,
            emotion_list, path_to_stylegan_checkpoints, device,
            eps=0.2,
    ):
        self.eps = eps
        self.emotion_list = emotion_list
        self.path_to_stylegan_checkpoints = path_to_stylegan_checkpoints
        self.device = device
        self.model = Model(path_to_checkpoint=join(path_to_stylegan_checkpoints, 'neutral'), device=device)

    def prepare_img(self, data):
        raise NotImplementedError()

    def get_batch(self, batch_size):
        inputs_b, targets_b, emotions_b = [], [], []
        for b in range(batch_size):
            noise = torch.randn(1, 512, device=self.device)
            emo = random.choice(self.emotion_list)
            self.model.load_checkpoint(join(self.path_to_stylegan_checkpoints, emo), self.device)
            data = self.model.inference(1, noise)[1:]

            data_0 = data[0].add(1).div(2).to(self.device)
            data_1 = data[1].add(1).div(2).to(self.device)
            inputs_b.append(data_0)
            target_emotion = torch.zeros([1, len(self.emotion_list)]).to(self.device)

            if np.random.rand() < self.eps:
                targets_b.append(data_0)
                emotions_b.append(target_emotion)
            else:
                targets_b.append(data_1)
                target_emotion[:, self.emotion_list.index(emo)] = 1
                emotions_b.append(target_emotion)
        return torch.cat(inputs_b), torch.cat(targets_b), torch.cat(emotions_b)
