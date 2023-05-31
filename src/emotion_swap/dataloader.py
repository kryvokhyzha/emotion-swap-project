import torch
import random
import numpy as np

from torchvision import transforms as T
from facenet_pytorch import MTCNN
from os.path import join
from stylegan_clip import Model


class EmotionSwapDataloader:
    def __init__(
            self,
            emotion_list, path_to_stylegan_checkpoints, device,
            eps=0.2, apply_transformations=True, apply_mtcnn=False,
    ):
        self.eps = eps
        self.apply_transformations = apply_transformations
        self.apply_mtcnn = apply_mtcnn
        self.emotion_list = emotion_list
        self.path_to_stylegan_checkpoints = path_to_stylegan_checkpoints
        self.device = device
        self.model = Model(path_to_checkpoint=join(path_to_stylegan_checkpoints, 'neutral'), device=device)

        if self.apply_mtcnn:
            self.mtcnn = MTCNN(
                image_size=256, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=device,
            )

    def augment_img(self, data, degrees=(0, 0), p_rand=0.5):

        if self.apply_transformations:
            kp_transformations_train = T.Compose([
                T.RandomRotation(degrees=degrees) if p_rand < 0.25 else T.RandomRotation(degrees=(0, 0)),
            ])
            data = kp_transformations_train(data).to(self.device)

        if self.apply_mtcnn:
            data = self.mtcnn(
                data.squeeze(0).permute(1, 2, 0).mul(255),
                return_prob=False,
            )
            if data is None:
                return None
            data = data.unsqueeze(0).add(1).div(2).to(self.device)
        return data

    def get_batch(self, batch_size, emotion=None):
        inputs_b, targets_b, emotions_b = [], [], []
        inputs_b_aug, targets_b_aug = [], []
        for b in range(batch_size):
            data_0 = None
            data_0_aug = None
            data_1 = None
            data_1_aug = None
            emo = None

            while data_0_aug is None or data_1_aug is None:
                noise = torch.randn(1, 512, device=self.device)
                if emotion is not None:
                    emo = emotion
                else:
                    emo = random.choice(self.emotion_list) if self.eps > 0 else self.emotion_list[b % len(self.emotion_list)]
                self.model.load_checkpoint(join(self.path_to_stylegan_checkpoints, emo), self.device)
                data = self.model.inference(1, noise)[1:]

                data_0 = data[0].add(1).div(2).to(self.device)
                data_1 = data[1].add(1).div(2).to(self.device)

                degree = np.random.randint(low=-45, high=45)
                p_rand = float(np.random.rand())

                data_0_aug = self.augment_img(data_0, degrees=(degree, degree), p_rand=p_rand)
                data_1_aug = self.augment_img(data_1, degrees=(degree, degree), p_rand=p_rand)

            inputs_b.append(data_0)
            inputs_b_aug.append(data_0_aug)
            target_emotion = torch.zeros([1, len(self.emotion_list)]).to(self.device)

            if np.random.rand() < self.eps:
                targets_b.append(data_0)
                targets_b_aug.append(data_0_aug)
                emotions_b.append(target_emotion)
            else:
                targets_b.append(data_1)
                targets_b_aug.append(data_1_aug)
                target_emotion[:, self.emotion_list.index(emo)] = 1
                emotions_b.append(target_emotion)
        return torch.cat(inputs_b), torch.cat(targets_b), torch.cat(inputs_b_aug), torch.cat(targets_b_aug), torch.cat(emotions_b)
