import itertools
import random
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm

from facenet_pytorch import MTCNN

from sklearn.metrics import f1_score

from src.config import opt
from src.stylegan_clip import Model
from src.emotion_recognition.model import EmotionModel

import warnings


warnings.filterwarnings("ignore")


def __init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_f1(preds, labels):
    return f1_score(labels, preds, average='micro')


class EmotionDataloader:
    def __init__(self, is_eval=False, use_mtcnn=True,):
        self.is_eval = is_eval
        self.use_mtcnn = use_mtcnn
        self.er_transformations = T.Compose([
            T.GaussianBlur(kernel_size=(3, 3), sigma=1),
            T.GaussianBlur(kernel_size=(5, 9), sigma=2),
            T.Normalize(mean=opt.mean, std=opt.std),
        ])

        self.er_transformations_train = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomChoice([
                T.RandomRotation(degrees=(-45, 45)),
                T.RandomRotation(degrees=(0, 0)),
            ], p=[0.25, 0.75]),
        ])

        self.mtcnn = MTCNN(
            image_size=256, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=opt.device
        )
        self.model = Model(path_to_checkpoint=join(opt.path_to_stylegan_checkpoints, 'neutral'), device=opt.device)

    def prepare_img(self, data):
        data = data.add(1).div(2).squeeze()
        if not self.is_eval:
            data = self.er_transformations_train(data).to(opt.device)

        if self.use_mtcnn:
            data = self.mtcnn(
                data.permute(1, 2, 0).mul(255).detach().cpu().numpy(),
                return_prob=False,
            )
            if data is None:
                return None
            data = data.add(1).div(2).squeeze()
        return self.er_transformations(data).to(opt.device)

    def get_batch(self, batch_size):
        targets_b, emotions_b = [], []
        for b in range(batch_size):
            data = None
            emo = None
            while data is None:
                noise = torch.randn(1, 512, device=opt.device)
                emo = random.choice(opt.emotion_list)
                self.model.load_checkpoint(join(opt.path_to_stylegan_checkpoints, emo))
                data = self.model.inference(1, noise)[1:]
                data = self.prepare_img(data[1])
            data = data.unsqueeze(0)

            targets_b.append(data)
            target_emotion = torch.zeros([1, 7]).to(opt.device)
            target_emotion[:, opt.emotion_list.index(emo)] = 1
            emotions_b.append(target_emotion)
        return torch.cat(targets_b), torch.cat(emotions_b)


def main():
    dataloader = EmotionDataloader(is_eval=False, use_mtcnn=False)

    emotion_model = EmotionModel().train().requires_grad_(True).to(opt.device)
    emotion_model.freeze_middle_layers()

    init_step = 0
    if opt.path_to_er_weights_last3.exists():
        last_state = torch.load(opt.path_to_er_weights_last3)
        emotion_model.load_state_dict(last_state['state_dict'])
        init_step = last_state['last_step'] + 1
        print(f'weight were loaded {opt.path_to_er_weights_last3}')

    emotion_model.unfreeze_all_layers()
    optimizer = Adam(emotion_model.parameters(), lr=opt.lr_er)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.75, verbose=True)

    loss_history = []
    f1_history = []
    best_avg_f1 = 0
    for step in tqdm(itertools.count(init_step), initial=init_step, desc='infinity training loop'):
        optimizer.zero_grad()
        target, target_emotions = dataloader.get_batch(opt.batch_size_er)
        predicted_emotions = emotion_model(target)

        val_target, val_target_emotions = dataloader.get_batch(opt.batch_size_er)
        val_predicted_emotions = emotion_model(val_target)

        loss = F.cross_entropy(predicted_emotions, target_emotions)
        val_loss = F.cross_entropy(val_predicted_emotions, val_target_emotions)

        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        val_f1_score_res = calculate_f1(
            np.argmax(F.softmax(val_predicted_emotions, dim=1).detach().cpu().numpy().tolist(), axis=1),
            np.argmax(val_target_emotions.cpu().numpy().tolist(), axis=1),
        )

        loss_history.append(val_loss.item())
        f1_history.append(val_f1_score_res)

        if ((step % opt.n_write_log) == 0) and opt.enable_log_flag:
            grid = make_grid(target, nrow=opt.batch_size_er, padding=0, normalize=True,)
            writer.add_image('er-img/target', grid, step)

            avg_f1 = np.mean(f1_history)

            writer.add_scalar('er-lr/learning_rate', optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('er-loss/cross-entropy-loss', val_loss.item(), step)
            writer.add_scalar('er-metric/f1-micro-metric', val_f1_score_res, step)
            writer.add_scalar('er-loss/avg-cross-entropy-loss', np.mean(loss_history), step)
            writer.add_scalar('er-metric/avg-f1-micro-metric', avg_f1, step)
            loss_history = []
            f1_history = []

            if best_avg_f1 < avg_f1:
                best_avg_f1 = avg_f1
                last_state = {
                    'last_step': step,
                    'state_dict': emotion_model.state_dict(),
                }
                torch.save(last_state, opt.path_to_er_weights_last3)

                if best_avg_f1 >= 1.0:
                    return


if __name__ == "__main__":
    __init_seeds(opt.seed)

    writer = SummaryWriter(log_dir=opt.path_to_er_tf_logs3)
    main()
    writer.close()

    print('Training is finished!')
