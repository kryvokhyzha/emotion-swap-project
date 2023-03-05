import itertools
import random
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchvision.utils import make_grid
from torchvision import transforms as T
from facenet_pytorch import MTCNN
from tqdm import tqdm

from src.config import opt
from src.fomm import load_checkpoints
from src.fomm.modules.keypoint_detector import KPDetector
from src.stylegan_clip import Model
from src.emotion_recognition.model import EmotionModel
from train_emotion_recognition import EmotionDataloader

import warnings

warnings.filterwarnings("ignore")


def __init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return (0.5 * F.kl_div(p, m, reduction='batchmean', log_target=False)
            + 0.5 * F.kl_div(q, m, reduction='batchmean', log_target=False))


class Dataloader:
    def __init__(self, is_eval=False, use_mtcnn=True,):
        self.is_eval = is_eval
        self.use_mtcnn = use_mtcnn
        self.mtcnn = MTCNN(
            image_size=256, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=opt.device
        )
        self.model = Model(path_to_checkpoint=join(opt.path_to_stylegan_checkpoints, 'neutral'), device=opt.device)

    def prepare_img(self, data, degrees=(0, 0), p_rand=0.5):
        data = data.add(1).div(2).squeeze()
        if not self.is_eval:
            kp_transformations_train = T.Compose([
                T.RandomRotation(degrees=degrees) if p_rand < 0.25 else T.RandomRotation(degrees=(0, 0)),
            ])
            data = kp_transformations_train(data).to(opt.device)

        if self.use_mtcnn:
            data = self.mtcnn(
                data.permute(1, 2, 0).mul(255).detach().cpu().numpy(),
                return_prob=False,
            )
            if data is None:
                return None
            data = data.add(1).div(2).squeeze()
        return data.to(opt.device)  # self.kp_transformations(data).to(opt.device)

    def get_batch(self, batch_size):
        inputs_b, targets_b, emotions_b = [], [], []
        for b in range(batch_size):
            data_0 = None
            data_1 = None
            emo = None
            while data_0 is None or data_1 is None:
                noise = torch.randn(1, 512, device=opt.device)
                emo = random.choice(opt.emotion_list)
                self.model.load_checkpoint(join(opt.path_to_stylegan_checkpoints, emo))
                data = self.model.inference(1, noise)[1:]

                degree = np.random.randint(low=-45, high=45)
                p_rand = float(np.random.rand())
                data_0 = self.prepare_img(data[0], degrees=(degree, degree), p_rand=p_rand)
                data_1 = self.prepare_img(data[1], degrees=(degree, degree), p_rand=p_rand)
            inputs_b.append(data_0.unsqueeze(0))
            target_emotion = torch.zeros([1, 7]).to(opt.device)

            if np.random.rand() < 0.2:
                targets_b.append(data_0.unsqueeze(0))
                emotions_b.append(target_emotion)
            else:
                targets_b.append(data_1.unsqueeze(0))
                target_emotion[:, opt.emotion_list.index(emo)] = 1
                emotions_b.append(target_emotion)
        return torch.cat(inputs_b), torch.cat(targets_b), torch.cat(emotions_b)


def main():
    dataloader = Dataloader(is_eval=False, use_mtcnn=False)
    em_dataloader = EmotionDataloader(is_eval=True, use_mtcnn=False)

    generator, kp_detector = load_checkpoints(
        config_path=opt.path_to_fomm_configs / 'vox-adv-256.yaml',
        checkpoint_path=opt.path_to_fomm_checkpoints / 'vox-adv-cpk.pth.tar',
        device=opt.device,
    )
    kp_detector_trainable = KPDetector(
        block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5,
        temperature=0.1, estimate_jacobian=True, scale_factor=0.25,
        single_jacobian_map=False, pad=0, adain_size=7,
    ).train().requires_grad_(True).to(opt.device)

    init_step = 0
    load_path = opt.path_to_kp_weights_last3
    if load_path.exists():
        last_state = torch.load(load_path)
        kp_detector_trainable.load_state_dict(last_state['state_dict'])
        init_step = last_state['last_step'] + 1
        print(f'weight were loaded {load_path}')
    else:
        kp_detector_trainable.load_state_dict(kp_detector.state_dict(), strict=False)
        print(f'weight were loaded from original KP detector')

    kp_detector_trainable = kp_detector_trainable.requires_grad_(True).train()

    emotion_estimator = EmotionModel()
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last3)['state_dict'])
    emotion_estimator = emotion_estimator.eval().requires_grad_(False).to(opt.device)

    optimizer = Adam(kp_detector_trainable.parameters(), lr=opt.lr_kp)
    for step in tqdm(itertools.count(init_step), initial=init_step, desc='infinity training loop'):
        optimizer.zero_grad()

        inputs, target, target_emotions = dataloader.get_batch(opt.batch_size_kp)
        target_prepared = torch.stack([em_dataloader.prepare_img(pred) for pred in target])
        with torch.no_grad():
            target_kp, target_heatmap = kp_detector(target)
            target_emotions_pred = F.softmax(emotion_estimator(target_prepared) / opt.temperature, dim=1)
        mask = target_emotions.sum(dim=1)
        target_emotions_pred = mask.unsqueeze(0).T * target_emotions_pred
        mask = mask > 0
        pred_kp, pred_heatmap = kp_detector_trainable(inputs, target_emotions_pred)

        source_kp, source_heatmap = kp_detector(inputs)  # x10 affine, x10 heatmaps
        out_pred = generator(inputs, kp_source=source_kp, kp_driving=pred_kp)['prediction']

        out_pred_prepared = torch.stack([em_dataloader.prepare_img(pred) for pred in out_pred])
        out_pred_emotions = F.softmax(emotion_estimator(out_pred_prepared) / opt.temperature, dim=1)

        losses = {
            'l1_kp': sum([F.l1_loss(pred_kp[k], target_kp[k]) for k in target_kp.keys()]),
            'l1_target': F.l1_loss(out_pred_emotions[mask], target_emotions_pred[mask]),
            'js_div_target': F.kl_div(
                pred_heatmap.sum(dim=1).view(opt.batch_size_kp, -1).add(1).log(),
                source_heatmap.sum(dim=1).view(opt.batch_size_kp, -1).add(1),
                reduction='batchmean', log_target=False
            ),
            # 'js_div_target': js_divergence(
            #     pred_heatmap.sum(dim=1).view(opt.batch_size_kp, -1).add(1).log(),
            #     source_heatmap.sum(dim=1).view(opt.batch_size_kp, -1).add(1).log()
            # ),
            # 'l1_heatmap': F.l1_loss(
            #     pred_heatmap.sum(dim=1).view(opt.batch_size_kp, -1),
            #     source_heatmap.sum(dim=1).view(opt.batch_size_kp, -1),
            # ),
        }
        loss = sum([getattr(opt, k) * l for k, l in losses.items()])

        loss.backward()
        optimizer.step()

        if ((step % opt.n_write_log) == 0) and opt.enable_log_flag:
            writer.add_scalar('kp-lr/learning_rate', optimizer.param_groups[0]['lr'], step)
            for name, l in losses.items():
                writer.add_scalar("kp-loss/" + name, l.item(), step)
                writer.add_scalar("kp-coef/" + name, getattr(opt, name), step)

            with torch.no_grad():
                out_GT = generator(inputs, kp_source=source_kp, kp_driving=target_kp)['prediction']
                out_pred = generator(inputs, kp_source=source_kp, kp_driving=pred_kp)['prediction']
                images = torch.cat([inputs, target, out_GT, out_pred])
                grid = make_grid(images, nrow=opt.batch_size_kp, padding=0)
                writer.add_image('kp-img/inputs__target__out_GT__out_pred', grid, step)

        if ((step % opt.save_n_steps) == 0) and opt.enable_log_flag:
            last_state = {
                'last_step': step,
                'state_dict': kp_detector_trainable.state_dict(),
            }
            # torch.save(last_state, opt.path_to_kp_weights_last1)
            torch.save(last_state, opt.path_to_kp_weights_last3)


if __name__ == "__main__":
    __init_seeds(opt.seed)

    writer = SummaryWriter(log_dir=opt.path_to_kp_tf_logs3)
    main()
    writer.close()
