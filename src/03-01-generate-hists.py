import yaml
import numpy as np

import torch
import torch.nn.functional as functional
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import opt
from emotion_swap.dataloader import EmotionSwapDataloader
from emotion_recognition.model import EmotionRecognitionModel
from emotion_recognition.dataloader import EmotionRecognitionDataloader
from helper import init_seeds, disable_warnings


def main(
        writer: SummaryWriter
):
    dataloader = EmotionSwapDataloader(
        opt.emotion_list, opt.path_to_stylegan_checkpoints, opt.device,
        eps=-1.0, apply_mtcnn=True, apply_transformations=False,
    )
    em_dataloader = EmotionRecognitionDataloader(
        emotion_list=opt.emotion_list,
        path_to_stylegan_checkpoints=opt.path_to_stylegan_checkpoints,
        device=opt.device,
    )

    emotion_estimator = EmotionRecognitionModel()
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last)['state_dict'])
    emotion_estimator = emotion_estimator.eval().requires_grad_(False).to(opt.device)

    with open(opt.path_to_configs / 'emotion-swap-kp.yaml') as f:
        emotion_swap_config = yaml.full_load(f)

    opt.n_write_log_kp = 1
    opt.n_write_images_kp = 1
    opt.batch_size_kp = len(opt.emotion_list)
    steps = 100

    for emotion in opt.emotion_list:
        results = []
        for _ in tqdm(range(steps), desc=f'generating hists for emotion `{emotion}`'):
            _, driving, _, _, _ = dataloader.get_batch(1, emotion=emotion)

            with torch.no_grad():
                # compute driving emotion vector
                driving_er_prepared = em_dataloader.prepare_img(driving)
                results.append(functional.softmax(
                    emotion_estimator(driving_er_prepared) / emotion_swap_config['train_params']['emotion_temperature'],
                    dim=1,
                ).detach().cpu().numpy())

        results = np.concatenate(results, axis=0)

        for i in range(len(opt.emotion_list)):
            writer.add_histogram(f'model-{emotion}/distribution-of-emotion-{opt.emotion_list[i]}', results[:, i], i)


if __name__ == "__main__":
    disable_warnings()
    init_seeds(opt.seed)

    writer = SummaryWriter(log_dir=opt.path_to_kp_tf_logs)
    main(writer)
    writer.close()
