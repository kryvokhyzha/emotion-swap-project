import itertools
import yaml

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from tqdm import tqdm
from collections.abc import Iterable

from config import opt
from fomm.fomm_infer import load_checkpoints
from fomm.modules.keypoint_detector import KPDetector
from emotion_swap.model import EmotionSwapFullModel
from emotion_swap.dataloader import EmotionSwapDataloader
from emotion_recognition.model import EmotionRecognitionModel
from emotion_recognition.dataloader import EmotionRecognitionDataloader
from helper import init_seeds, disable_warnings


def main(
        writer: SummaryWriter
):

    dataloader = EmotionSwapDataloader(
        opt.emotion_list, opt.path_to_stylegan_checkpoints, opt.device, eps=opt.exploration_ratio_kp,
    )
    em_dataloader = EmotionRecognitionDataloader(
        emotion_list=opt.emotion_list,
        path_to_stylegan_checkpoints=opt.path_to_stylegan_checkpoints,
        device=opt.device,
    )

    generator, kp_detector = load_checkpoints(
        config_path=opt.path_to_fomm_configs / 'vox-adv-256.yaml',
        checkpoint_path=opt.path_to_fomm_checkpoints / 'vox-adv-cpk.pth.tar',
        device=opt.device,
    )

    with open(opt.path_to_configs / 'emotion-swap-kp.yaml') as f:
        emotion_swap_config = yaml.full_load(f)

    kp_detector_trainable = KPDetector(
        **emotion_swap_config['model_params']['kp_detector_params'],
        **emotion_swap_config['model_params']['common_params']
    ).train().requires_grad_(True).to(opt.device)

    init_step = 0
    load_path = opt.path_to_kp_weights_last
    if load_path.exists():
        last_state = torch.load(load_path)
        kp_detector_trainable.load_state_dict(last_state['state_dict'])
        init_step = last_state['last_step'] + 1
        print(f'weights were loaded {load_path}')
    else:
        kp_detector_trainable.load_state_dict(kp_detector.state_dict(), strict=False)
        print(f'weights were loaded from original KP detector')

    kp_detector_trainable = kp_detector_trainable.requires_grad_(True).train()

    emotion_estimator = EmotionRecognitionModel()
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last)['state_dict'])
    emotion_estimator = emotion_estimator.eval().requires_grad_(False).to(opt.device)

    optimizer = Adam(kp_detector_trainable.parameters(), lr=emotion_swap_config['train_params']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.25, verbose=True)

    emotion_swap_full_model = EmotionSwapFullModel(
        kp_extractor=kp_detector_trainable, kp_extractor_static=kp_detector,
        generator=generator, emotion_recognition=emotion_estimator,
        emotion_recognition_dataloader=em_dataloader,
        train_params=emotion_swap_config['train_params'], device=opt.device,
    )

    for step in tqdm(itertools.count(init_step), initial=init_step, desc='infinity training loop'):
        optimizer.zero_grad()

        source, driving, target_emotions = dataloader.get_batch(opt.batch_size_kp)
        losses, generated = emotion_swap_full_model(
            {'source': source, 'driving': driving, 'target_emotions': target_emotions}
        )

        loss = sum(losses.values())

        loss.backward()
        optimizer.step()

        scheduler.step(loss.item())

        if ((step % opt.n_write_log_kp) == 0) and opt.enable_log_flag:
            writer.add_scalar('kp-lr/learning_rate', optimizer.param_groups[0]['lr'], step)
            writer.add_scalar("kp-loss/general_loss", loss.item(), step)

            for name, l in losses.items():
                coef_val = emotion_swap_config['train_params']['loss_weights'][name]
                coef_val = coef_val if not isinstance(coef_val, Iterable) else min(coef_val)

                writer.add_scalar("kp-loss/" + name, l.item(), step)
                writer.add_scalar("kp-coef/" + name, coef_val, step)

            if (step % opt.n_write_images_kp) == 0:
                with torch.no_grad():
                    out_GT = generator(source, kp_source=generated['kp_source'], kp_driving=generated['kp_driving_init'])['prediction']
                    out_pred = generator(source, kp_source=generated['kp_source'], kp_driving=generated['kp_driving'])['prediction']
                    images = torch.cat([source, driving, out_GT, out_pred])
                    grid = make_grid(images, nrow=opt.batch_size_kp, padding=0)
                    writer.add_image('kp-img/source__driving__out_GT__out_pred', grid, step)

        if ((step % opt.save_n_steps_kp) == 0) and opt.enable_log_flag:
            last_state = {
                'last_step': step,
                'state_dict': kp_detector_trainable.state_dict(),
            }
            torch.save(last_state, opt.path_to_kp_weights_last)


if __name__ == "__main__":
    disable_warnings()
    init_seeds(opt.seed)

    writer = SummaryWriter(log_dir=opt.path_to_kp_tf_logs)
    main(writer)
    writer.close()
