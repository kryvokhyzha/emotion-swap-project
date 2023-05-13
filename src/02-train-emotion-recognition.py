import itertools
import numpy as np
import torch
import torch.nn.functional as functional

from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from tqdm import tqdm

from config import opt
from emotion_recognition.dataloader import EmotionRecognitionDataloader
from emotion_recognition.model import EmotionRecognitionModel
from helper import calculate_f1, init_seeds, disable_warnings
from helper.early_stopping import EarlyStopping


def main(
        writer: SummaryWriter
):
    dataloader = EmotionRecognitionDataloader(
        emotion_list=opt.emotion_list,
        path_to_stylegan_checkpoints=opt.path_to_stylegan_checkpoints,
        device=opt.device,
    )

    emotion_model = EmotionRecognitionModel().train().requires_grad_(True).to(opt.device)

    init_step = 0
    if opt.path_to_er_weights_last.exists():
        last_state = torch.load(opt.path_to_er_weights_last)
        emotion_model.load_state_dict(last_state['state_dict'])
        init_step = last_state['last_step'] + 1
        print(f'weights were loaded {opt.path_to_er_weights_last}')

    emotion_model.unfreeze_all_layers()
    optimizer = Adam(emotion_model.parameters(), lr=opt.lr_er)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.75, verbose=True)
    early_stopping = EarlyStopping(tolerance=500)

    loss_history = []
    f1_history = []
    best_avg_f1 = 0
    for step in tqdm(itertools.count(init_step), initial=init_step, desc='infinity training loop'):
        optimizer.zero_grad()

        generated_imgs, target_emotions = dataloader.get_batch(opt.batch_size_er)
        predicted_emotions = emotion_model(generated_imgs.detach())

        val_generated_imgs, val_target_emotions = dataloader.get_batch(opt.batch_size_er)
        val_predicted_emotions = emotion_model(val_generated_imgs.detach())

        loss = functional.cross_entropy(predicted_emotions, target_emotions)
        val_loss = functional.cross_entropy(val_predicted_emotions, val_target_emotions)

        loss.backward()
        optimizer.step()

        scheduler.step(loss.item())

        val_f1_score_res = calculate_f1(
            np.argmax(functional.softmax(val_predicted_emotions, dim=1).detach().cpu().numpy().tolist(), axis=1),
            np.argmax(val_target_emotions.cpu().numpy().tolist(), axis=1),
        )

        loss_history.append(val_loss.item())
        f1_history.append(val_f1_score_res)

        if ((step % opt.n_write_log_er) == 0) and opt.enable_log_flag:
            grid = make_grid(generated_imgs, nrow=opt.batch_size_er, padding=0, normalize=True,)
            writer.add_image('er-img/generated-imgs', grid, step)

            avg_f1 = np.mean(f1_history[-opt.aggregation_metric_slice_window_width_er:])
            avg_loss = np.mean(loss_history[-opt.aggregation_metric_slice_window_width_er:])

            writer.add_scalar('er-lr/learning_rate', optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('er-loss/cross-entropy-loss', val_loss.item(), step)
            writer.add_scalar('er-metric/f1-micro-metric', val_f1_score_res, step)
            writer.add_scalar(f'er-loss/avg-last-{opt.aggregation_metric_slice_window_width_er}-cross-entropy-loss', avg_loss, step)
            writer.add_scalar(f'er-metric/avg-last-{opt.aggregation_metric_slice_window_width_er}-f1-micro-metric', avg_f1, step)
            loss_history = []
            f1_history = []

            if best_avg_f1 < avg_f1:
                best_avg_f1 = avg_f1
                last_state = {
                    'last_step': step,
                    'state_dict': emotion_model.state_dict(),
                    'metric': avg_f1
                }
                torch.save(last_state, opt.path_to_er_weights_last)

                if best_avg_f1 >= 1.0:
                    print(f'Training is finished! Best avg f1 score is {best_avg_f1}!')
                    break

        early_stopping(loss.item(), val_loss.item())
        if early_stopping.early_stop:
            print(f'Early stopping! Average f1 score is {np.mean(f1_history)}!')
            break


if __name__ == "__main__":
    disable_warnings()
    init_seeds(opt.seed)

    writer = SummaryWriter(log_dir=opt.path_to_er_tf_logs)
    main(writer)
    writer.close()

    print('Training is finished!')
