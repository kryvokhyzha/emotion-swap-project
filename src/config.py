import os
import torch
from pathlib import Path


class Config:
    def __init__(self):
        self.path_to_root = Path('..')
        self.path_to_project = Path('')

        self.path_to_output = self.path_to_root / 'output'

        self.path_to_pretrained_checkpoints = self.path_to_output / 'checkpoints'
        self.path_to_stylegan_checkpoints = self.path_to_project / 'stylegan_clip' / 'checkpoints' / 'stylegan'
        self.path_to_fomm_checkpoints = self.path_to_project / 'fomm' / 'checkpoints'

        self.path_to_fomm_configs = self.path_to_project / 'fomm' / 'config'

        self.path_to_kp_checkpoints2 = self.path_to_pretrained_checkpoints / 'kp-detector-with-er-heatmap-v1-snp'
        self.path_to_kp_weights2 = self.path_to_kp_checkpoints2 / 'weights'
        self.path_to_kp_weights_last2 = self.path_to_kp_weights2 / 'latest.pth'
        self.path_to_kp_tf_logs2 = self.path_to_kp_checkpoints2 / 'tf_log'

        self.path_to_kp_checkpoints3 = self.path_to_pretrained_checkpoints / 'kp-detector-with-er-heatmap-v1'
        self.path_to_kp_weights3 = self.path_to_kp_checkpoints3 / 'weights'
        self.path_to_kp_weights_last3 = self.path_to_kp_weights3 / 'latest.pth'
        self.path_to_kp_tf_logs3 = self.path_to_kp_checkpoints3 / 'tf_log'

        self.path_to_er_checkpoints2 = self.path_to_pretrained_checkpoints / 'emotion-recognition-v2'
        self.path_to_er_weights2 = self.path_to_er_checkpoints2 / 'weights'
        self.path_to_er_weights_last2 = self.path_to_er_weights2 / 'latest.pth'
        self.path_to_er_tf_logs2 = self.path_to_er_checkpoints2 / 'tf_log'

        self.path_to_er_checkpoints3 = self.path_to_pretrained_checkpoints / 'emotion-recognition-v3'
        self.path_to_er_weights3 = self.path_to_er_checkpoints3 / 'weights'
        self.path_to_er_weights_last3 = self.path_to_er_weights3 / 'latest.pth'
        self.path_to_er_tf_logs3 = self.path_to_er_checkpoints3 / 'tf_log'

        self.emotion_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.load_checkpoint_path = ''
        self.lr_kp = 3e-5
        self.lr_er = 3e-4

        self.batch_size_kp = 10
        self.batch_size_er = 16

        self.n_write_log = 10
        self.save_n_steps = 300

        self.l1_kp = 1.0
        self.js_div_target = 0.1
        self.l1_target = 0.1
        self.ce_loss = 0.0  # 0.1
        self.l1_heatmap = 0.0  # 0.1

        self.temperature = 2

        self.seed = 42
        self.enable_log_flag = False


opt = Config()

opt.path_to_kp_weights3.mkdir(exist_ok=True)
opt.path_to_er_weights3.mkdir(exist_ok=True)
opt.path_to_output.mkdir(exist_ok=True)
