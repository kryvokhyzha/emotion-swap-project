import torch
from pathlib import Path


class Config:

    def __init__(self):

        self.path_to_root = Path('..')
        self.path_to_src = Path('')

        self.path_to_output = self.path_to_root / 'output'
        self.path_to_configs = self.path_to_root / 'configs'

        self.path_to_pretrained_checkpoints = self.path_to_output / 'checkpoints'
        self.path_to_stylegan_checkpoints = self.path_to_pretrained_checkpoints / 'styleclip' / 'stylegan'
        self.path_to_stylegan_segmentation = self.path_to_pretrained_checkpoints / 'styleclip' / 'segmentation.pkl'
        self.path_to_fomm_checkpoints = self.path_to_pretrained_checkpoints / 'fomm'

        self.path_to_fomm_configs = self.path_to_src / 'fomm' / 'config'

        self.path_to_kp_checkpoints = self.path_to_pretrained_checkpoints / 'kp-detector'
        # self.path_to_kp_checkpoints = self.path_to_pretrained_checkpoints / 'kp-detector-temp'
        self.path_to_kp_weights = self.path_to_kp_checkpoints / 'weights'
        self.path_to_kp_weights_last = self.path_to_kp_weights / 'latest.pth'
        self.path_to_kp_tf_logs = self.path_to_kp_checkpoints / 'tf_log'

        self.path_to_er_checkpoints = self.path_to_pretrained_checkpoints / 'emotion-recognition-gradient-magnitude'
        # self.path_to_er_checkpoints = self.path_to_pretrained_checkpoints / 'emotion-recognition-gradient-magnitude-temp'
        self.path_to_er_weights = self.path_to_er_checkpoints / 'weights'
        self.path_to_er_weights_last = self.path_to_er_weights / 'latest.pth'
        self.path_to_er_tf_logs = self.path_to_er_checkpoints / 'tf_log'

        self.emotion_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

        self.lr_er = 3e-4

        self.exploration_ratio_kp = 0.2

        self.batch_size_kp = 8
        self.batch_size_er = 16

        self.aggregation_metric_slice_window_width_er = 10
        self.aggregation_metric_slice_window_width_kp = 10

        self.n_write_log_er = 10

        self.n_write_log_kp = 10
        self.n_write_images_kp = 100
        self.save_n_steps_kp = 300

        # self.l1_kp = 1.0
        # self.js_div_target = 0.1
        # self.l1_target = 0.1
        # self.ce_loss = 0.0  # 0.1
        # self.l1_heatmap = 0.0  # 0.1

        # self.temperature = 2

        self.seed = 42
        self.enable_log_flag = True

        self._init_dirs()

    def _init_dirs(self):
        self.path_to_kp_weights.mkdir(exist_ok=True, parents=True)
        self.path_to_er_weights.mkdir(exist_ok=True, parents=True)


opt = Config()
