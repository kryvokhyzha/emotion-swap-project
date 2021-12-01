import streamlit as st
import torch
import warnings
import numpy as np
import torch.nn.functional as F

from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image

from modules.keypoint_detector import KPDetector
from config import opt
from demo_autoencoder import load_checkpoints
from emotion_recognition_model import EmotionModel

warnings.filterwarnings("ignore")


def load_fomm():
    generator, kp_detector = load_checkpoints(
        config_path=opt.path_to_fomm_checkpoints / 'config/vox-adv-256.yaml',
        checkpoint_path=opt.path_to_fomm_checkpoints / 'vox-adv-cpk.pth.tar',
        device=opt.device,
    )

    kp_detector_trainable = KPDetector(
        block_expansion=32, num_kp=10, num_channels=3, max_features=1024, num_blocks=5,
        temperature=0.1, estimate_jacobian=True, scale_factor=0.25,
        single_jacobian_map=False, pad=0, adain_size=7,
    )

    last_state = torch.load(opt.path_to_kp_weights_last2)
    kp_detector_trainable.load_state_dict(last_state['state_dict'])
    return (
        generator.eval().requires_grad_(False).to(opt.device),
        kp_detector.eval().requires_grad_(False).to(opt.device),
        kp_detector_trainable.eval().requires_grad_(False).to(opt.device),
    )


if __name__ == '__main__':
    generator, kp_detector, kp_detector_trainable = load_fomm()

    mtcnn = MTCNN(
        image_size=256, margin=100, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=opt.device
    )

    emotion_estimator = EmotionModel()
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last)['state_dict'])
    emotion_estimator = emotion_estimator.eval().requires_grad_(False).to(opt.device)

    img_file_buffer = st.file_uploader('img_file_uploader', type=['jpeg', 'jpg'], accept_multiple_files=False)
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = mtcnn(np.array(image), return_prob=False).add(1).div(2)

        inputs = img_array.unsqueeze(0).to(opt.device)

        emotions_vector = []
        inputs_normalized = torch.stack([transforms.Normalize(mean=opt.mean, std=opt.std)(pred) for pred in inputs])
        default_emotion_vector = F.softmax(emotion_estimator(inputs_normalized) / opt.temperature, dim=1)

        st.sidebar.title('Please, enter emotion vector')
        for emotion, default_value in zip(opt.emotion_list, default_emotion_vector[0].detach().cpu().numpy()):
            emotions_vector.append(
                st.sidebar.slider(
                    emotion, min_value=0.0, max_value=1.0,
                    value=float(np.round(default_value, 2)),
                    step=0.01
                )
            )
        emotions_vector = torch.tensor(emotions_vector, dtype=torch.float).unsqueeze(0).to(opt.device)

        source_kp, _ = kp_detector(inputs)
        pred_kp, _ = kp_detector_trainable(inputs, emotions_vector)
        out_pred = generator(inputs, kp_source=source_kp, kp_driving=pred_kp)['prediction']

        with st.container():
            beta_columns = st.columns(2)
            beta_columns[0].image(
                img_array.detach().permute(1, 2, 0).cpu().numpy(),
                caption="Uploaded image", width=256,
            )
            beta_columns[1].image(
                out_pred.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(),
                caption="Predicted image", width=256,
            )
