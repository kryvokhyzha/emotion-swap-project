import streamlit as st
import torch
import warnings
import numpy as np
import torchvision.transforms.functional as F
import cv2

from facenet_pytorch import MTCNN
from PIL import Image

from modules.keypoint_detector import KPDetector
from config import opt
from demo_autoencoder import load_checkpoints
from emotion_recognition_model import EmotionModel
from train_er import EmotionDataloader

warnings.filterwarnings("ignore")


@st.cache(allow_output_mutation=True)
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

    last_state = torch.load(opt.path_to_kp_weights_last3)
    kp_detector_trainable.load_state_dict(last_state['state_dict'])
    return (
        generator.eval().requires_grad_(False).to(opt.device),
        kp_detector.eval().requires_grad_(False).to(opt.device),
        kp_detector_trainable.eval().requires_grad_(False).to(opt.device),
    )


@st.cache(allow_output_mutation=True,)
def load_mtcnn():
    return MTCNN(
        image_size=256, margin=100, min_face_size=20,
        thresholds=[0.8, 0.8, 0.8], factor=0.709, post_process=True,
        keep_all=True,
        device=opt.device
    )


@st.cache(allow_output_mutation=True)
def load_emotion():
    emotion_estimator = EmotionModel()
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last2)['state_dict'])
    return emotion_estimator.eval().requires_grad_(False).to(opt.device)


def get_emotion_vector():
    st.sidebar.title('Please, enter emotion vector')
    emotions_vector = []
    for emotion in opt.emotion_list:
        emotions_vector.append(
            st.sidebar.slider(
                emotion, min_value=0.0, max_value=1.0,
                value=0.0,
                step=0.01
            )
        )
    return torch.tensor(emotions_vector, dtype=torch.float).unsqueeze(0).to(opt.device)


def generate_image(img_file_buffer):
    image = Image.open(img_file_buffer)
    img_original = np.array(image)

    # Detect faces
    batch_boxes, batch_probs, batch_points = mtcnn.detect(img_original, landmarks=True)
    # Select faces
    if not mtcnn.keep_all:
        batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
            batch_boxes, batch_probs, batch_points, img_original, method=mtcnn.selection_method
        )
    # Extract faces
    img_cropped = mtcnn.extract(img_original, batch_boxes, None)

    if img_cropped is None:
        return img_original, None, None
    else:
        img_cropped = img_cropped.add(1).div(2)

    if len(img_cropped.shape) == 3:
        inputs = img_cropped.unsqueeze(0).to(opt.device)
    else:
        inputs = img_cropped.to(opt.device)

    source_kp, _ = kp_detector(inputs)
    pred_kp, _ = kp_detector_trainable(inputs, emotions_vector)
    out_pred = generator(inputs, kp_source=source_kp, kp_driving=pred_kp)['prediction']

    result = []
    for idx, (out_img, box) in enumerate(zip(out_pred, batch_boxes)):
        new_w = int(box[3] - box[1])
        new_h = int(box[2] - box[0])

        # result.append(F.resize(out_img, size=[new_w, new_h]).detach().permute(1, 2, 0).cpu().numpy())
        result.append(out_img.detach().permute(1, 2, 0).cpu().numpy())

    img_cropped = img_cropped.detach().permute(0, 2, 3, 1).cpu().numpy()

    return img_original, img_cropped, result, batch_boxes


if __name__ == '__main__':
    emotion_dataloader = EmotionDataloader(is_eval=True)
    generator, kp_detector, kp_detector_trainable = load_fomm()

    mtcnn = load_mtcnn()
    emotion_estimator = load_emotion()

    emotions_vector = get_emotion_vector()

    img_file_buffer = st.file_uploader('img_file_uploader', type=['jpeg', 'jpg'], accept_multiple_files=False)
    if img_file_buffer is not None:
        img_original, img_cropped, out_pred, batch_boxes = generate_image(img_file_buffer)

        st.image(
            img_original,
            caption="Uploaded image",
        )

        # cp_or1 = img_original.copy()
        # cp_or2 = img_original.copy()

        rec_img = img_original.copy()

        if img_cropped is not None:
            for person_cropped, person_output, box in zip(img_cropped, out_pred, batch_boxes):
                with st.container():
                    beta_columns = st.columns(2)
                    beta_columns[0].image(
                        person_cropped,
                        caption="Cropped uploaded image",
                    )
                    beta_columns[1].image(
                        person_output,
                        caption="Cropped predicted image",
                    )

                    # Create an all white mask
                    mask = 255 * np.ones(person_output.shape, 'uint8')
                    center = (int(np.abs(box[2] + box[0]) // 2), int(np.abs(box[3] + box[1]) // 2))
                    # The location of the center of the src in the dst
                    width, height, channels = img_original.shape

                    # rec_img = cv2.rectangle(rec_img, (int(box[0])-50, int(box[1])-50), (int(box[2])+50, int(box[3])+50), (255, 0, 0), 2)

                    # cp_or1 = cv2.seamlessClone(person_output.copy(), img_original.copy(), mask, p=center, flags=cv2.NORMAL_CLONE)
                    # cp_or2 = cv2.seamlessClone(person_output.copy(), img_original.copy(), mask, p=center, flags=cv2.MIXED_CLONE)

                    # st.image(cp_or1)
                    # st.image(cp_or2)
            # st.image(rec_img)
        else:
            st.warning("Sorry, I can't find face on this image.")

