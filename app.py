import streamlit as st
import torch
import warnings
import numpy as np
import torchvision.transforms.functional as F

from facenet_pytorch import MTCNN
from PIL import Image

from fomm.modules.keypoint_detector import KPDetector
from config import opt
from fomm.fomm_infer import load_checkpoints
from emotion_recognition_model import EmotionModel
from train_emotion_recognition import EmotionDataloader

warnings.filterwarnings("ignore")


@st.cache(allow_output_mutation=True)
def load_fomm():
    generator, kp_detector = load_checkpoints(
        config_path=opt.path_to_fomm_configs / 'vox-adv-256.yaml',
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
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last3)['state_dict'])
    return emotion_estimator.eval().requires_grad_(False).to(opt.device)


def get_emotion_vector():
    st.sidebar.title('Please, enter emotion vector')
    emotions_vector = []
    for emotion in opt.emotion_list:
        emotions_vector.append(
            st.sidebar.slider(
                emotion, min_value=0.0, max_value=1.0,
                value=0.0,
                step=0.01,
            )
        )
    return torch.tensor(emotions_vector, dtype=torch.float).unsqueeze(0).to(opt.device)


def generate_image(img_file_buffer, emotions_vector):
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
    boxes = []
    for i in range(len(batch_boxes)):
        margin = [
            mtcnn.margin * (batch_boxes[i][2] - batch_boxes[i][0]) / (mtcnn.image_size - mtcnn.margin),
            mtcnn.margin * (batch_boxes[i][3] - batch_boxes[i][1]) / (mtcnn.image_size - mtcnn.margin),
        ]

        box = [
            int(max(batch_boxes[i][0] - margin[0] / 2, 0)),
            int(max(batch_boxes[i][1] - margin[1] / 2, 0)),
            int(min(batch_boxes[i][2] + margin[0] / 2, img_original.shape[1])),
            int(min(batch_boxes[i][3] + margin[1] / 2, img_original.shape[0])),
        ]
        boxes.append(box)

    for idx, (out_img, box) in enumerate(zip(out_pred, boxes)):
        new_w = int(box[3] - box[1])
        new_h = int(box[2] - box[0])

        result.append(F.resize(out_img, size=[new_w, new_h]).detach().permute(1, 2, 0).cpu().numpy())

    img_cropped = img_cropped.detach().permute(0, 2, 3, 1).cpu().numpy()

    return img_original, img_cropped, result, boxes


if __name__ == '__main__':
    emotion_dataloader = EmotionDataloader(is_eval=True, use_mtcnn=False)
    generator, kp_detector, kp_detector_trainable = load_fomm()

    mtcnn = load_mtcnn()
    emotion_estimator = load_emotion()

    emotions_vector = get_emotion_vector()

    img_file_buffer = st.file_uploader('img_file_uploader', type=['jpeg', 'jpg'], accept_multiple_files=False)
    if img_file_buffer is not None:
        img_original, img_cropped, out_pred, boxes = generate_image(img_file_buffer, emotions_vector)

        st.image(
            img_original,
            caption="Uploaded image",
            width=512,
        )

        result_img = img_original.copy()
        if img_cropped is not None:
            for idx, (person_cropped, person_output, box) in enumerate(zip(img_cropped, out_pred, boxes)):
                result_img[int(box[1]):int(box[1]) + person_output.shape[0], int(box[0]):int(box[0]) + person_output.shape[1]] = (person_output*255).astype(int).clip(0, 255)
                with st.container():
                    beta_columns = st.columns(2)
                    beta_columns[0].image(
                        person_cropped,
                        caption=f"Cropped person {idx+1} - original",
                    )
                    beta_columns[1].image(
                        person_output,
                        caption=f"Cropped person {idx+1} - predicted",
                    )

            st.image(
                result_img,
                caption="Result of emotion swap",
                width=512,
            )
        else:
            st.warning("Sorry, I can't find face on this image.")

