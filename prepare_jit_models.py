import torch
import warnings

from fomm.modules.keypoint_detector import KPDetector
from config import opt
from fomm.fomm_infer import load_checkpoints
from emotion_recognition_model import EmotionModel

warnings.filterwarnings("ignore")


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


def load_emotion():
    emotion_estimator = EmotionModel()
    emotion_estimator.load_state_dict(torch.load(opt.path_to_er_weights_last3)['state_dict'])
    return emotion_estimator.eval().requires_grad_(False).to(opt.device)


def prepare_jit_files():
    generator, kp_detector, kp_detector_trainable = load_fomm()
    emotion_estimator = load_emotion()

    generator_script = torch.jit.script(generator)
    kp_detector_script = torch.jit.script(kp_detector)
    kp_detector_trainable_script = torch.jit.script(kp_detector_trainable)
    emotion_estimator_script = torch.jit.script(emotion_estimator)

    torch.jit.save(generator_script, 'generator.jit.pt')
    torch.jit.save(kp_detector_script, 'kp_detector.jit.pt')
    torch.jit.save(kp_detector_trainable_script, 'kp_detector_trainable.jit.pt')
    torch.jit.save(emotion_estimator_script, 'emotion_estimator.jit.pt')


def main():
    prepare_jit_files()


if __name__ == '__main__':
    main()
