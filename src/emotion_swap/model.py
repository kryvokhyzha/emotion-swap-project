import torch
import torch.nn.functional as functional
from fomm.modules.model import ImagePyramide, Vgg19, Transform


class EmotionSwapFullModel:
    """
    Merge all `emotion swap` related updates into single model
    """

    def __init__(
            self, kp_extractor, kp_extractor_static,
            generator, emotion_recognition,
            emotion_recognition_dataloader,
            train_params, device, suffix,
    ):
        super(EmotionSwapFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.kp_extractor_static = kp_extractor_static
        self.generator = generator
        self.emotion_recognition = emotion_recognition
        self.emotion_recognition_dataloader = emotion_recognition_dataloader
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, generator.module.num_channels)
        self.pyramid = self.pyramid.to(device)
        self.suffix = suffix

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19().to(device)

    def __call__(self, x):

        with torch.no_grad():
            # compute initial driving KP (driving -> kp_extractor_static)
            kp_driving_init, heatmap_driving_init = self.kp_extractor_static(x[f'driving{self.suffix}'])

            # compute driving emotion vector
            driving_er_prepared = self.emotion_recognition_dataloader.prepare_img(x['driving'])
            driving_emotion_vector = functional.softmax(
                self.emotion_recognition(driving_er_prepared) / self.train_params['emotion_temperature'],
                dim=1,
            )

        # compute emotion mask
        mask = x['target_emotions'].sum(dim=1)
        driving_emotion_vector = mask.unsqueeze(0).T * driving_emotion_vector
        mask = mask > 0

        # compute predicted driving KP (source + emotion -> kp_extractor)
        kp_driving, heatmap_driving = self.kp_extractor(x[f'source{self.suffix}'], driving_emotion_vector)
        # compute source KP (source -> kp_extractor_static)
        kp_source, heatmap_source = self.kp_extractor_static(x[f'source{self.suffix}'])

        # generate image
        generated = self.generator(x[f'source{self.suffix}'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({f'kp_source{self.suffix}': kp_source, f'kp_driving{self.suffix}': kp_driving, 'kp_driving_init': kp_driving_init,})

        loss_values = {}
        loss_values_raw = {}

        pyramide_real = self.pyramid(x[f'driving{self.suffix}'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            value_without_weight = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                    value_without_weight += value
                loss_values['perceptual'] = value_total
                loss_values_raw['perceptual'] = value_without_weight

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x[f'driving{self.suffix}'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x[f'driving{self.suffix}'])
            transformed_kp = self.kp_extractor(transformed_frame, driving_emotion_vector)[0]

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            # Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
                loss_values_raw['equivariance_value'] = value

            # jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(
                    transform.jacobian(transformed_kp['value']),
                    transformed_kp['jacobian'],
                )

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
                loss_values_raw['equivariance_jacobian'] = value

        # emotion vector loss part
        if self.loss_weights['emotion_vectors'] != 0:
            generated_er_prepared = self.emotion_recognition_dataloader.prepare_img(generated['prediction'])
            generated_emotion_vector = functional.log_softmax(
                self.emotion_recognition(generated_er_prepared) / self.train_params['emotion_temperature'],
                dim=1,
            )
            # value = functional.l1_loss(
            #     generated_emotion_vector[mask], driving_emotion_vector[mask]
            # )

            value = functional.kl_div(
                generated_emotion_vector[mask], driving_emotion_vector[mask]
            )
            loss_values['emotion_vectors'] = self.loss_weights['emotion_vectors'] * value
            loss_values_raw['emotion_vectors'] = value

        # key point location loss part
        if self.loss_weights['kp_location'] != 0:
            value = sum([
                functional.l1_loss(kp_driving[k], kp_driving_init[k]) for k in kp_driving_init.keys()
            ])
            loss_values['kp_location'] = self.loss_weights['kp_location'] * value
            loss_values_raw['kp_location'] = value

        return loss_values, loss_values_raw, generated
