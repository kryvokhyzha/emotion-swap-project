from torchvision.models import resnet18
from torch import nn
from config import opt


class EmotionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(EmotionModel, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        self.model.fc = nn.Linear(self.model.fc.in_features, len(opt.emotion_list), bias=True)

    def forward(self, x):
        return self.model(x)

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_middle_layers(self):
        self.freeze_all_layers()

        for param in self.model.fc.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
