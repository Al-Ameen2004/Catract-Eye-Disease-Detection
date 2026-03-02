import torch
import torch.nn as nn
import torchvision.models as models
import timm

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.classifier = nn.Linear(512 + 192, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)

        combined = torch.cat((cnn_features, vit_features), dim=1)
        output = self.classifier(combined)

        return output