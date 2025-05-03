import torch
import torch.nn as nn
import torchvision.models as models

# 개별 모델 정의: pretrained 모델을 feature extractor로 사용
class CNNEnsemble(nn.Module):
    def __init__(self, num_classes):
        super(CNNEnsemble, self).__init__()

        # Feature Extractors
        self.resnet = models.resnet50(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)
        self.vgg = models.vgg16(pretrained=True)

        # FC layers 제거하고 feature만 추출
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])   # Output: (batch, 2048)
        self.densenet = nn.Sequential(*list(self.densenet.features))      # Output: (batch, 1024, 7, 7)
        self.vgg = nn.Sequential(*list(self.vgg.features))                # Output: (batch, 512, 7, 7)

        # 공통 Fully Connected layer
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 1024 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet
        res_out = self.resnet(x)
        res_out = res_out.view(res_out.size(0), -1)

        # DenseNet
        densenet_out = self.densenet(x)
        densenet_out = nn.functional.adaptive_avg_pool2d(densenet_out, (1, 1))
        densenet_out = densenet_out.view(densenet_out.size(0), -1)

        # VGG
        vgg_out = self.vgg(x)
        vgg_out = nn.functional.adaptive_avg_pool2d(vgg_out, (1, 1))
        vgg_out = vgg_out.view(vgg_out.size(0), -1)

        # Concatenate
        combined = torch.cat([res_out, densenet_out, vgg_out], dim=1)
        out = self.classifier(combined)

        return out
