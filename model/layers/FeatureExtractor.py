import torch
import torch.nn as nn
import torchvision as tv


class FeatureExtractor(nn.Module):
    """Class is responsible for extracting deep features of a video frame (image)"""

    def __init__(self, deep_feature_model="googlenet", use_gpu=True):
        super(FeatureExtractor, self).__init__()
        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.set_model(deep_feature_model)

    def set_model(self, model_name):
        # alexnet, resnet50, resnet152, googlenet
        model_name = model_name.lower()
        if not hasattr(tv.models, model_name):
            print(f"Unsupported model {model_name}!")
            model_name = "googlenet"

        print(f"deep feature model: {model_name}")
        model = getattr(tv.models, model_name)(pretrained=True)
        # print(model)

        pool_index = -3 if model_name == "googlenet" else -2
        layers = list(model.children())[:pool_index + 1]
        # print(layers)

        self.model = nn.Sequential(*layers).float().eval().to(self._device)
        self.preprocess = tv.transforms.Compose([
            tv.transforms.Resize([224, 224]),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, frame):
        with torch.no_grad():
            frame = self.preprocess(frame)

            # add a dimension for batch
            batch = frame.unsqueeze(0).to(self._device)

            features = self.model(batch)
            features = features.squeeze()

            return features
