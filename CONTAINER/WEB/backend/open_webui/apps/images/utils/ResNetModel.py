import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

class ResNetModelInference:
    def __init__(self, num_classes=1000, pretrained=True):
        self.model = models.resnet50(pretrained=pretrained)
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.eval()  # Set model to evaluation mode
        
        # Preprocessing transformations for the input image
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(img)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

    def predict_image(self, image):
        # This function processes image in PIL format
        img = image.convert('RGB')
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
