# custom_model.py
import torch
from ResNet import ResNetModel  # Ganti dengan definisi model Anda

def get_custom_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetModel()
    model.load_state_dict(torch.load('/app/my_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device
