from flask import Flask, request, jsonify
from LoadResNet import get_custom_model
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests

app = Flask(__name__)

# Load the ResNet model
model, device = get_custom_model()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
with open('imagenet_classes.txt') as f:
    idx_to_labels = [line.strip() for line in f.readlines()]

@app.route('/api', methods=['POST'])
def process_input():
    image_file = request.files.get('image')
    text_input = request.form.get('text', '')

    # Process the image
    image = Image.open(image_file).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        class_label = idx_to_labels[class_idx]

    # Menyiapkan prompt untuk diproses oleh Model NLP
    prompt = f"{text_input}\nThe image is classified as: {class_label}. Provide a detailed description."

    # Berinteraksi langsung dengan Model melalui sistem post.
    response = requests.post(
        'http://localhost:11434/generate',
        json={'model': 'llama-3.1', 'prompt': prompt}
    )
    generated_text = response.json().get('response', '')

    return jsonify({
        'classification': class_label,
        'generated_text': generated_text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
