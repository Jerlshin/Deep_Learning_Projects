import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms as T

import sys
import os

import timm
from PIL import Image
DEVICE = 'cuda'
MODEL_NAME = 'efficientnet_b0'

class FaceModel(torch.nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model(MODEL_NAME, pretrained=True, num_classes=7)
    
    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits


def preprocess_image(image_path, transformer):
    image = Image.open(image_path)
    preprocessed_image = transformer(image).unsqueeze(0).to(DEVICE)
    return preprocessed_image

def predict_expression(image_path):
    valid_augs = T.Compose([T.ToTensor()])
    preprocessed_image = preprocess_image(image_path, valid_augs)

    model = FaceModel()
    model.load_state_dict(torch.load('best-weights.pt'))
    model.eval()
    model.to(device=DEVICE)
    
    with torch.no_grad():
        logits = model(preprocessed_image)
        prob = torch.softmax(logits, dim=1)
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_class_idx = torch.argmax(prob)
    predicted_class = classes[predicted_class_idx]
    
    return predicted_class

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_expression.py <image_path>")
    else:
        image_path = sys.argv[1]
        predicted_expression = predict_expression(image_path)
        print("Predicted Expression:", predicted_expression)