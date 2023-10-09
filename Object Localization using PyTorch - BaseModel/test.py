import sys
import cv2
import torch
from torchvision import transforms as T
import numpy as np
import timm
import matplotlib.pyplot as plt

MODEL_NAME = 'efficientnet_b0'
NUM_COR = 4
IMG_SIZE = 140
DEVICE = 'cuda'

# Create the Object Localization model
class ObjLocModel(torch.nn.Module):
    def __init__(self):
        super(ObjLocModel, self).__init__()
        self.backbone = timm.create_model(model_name=MODEL_NAME, pretrained=False, num_classes=NUM_COR)

    def forward(self, images):
        return self.backbone(images)

model = ObjLocModel()

# Load the weights of the model
model.load_state_dict(torch.load('model_40EPOCHS.pt'))
model.to(device=DEVICE)
model.eval()

# Load and preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img, img_tensor

# Predict bounding boxes and draw them on the image
def predict_and_visualize(image_path):
    img, img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(img_tensor)
    predictions = predictions[0].cpu().numpy()

    xmin, ymin, xmax, ymax = predictions[:4]
    img_with_bbox = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, ), 4)

    plt.imshow(img_with_bbox)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
    else:
        image_path = sys.argv[1]
        predict_and_visualize(image_path)
