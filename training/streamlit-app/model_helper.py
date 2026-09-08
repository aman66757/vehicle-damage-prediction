import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal"
]


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class CarClassifierResNet(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.model = models.resnet50(weights=None)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = CarClassifierResNet(num_classes=6)


# Find saved_model.pth
# model_helper.py is inside:
# damage-prediction/streamlit-app/
#
# saved_model.pth is inside:
# damage-prediction/

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_model.pth"


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


# Load trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

trained_model.load_state_dict(state_dict)

trained_model.to(device)
trained_model.eval()


# --------------------------------------------------
# IMAGE TRANSFORMATION
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(image):

    # If image is a file path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # If image is already a PIL image
    else:
        image = image.convert("RGB")

    # Transform image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to CPU/GPU
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        output = trained_model(image_tensor)

        # Convert output to probabilities
        probabilities = torch.softmax(output, dim=1)

        # Get highest probability
        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()
    confidence_value = confidence.item()

    predicted_name = class_names[predicted_index]

    return predicted_name, confidence_value, probabilities[0].cpu().tolist()