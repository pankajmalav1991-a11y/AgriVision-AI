import torch
import torch.nn as nn
import gradio as gr

from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# --------------------------------------------------
# Device Configuration
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# Load Checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    "model/agrivision_vit_model.pth",
    map_location=device
)

class_names = checkpoint["class_names"]

# --------------------------------------------------
# Recreate ViT Architecture
# --------------------------------------------------

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

model.classifier = nn.Linear(
    model.config.hidden_size,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_disease(image):

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[
        prediction.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    return (
        predicted_class,
        f"{confidence_score:.2f}%"
    )

# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

app = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="🌿 AgriVision AI",
    description=(
        "Plant Disease Classification using "
        "Vision Transformers (ViT)"
    ),
    examples=[
        ["sample_images/potato_early_blight.png"],
        ["sample_images/potato_late_blight.png"],
        ["sample_images/tomato_early_blight.png"],
        ["sample_images/tomato_healthy.png"]
    ]
)

if __name__ == "__main__":
    app.launch()