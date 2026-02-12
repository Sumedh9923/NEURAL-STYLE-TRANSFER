import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Image Loader
# -----------------------------
def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# -----------------------------
# Gram Matrix Function
# -----------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# -----------------------------
# Feature Extraction
# -----------------------------
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # Content layer
        '28': 'conv5_1'
    }

    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# -----------------------------
# Load Images
# -----------------------------
content = load_image("content.jpg")
style = load_image("style.jpg")

# -----------------------------
# Load Pretrained VGG19
# -----------------------------
weights = VGG19_Weights.DEFAULT
vgg = models.vgg19(weights=weights).features.to(device).eval()

# Freeze VGG parameters
for param in vgg.parameters():
    param.requires_grad = False

# -----------------------------
# Extract Features
# -----------------------------
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# 🔥 CRITICAL FIX: DETACH FEATURES
for key in content_features:
    content_features[key] = content_features[key].detach()

for key in style_features:
    style_features[key] = style_features[key].detach()

# Compute style Gram matrices (detached)
style_grams = {
    layer: gram_matrix(style_features[layer]).detach()
    for layer in style_features
}

# -----------------------------
# Create Target Image
# -----------------------------
target = content.clone().requires_grad_(True).to(device)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.Adam([target], lr=0.003)

# -----------------------------
# Loss Weights
# -----------------------------
content_weight = 1e4
style_weight = 1e2

# -----------------------------
# Training Loop
# -----------------------------
steps = 200

for step in range(steps):

    target_features = get_features(target, vgg)

    # Content Loss
    content_loss = torch.mean(
        (target_features['conv4_2'] - content_features['conv4_2']) ** 2
    )

    # Style Loss
    style_loss = 0

    for layer in style_grams:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)

        style_gram = style_grams[layer]

        layer_loss = torch.mean((target_gram - style_gram) ** 2)

        b, c, h, w = target_feature.shape
        style_loss += layer_loss / (c * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item()}")

# -----------------------------
# Save Output Image
# -----------------------------
final_img = target.clone().detach().cpu().squeeze()
final_img = transforms.ToPILImage()(final_img)

final_img.save("styled_output.jpg")

print("\n✅ Style Transfer Complete! Image saved as styled_output.jpg")

# Optional Display
plt.imshow(final_img)
plt.axis("off")
plt.show()