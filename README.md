# NEURAL-STYLE-TRANSFER
# COMPANY - CODETECH IT SOLUTIONS
# NAME: SUMEDH NARAYAN PATIL
# INTERN ID: CTIS3852
# DOMAIN: ARTIFICIAL INTELLIGENCE
# DURATION: 4 WEEKS
# MENTOR: NEELA SANTHOSH KUMAR

# DESCRIPTION:
## 🎨 Neural Style Transfer using PyTorch
Transforming ordinary photographs into artistic masterpieces using Deep Learning.

### 🚀 Project Overview
This project implements Neural Style Transfer (NST) using a pre-trained VGG19 Convolutional Neural Network in PyTorch.

Neural Style Transfer blends:
📷 The content of one image
🎨 The style of another image

To generate a new image that artistically combines both.
This implementation uses optimization-based style transfer, where the target image is iteratively updated to minimize content and style loss.

### 🧠 How It Works

The model:
Loads a pre-trained VGG19 network

Extracts:
Content features (from deeper layers)
Style features (from multiple convolution layers)

Computes:
Content Loss (preserves structure)
Style Loss (using Gram Matrix)
Optimizes the target image using backpropagation

### 🛠 Technologies Used
Python 3.11.1
PyTorch
Torchvision
PIL (Pillow)
Matplotlib

### ⚙️ Installation
Clone the repository:
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer

Install dependencies:
pip install torch torchvision pillow matplotlib

### ▶️ How to Run

Place your images:
content.jpg
style.jpg

Then run:
python neural_style_transfer.py

After training completes:
styled_output.jpg
will be generated in the project folder.

### 📸 Example
🖼 Content Image

A real-world photograph.
🎨 Style Image
An artistic painting (e.g., Van Gogh style).
✨ Output
A stylized image combining both.

🧮 Loss Functions Used
📌 Content Loss

Mean Squared Error between target and content feature maps.
📌 Style Loss

Computed using Gram Matrix to capture texture and artistic patterns.
📌 Total Loss
Total Loss = α(Content Loss) + β(Style Loss)

Where:
α controls structure preservation
β controls artistic intensity

### 📊 Model Details

Backbone: VGG19 (Pretrained on ImageNet)
Optimization: Adam Optimizer
Default Steps: 300 or 200
Image Size: 512px (adjustable)

### ⚡ Performance Notes
CPU training may take 5–20 minutes
GPU significantly improves speed
Reduce steps or image size for faster execution

### 🔥 Key Learning Outcomes

✔ Understanding CNN feature extraction
✔ Gram Matrix implementation
✔ Backpropagation through images
✔ Optimization-based image transformation
✔ Handling PyTorch computational graphs

### 🌟 Future Improvements

Fast Neural Style Transfer (real-time)
Web App using Gradio
Streamlit deployment
GPU optimization
Multiple style blending
Video style transfer


### 📌 Why This Project Matters

Neural Style Transfer demonstrates:
Power of deep feature representations
Practical applications of CNNs beyond classification
Creative AI in computer vision

This project showcases real-world implementation of deep learning concepts suitable for AI internships and portfolio presentation.

### Output:

![Image](https://github.com/user-attachments/assets/5dd90ab8-c87c-4646-8e33-9d275fbe9ed2)

[Output.txt](https://github.com/user-attachments/files/25255420/Output.txt)


