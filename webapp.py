import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Generator class (must match training code)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(110, 256),   # same architecture as training
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        return self.model(x)


# One-hot encoding
def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Load model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("fast_generator.pth", map_location="cpu"))
    model.eval()
    return model

# Generate digit images
def generate_images(model, digit, num_images=5):
    z = torch.randn(num_images, 100)
    labels = torch.tensor([digit] * num_images)
    labels_oh = one_hot(labels)
    with torch.no_grad():
        generated = model(z, labels_oh).view(-1, 28, 28)
    return (generated + 1) / 2  # Scale to [0,1]

# Streamlit UI
st.title("🧠 Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    model = load_model()
    images = generate_images(model, digit)

    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        fig, ax = plt.subplots()
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        cols[i].pyplot(fig)
