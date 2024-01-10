import torch
import clip
from PIL import Image
import requests

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare the image
image_url = "https://example.com/image.jpg"  # Replace with your image URL
image_response = requests.get(image_url)
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = preprocess(image).unsqueeze(0).to(device)

# Prepare the text
text = clip.tokenize(["a description of the image"]).to(device)

# Encode image and text
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# You can now use the encoded features for various tasks
# For example, calculating the similarity between the image and the text
similarity = (image_features @ text_features.T).softmax(dim=-1)

print("Similarity:", similarity.item())