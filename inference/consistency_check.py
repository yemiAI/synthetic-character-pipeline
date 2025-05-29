from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

class ConsistencyChecker:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def check(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to("cuda")
        features = self.model.get_image_features(**inputs)
        normed = torch.nn.functional.normalize(features, dim=-1)
        return torch.mm(normed, normed.T)
