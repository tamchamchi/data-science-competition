from transformers import ViltProcessor, ViltModel
from PIL import Image
import torch

class ViLT():
     def __init__(self, model_name = 'dandelin/vilt-b32-mlm'):
           self.processor = ViltProcessor.from_pretrained(model_name)
           self.model = ViltModel.from_pretrained(model_name)

     def extract_features(self, texts, images):

          images = [Image.open(image_path).convert("RGB") for image_path in images]
          
          inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
          
          with torch.no_grad():
               # Lấy đặc trưng hình ảnh và văn bản
               outputs = self.model(**inputs)

          cls_token = outputs.last_hidden_state[:, 0, :].detach().numpy()
          return cls_token