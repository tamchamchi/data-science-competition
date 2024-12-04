from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIP():
     def __init__(self, model_name = 'openai/clip-vit-large-patch14'):
           self.processor = CLIPProcessor.from_pretrained(model_name)
           self.model = CLIPModel.from_pretrained(model_name)

     def extract_features(self, texts, images):

          images = [Image.open(image_path).convert("RGB") for image_path in images]
          
          inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
          
          with torch.no_grad():
               # Lấy đặc trưng hình ảnh và văn bản
               image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
               text_features = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

          image_features = image_features.detach().numpy()
          text_features = text_features.detach().numpy()
          return image_features, text_features