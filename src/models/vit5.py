from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class ViT5():
     def __init__(self, model_name = 'VietAI/vit5-base'):
         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
     
     def extract_features(self, texts):
         inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
         with torch.no_grad():
          outputs = self.model.encoder(**inputs)
         cls_token = outputs.last_hidden_state[:, 0, :].detach().numpy()
         return cls_token
