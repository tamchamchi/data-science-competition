import json
import os
from src.utils.preprocessing import clean_caption

class CustomDataset():
     def __init__(self, image_data_dir, data_json_path):
          self.data_json_path = data_json_path
          self.image_data_dir = image_data_dir
          self.label_mapping = {
               "not-sarcasm": 0,
               "text-sarcasm": 1,
               "image-sarcasm": 2,
               "multi-sarcasm": 3,
          }
     
     def load_data(self):
          with open(self.data_json_path, 'r', encoding='utf-8') as file:
               data = json.load(file)
          return data

     
     def get_data(self):
          data_id = 0
          combined_data = []

          for key, value in self.load_data().items():
               image_name = value['image']
               image_path = os.path.join(self.image_data_dir, image_name)

               label = value.get('label')
               combined_data.append({
                    'id': data_id,
                    'image': image_path,
                    'label': self.label_mapping[label],
                    'caption': clean_caption(value.get('caption')),
               })
               data_id += 1

          return combined_data

