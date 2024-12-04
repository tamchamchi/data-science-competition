from src.models.clip import CLIP
from src.models.vilt import ViLT
from src.models.vit5 import ViT5
import joblib
import numpy as np


def predict(text, image):
     # Load label mapping
     label_mapping = {
          0: "not-sarcasm",
          1: "text-sarcasm",
          2: "image-sarcasm",
          3: "multi-sarcasm",
     }
     # Load models
     clip_model = CLIP()
     vilt_model = ViLT()
     vit5_model = ViT5()
     
     # Extract features
     image_features_clip, text_features_clip = clip_model.extract_features(texts=[text], images=[image])
     features_vilt = vilt_model.extract_features(texts=[text], images=[image])
     features_vit5 = vit5_model.extract_features(texts=[text])

     X_test = np.concatenate([image_features_clip, text_features_clip, features_vilt, features_vit5], axis=1)

     model = joblib.load('best_svm_model.pkl')
     scaler = joblib.load('scaler.pkl')
     pca = joblib.load('pca.pkl')

     X_test = scaler.transform(X_test)
     X_test = pca.transform(X_test)

     test_predictions = model.predict(X_test)

     return label_mapping[test_predictions[0]]



     
     