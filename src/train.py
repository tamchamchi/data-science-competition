from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import joblib
import os
import torch
import json
import numpy as np
import pandas as pd
from models.vit5 import ViT5
from models.clip import CLIP
from models.vilt import ViLT


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
                    'caption': value.get('caption'),
               })
               data_id += 1

          return combined_data


def train():
     train_data = CustomDataset(image_data_dir=r'F:\Năm 3 - HK1\data-science-competition\data\Data_train\training-images\train-images', data_json_path=r'F:\Năm 3 - HK1\data-science-competition\data\Data_train\vimmsd-train.json')
     train_data = train_data.get_data()

     texts_train = [item['caption'] for item in train_data]
     labels_train = [item['label'] for item in train_data]
     id_train = [item['id'] for item in train_data]
     images_train = [item['image'] for item in train_data]
     
     clip_model = CLIP()
     vilt_model = ViLT()
     vit5_model = ViT5()

     clip_image_features, clip_text_features = clip_model.extract_features(texts=texts_train, images=images_train)
     vilt_features = vilt_model.extract_features(texts=texts_train, images=images_train)
     vit5_features = vit5_model.extract_features(texts=texts_train)

     X_train = np.concatenate([clip_image_features, clip_text_features, vilt_features, vit5_features], axis=1)
     
     df_train = pd.DataFrame(X_train)
     df_target = pd.DataFrame(labels_train, columns=['label'])

     print(f"Feature shape: {df_train.shape}")
     print(f"Target shape: {df_target.shape}")

     X_train, X_val, y_train, y_val = train_test_split(df_train, df_target, test_size=0.2, random_state=42)
     print(f"Training data shape: {X_train.shape}")
     print(f"Validation data shape: {X_val.shape}")
     print(f"Training label shape: {y_train.shape}")
     print(f"Validation label shape: {y_val.shape}")

     smote = smote = SMOTE(sampling_strategy='auto',random_state=42)

     # Tăng cường dữ liệu lớp thiểu số
     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

     # Bước 1: Chuẩn hóa các đặc trưng
     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train_resampled)

     # Bước 2: Áp dụng PCA
     pca = PCA(n_components=0.75)  # Giữ lại 95% phương sai
     X_train_pca = pca.fit_transform(X_train_scaled)

     # Kiểm tra số lượng thành phần chính
     print(f"Number of principal components: {pca.n_components_}")

     # Nếu bạn có tập validation, hãy chuẩn hóa và biến đổi chúng cũng như vậy
     X_val_scaled = scaler.transform(X_val)
     X_val_pca = pca.transform(X_val_scaled)

     # Kiểm tra tỷ lệ phương sai được giữ lại
     print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")

     # Bước 3: Thiết lập mô hình SVM với các tham số cụ thể
     svm_model = SVC(C=1, kernel='rbf', gamma='scale')

     # Bước 4: Huấn luyện mô hình trên tập dữ liệu huấn luyện
     svm_model.fit(X_train_pca, y_train_resampled)

     # Bước 5: Đánh giá mô hình trên tập validation
     val_predictions = svm_model.predict(X_val_pca)
     accuracy = accuracy_score(y_val, val_predictions)
     report = classification_report(y_val, val_predictions, target_names=["not-sarcasm", "text-sarcasm", "image-sarcasm", "multi-sarcasm"])

     # Bước 6: In kết quả
     print(f"Validation Accuracy: {accuracy * 100:.2f}%")
     print("Classification Report:")
     print(report)

     # Bước 7: Lưu mô hình
     joblib.dump(svm_model, 'best_svm_model.pkl')
     joblib.dump(scaler, 'scaler.pkl')
     joblib.dump(pca, 'pca.pkl')


if __name__ == '__main__':
     train()
          

     