# Multimodal Sarcasm Detection on Vietnamese Social Media Texts

## Problem Description
- **Input**: An image-text dataset collected from social media platforms.  
- **Output**: A classification into one of four labels:
  1. **not-sarcasm**
  2. **text-sarcasm**
  3. **image-sarcasm**
  4. **multi-sarcasm**

## Preprocessing
1. Convert all text in captions to lowercase.  
2. Remove hashtags and URLs.  
3. Transform emojis into textual descriptions using the `emoji` library.

## Proposed Method
1. Use three pre-trained models to extract features:
   - **CLIP**
   - **ViT5**
   - **ViLT**
2. Apply the **early-fusion** method to concatenate features from the image and text modalities.  
3. Train an **SVM** model for sarcasm detection.  

## Results
![Detection Results](./resource/Screenshot%202024-12-04%20163423.png)  
*Description*: The results showcase the performance of our approach in detecting sarcasm across the four defined labels.

---

## Demo with Streamlit

### Requirements
1. **Python Environment Setup**
   - Ensure you have Python 3.11.9 installed on your machine.

2. **Create and Activate a Virtual Environment**
     # Create a virtual environment
     `python -m venv venv`

     # Activate the virtual environment
     ```
          # On Windows:
          venv\Scripts\activate

          # On macOS/Linux:
          source venv/bin/activate
     ```
     # Install Required Libraries
     ```
          # Upgrade pip
          pip install --upgrade pip

          # Install required libraries
          pip install -r requirements.txt
     ```
     # Run the Streamlit Application
     ```
          streamlit run main.py
     ```
     

