import re
import emoji

def clean_caption(caption):
     # Chuyển đổi caption sang chữ thường
     caption = caption.lower()

     # Loại bỏ hashtag
     caption = re.sub(r'#\S+', '', caption)  # Loại bỏ hashtag và các từ liền theo sau

     # Loại bỏ URL
     caption = re.sub(r'http\S+|www\S+', '', caption)  # Loại bỏ URL

     # # Chuyển đổi emoji thành dạng văn bản
     caption = emoji.demojize(caption)
     
     return caption.strip() 