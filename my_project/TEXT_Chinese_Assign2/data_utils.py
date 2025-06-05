import pandas as pd
import os
import re


# Load dataset from CSV and append image path
def load_dataset(csv_path, image_root_dir):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image'].apply(lambda x: os.path.join(image_root_dir, x))
    return df[['content', 'image_path', 'label']]


# Clean Chinese text by removing URLs and non-Chinese punctuation
def clean_chinese_text(text):
    text = re.sub(r'https?://[a-zA-Z0-9./]+', '', text)  # remove URLs
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9。，、！？：；“”‘’（）【】《》…\s]', '', text)  # keep Chinese chars & punctuation
    return re.sub(r'\s+', '', text)  # remove extra whitespace
