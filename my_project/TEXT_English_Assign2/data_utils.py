import pandas as pd
import os
import re


def load_dataset_en(csv_path, image_root_dir):
    df = pd.read_csv(csv_path)

    # Extract English text and corresponding image path
    df['content'] = df['post_text'].astype(str)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_root_dir, f"{x}.jpg"))

    # Map labels from 'fake'/'real' to 0/1
    df['label'] = df['label'].map({'fake': 0, 'real': 1})

    return df[['content', 'image_path', 'label']]


def clean_english_text(text):
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text
