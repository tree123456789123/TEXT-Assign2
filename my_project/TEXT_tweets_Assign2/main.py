import torch
from torch.utils.data import DataLoader
from data_utils import load_dataset_en, clean_english_text
from dataset import FakeNewsDatasetEn
from models import BERTCLIPFusionModelEn, TextOnlyModelEn
from train import train_model, train_textonly_model
from evaluate import evaluate_model, evaluate_text_model

def main():
    # File paths
    train_csv = "df_train2.csv"
    test_csv = "df_test2.csv"
    image_root = "images"

    # Load and preprocess dataset
    train_df = load_dataset_en(train_csv, image_root)
    test_df = load_dataset_en(test_csv, image_root)

    # Clean English text
    train_df['content'] = train_df['content'].apply(clean_english_text)
    test_df['content'] = test_df['content'].apply(clean_english_text)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Initialize multimodal model
    model = BERTCLIPFusionModelEn()
    clip_preprocess = model.clip_preprocess  # CLIP preprocessing function

    # Dataset & DataLoader
    train_dataset = FakeNewsDatasetEn(train_df, clip_preprocess)
    test_dataset = FakeNewsDatasetEn(test_df, clip_preprocess)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    # Uncomment to train & evaluate multimodal model
    print("Training multimodal English model...")
    train_model(model, train_loader, epochs=5)
    print("Evaluating multimodal English model...")
    evaluate_model(model, test_loader)

    # Train & evaluate text-only model
    text_model = TextOnlyModelEn().to(model.device)
    print("Training English text-only model...")
    train_textonly_model(text_model, train_loader, epochs=8)

    print("Evaluating English text-only model...")
    evaluate_text_model(text_model, test_loader)

if __name__ == "__main__":
    main()
