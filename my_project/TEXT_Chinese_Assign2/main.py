from torch.utils.data import DataLoader
from data_utils import load_dataset, clean_chinese_text
from dataset import FakeNewsDataset
from models import BERTCLIPFusionModel, TextOnlyModel, BERTCLIPLateFusionModel
from train import train_general
from evaluate import evaluate


def main():
    # File paths and root directory for images
    train_csv, test_csv, image_root = "df_train2.csv", "df_test2.csv", "images"

    # Load train and test datasets with image paths appended
    train_df = load_dataset(train_csv, image_root)
    test_df = load_dataset(test_csv, image_root)

    # Clean the text content by removing URLs and unwanted characters
    train_df['content'] = train_df['content'].astype(str).apply(clean_chinese_text)
    test_df['content'] = test_df['content'].astype(str).apply(clean_chinese_text)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # --- Multimodal model: combines BERT text and CLIP image features ---
    fusion_model = BERTCLIPFusionModel()
    preprocess = fusion_model.clip_preprocess  # CLIP image preprocessing function

    # Create datasets and dataloaders for training and testing
    train_dataset = FakeNewsDataset(train_df, preprocess)
    test_dataset = FakeNewsDataset(test_df, preprocess)

    # DataLoader with batch size 8 and custom collate function to unzip batch tuples
    loader_args = dict(batch_size=8, collate_fn=lambda x: list(zip(*x)))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    print("Training Multimodal...")
    train_general(fusion_model, train_loader, epochs=1, lr=1e-4, save_prefix="Multimodal")
    evaluate(fusion_model, test_loader, model_type="Multimodal")

    # --- Text-only model: uses only BERT embeddings for classification ---
    text_model = TextOnlyModel()
    print("Training Text-only...")
    train_general(text_model, train_loader, epochs=1, lr=1e-4, save_prefix="Text")
    evaluate(text_model, test_loader, model_type="Text")

    # --- Late Fusion model: separately classifies text and image then averages results ---
    latefusion_model = BERTCLIPLateFusionModel()
    print("Training LateFusion...")
    train_general(latefusion_model, train_loader, epochs=1, lr=1e-4, save_prefix="LateFusion")
    evaluate(latefusion_model, test_loader, model_type="LateFusion")


if __name__ == "__main__":
    main()
