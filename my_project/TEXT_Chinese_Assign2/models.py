import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import clip


class BERTCLIPFusionModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, hidden_dim=512):
        super(BERTCLIPFusionModel, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load pre-trained BERT model and tokenizer for Chinese text
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        # Freeze BERT parameters to avoid training
        for param in self.bert.parameters():
            param.requires_grad = False

        # Load pre-trained CLIP model and image preprocess function
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        # Freeze CLIP visual encoder parameters
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False

        # Fusion classifier combining text and image features
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),  # Concatenate text and image features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Binary classification output
        )

    def forward(self, text_list, image_list):
        device = self.device

        # Tokenize and encode text input
        encoding = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_output = self.bert(**encoding)
        # Use [CLS] token representation as text feature
        text_feat = text_output.last_hidden_state[:, 0, :]

        # Preprocess and encode images using CLIP
        image_tensors = torch.stack([self.clip_preprocess(img) for img in image_list]).to(device)
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image_tensors)

        # Concatenate text and image features for fusion
        fused = torch.cat((text_feat, image_feat), dim=1)
        # Pass through classifier to get logits
        output = self.classifier(fused)
        return output


class TextOnlyModel(nn.Module):
    def __init__(self):
        super(TextOnlyModel, self).__init__()
        # Load BERT tokenizer and model for Chinese text
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Classifier for text features only
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification
        )

    def forward(self, text_list):
        # Tokenize and encode text input
        encoding = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
        with torch.no_grad():
            output = self.bert(**encoding)
        # Use [CLS] token feature for classification
        cls_token = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


class BERTCLIPLateFusionModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, hidden_dim=512):
        super(BERTCLIPLateFusionModel, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Initialize BERT tokenizer and model for text encoding
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Text-specific classifier
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

        # Load CLIP model and preprocessing for image encoding
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        # Freeze CLIP visual encoder parameters
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False

        # Image-specific classifier
        self.image_classifier = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_list, image_list):
        device = self.device

        # Encode text
        encoding = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_output = self.bert(**encoding)
        text_feat = text_output.last_hidden_state[:, 0, :]
        text_logits = self.text_classifier(text_feat)  # Text classifier logits

        # Encode images
        image_tensors = torch.stack([self.clip_preprocess(img) for img in image_list]).to(device)
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image_tensors).float()
        image_logits = self.image_classifier(image_feat)  # Image classifier logits

        # Late fusion by averaging logits from text and image classifiers
        fused_logits = (text_logits + image_logits) / 2
        return fused_logits
