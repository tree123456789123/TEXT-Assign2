import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import clip


class BERTCLIPFusionModelEn(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, hidden_dim=512):
        super(BERTCLIPFusionModelEn, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load pretrained BERT
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT

        # Load pretrained CLIP (ViT-B/32)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False  # Freeze CLIP visual encoder

        # Fusion classifier: concatenated BERT+CLIP -> MLP -> output 2 classes
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_list, image_list):
        # Encode text
        encoding = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_output = self.bert(**encoding)
        text_feat = text_output.last_hidden_state[:, 0, :]  # CLS token

        # Encode images
        image_tensors = torch.stack([self.clip_preprocess(img) for img in image_list]).to(self.device)
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image_tensors)

        # Concatenate features and classify
        fused = torch.cat((text_feat, image_feat), dim=1)
        output = self.classifier(fused)
        return output


class TextOnlyModelEn(nn.Module):
    def __init__(self):
        super(TextOnlyModelEn, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT

        # Text classifier using only BERT CLS token
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2)
        )

    def forward(self, text_list):
        # Ensure inputs are on the same device as the model
        encoding = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
        with torch.no_grad():
            output = self.bert(**encoding)
        cls_token = output.last_hidden_state[:, 0, :]  # Use CLS token
        return self.classifier(cls_token)
