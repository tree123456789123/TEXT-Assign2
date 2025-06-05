from torch.utils.data import Dataset
from PIL import Image


# Custom PyTorch Dataset for fake news multimodal data
class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, clip_preprocess):
        self.data = dataframe
        self.clip_preprocess = clip_preprocess  # preprocessing function for images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['content']
        label = int(row['label'])

        # Attempt to open image; create blank if missing
        try:
            image = Image.open(row['image_path']).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), color='white')

        return text, image, label
