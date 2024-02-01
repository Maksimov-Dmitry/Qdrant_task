from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import base64


class CustomDataset(Dataset):
    def __init__(self, path, transforms=None, is_openai=False):
        self.transform = transforms
        self.images = [image for image in Path(path).rglob('*.jpg')]
        self.image2idx = {str(image): idx for idx, image in enumerate(self.images)}
        self.is_openai = is_openai

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_openai:
            image = open(self.images[idx], "rb")
            image = base64.b64encode(image.read()).decode('utf-8')
        else:
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
        return image, idx
