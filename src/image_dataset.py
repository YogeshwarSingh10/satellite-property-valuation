import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # extract ID from filename (e.g. "123456.png" → 123456)
        image_id = int(os.path.splitext(filename)[0])

        return image, image_id


