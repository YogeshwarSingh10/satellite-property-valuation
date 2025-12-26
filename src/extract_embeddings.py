import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from image_encoder import load_efficientnet
from image_dataset import ImageDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load frozen CNN
    model = load_efficientnet(device)

    # ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # intentional downsampling from 448 → 224
        transforms.ToTensor(),            # HWC [0–255] → CHW [0–1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])
    

    dataset = ImageDataset(
        image_dir="data/images",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    embeddings = []
    ids = []
    
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)
            feats = model(images)
    
            embeddings.append(feats.cpu())
            ids.extend(image_ids.numpy())
    

    embeddings = torch.cat(embeddings, dim=0).numpy()
    ids = np.array(ids)
    
    np.save("outputs/image_embeddings.npy", embeddings)
    np.save("outputs/image_ids.npy", ids)

    print(f"Saved embeddings: {embeddings.shape}")

if __name__ == "__main__":
    main()
