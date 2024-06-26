import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl



# util function
def visualize_batch(data_loader, class_names):
    images, labels = next(iter(data_loader))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].permute(1, 2, 0).numpy()  # Convert to HWC format
            ax.imshow(img)
            label = class_names[labels[i]]
            ax.set_title(f'Class {str(label)}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()





class FundusDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4,transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]) # C x H x W
        else: self.transform =transform

    def setup(self, stage=None):
        self.train_dataset = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
        self.val_dataset = ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.transform)
        self.test_dataset = ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.transform)
       
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
       
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
       
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



if __name__ == "__main__":
    data_dir = '/Users/leo/Desktop/MAE-Fundus/data/EAMDR'
    data_module = FundusDataModule(data_dir)

    # Set up the datasets
    data_module.setup()

    # Get class names from the dataset
    class_names = data_module.train_dataset.classes

    # Test the train dataloader
    train_loader = data_module.train_dataloader()
    print("Train DataLoader length:", len(train_loader))
    visualize_batch(train_loader, class_names)

    # Test the val dataloader
    val_loader = data_module.val_dataloader()
    print("Validation DataLoader length:", len(val_loader))
    visualize_batch(val_loader, class_names)

    # Test the test dataloader
    test_loader = data_module.test_dataloader()
    print("Test DataLoader length:", len(test_loader))
    visualize_batch(test_loader, class_names)

