import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

LABELS = [
    'No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
    'Lung Opacity','Lung Lesion','Edema','Consolidation',
    'Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion',
    'Pleural Other','Fracture','Support Devices'
]

class CheXpertDataset(Dataset):
    def __init__(self,csv_path,img_root_dir,transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir
        self.transform = transform

        for col in LABELS:
            self.df[col] = self.df[col].fillna(0).replace(-1,0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root_dir, row['Path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor([row[col] for col in LABELS], dtype = torch.float32)

        return image,labels