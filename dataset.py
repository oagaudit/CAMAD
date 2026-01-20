# adv_skin_cancer/src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
from .config import config

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.class_to_idx = config.CLASS_TO_IDX

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Data
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_str = row['dx']
        
        # 2. Open Image
        image = Image.open(img_path).convert('RGB')
        
        # 3. Smart Augmentation Logic
        if self.transform:
            if isinstance(self.transform, dict):
                # ถ้าเป็น Training Mode ให้เลือก Transform ตามความยากง่ายของโรค
                if label_str in config.MAJORITY_CLASSES:
                    image = self.transform['light'](image)
                else:
                    image = self.transform['heavy'](image)
            else:
                # Validation/Test Mode
                image = self.transform(image)
        
        # 4. Convert Label
        if isinstance(label_str, str):
            label = self.class_to_idx[label_str]
        else:
            label = int(label_str)

        return image, torch.tensor(label, dtype=torch.long)

# --- NEW HELPER FUNCTION ---
def get_weighted_dataloader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    สร้าง DataLoader ที่มาพร้อมกับ WeightedRandomSampler
    เพื่อบังคับให้แต่ละ Batch มีรูป Melanoma และ NV มาในสัดส่วนเท่าๆ กัน
    """
    # 1. คำนวณ Weight สำหรับแต่ละรูปใน Dataset
    targets = dataset.df['dx'].map(dataset.class_to_idx).values
    class_counts = np.bincount(targets)
    
    # Weight ของคลาส = 1 / จำนวนรูปในคลาสนั้น (รูปน้อย Weight เยอะ)
    class_weights = 1. / class_counts
    
    # Map weight กลับไปที่แต่ละรูป
    sample_weights = class_weights[targets]
    
    # 2. สร้าง Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # อนุญาตให้สุ่มรูปเดิมซ้ำได้ (จำเป็นสำหรับคลาสเล็ก)
    )
    
    # 3. สร้าง DataLoader (ต้องปิด shuffle เพราะ sampler ทำหน้าที่สุ่มให้แล้ว)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return loader