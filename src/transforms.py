# adv_skin_cancer/src/transforms.py
import torch
from torchvision import transforms
from .config import config

def get_transforms(split='train'):
    """
    Return transforms based on:
    1. Split (train/val)
    2. Model Type (Baseline vs Proposed) defined in config.BASE_MODEL
    """
    mean = config.IMAGENET_MEAN
    std = config.IMAGENET_STD
    img_size = config.IMAGE_SIZE

    # -----------------------------------------------------------
    # 1. Validation / Test Pipeline (เหมือนเดิมตลอด)
    # -----------------------------------------------------------
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if split != 'train':
        return val_transform

    # -----------------------------------------------------------
    # 2. Check Model Type form Config
    # -----------------------------------------------------------
    # ถ้าชื่อโมเดลมีคำว่า 'efficientnet' หรือ 'standard' -> ใช้ Standard Augmentation
    is_baseline = 'efficientnet' in config.BASE_MODEL or 'standard' in config.BASE_MODEL
    
    if is_baseline:
        print(f" Augmentation Strategy: STANDARD (For Baseline)")
        # --- Standard Augmentation (สำหรับ Baseline) ---
        # ใช้แค่การหมุนและ Flip พื้นฐาน ไม่แยกคลาส
        standard_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return standard_aug  # <--- ส่งกลับเป็นก้อนเดียว (dataset.py จะรู้เอง)

    else:
        print(f" Augmentation Strategy: CLASS-SPECIFIC (Light/Heavy for Proposed)")
        # -----------------------------------------------------------
        # 3. Class-Specific Augmentation (สำหรับ ResNet / Proposed)
        # -----------------------------------------------------------
        
        # Light: สำหรับ NV
        light_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Heavy: สำหรับ Melanoma/Others (ปรับลดความโหดลงตามแผนล่าสุด)
        heavy_aug = transforms.Compose([
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # ส่งกลับเป็น Dict (dataset.py จะแยกทำ Heavy/Light ให้เอง)
        return {
            'light': light_aug,
            'heavy': heavy_aug,
            'val': val_transform
        }