# adv_skin_cancer/src/utils.py
import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import config  # Import config เข้ามาเพื่อใช้ในการคำนวณ weight

def seed_everything(seed=42):
    """
    Lock seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Global seed set to {seed}")

def make_stratified_split(df, test_size=0.15, val_size=0.15, seed=42):
    """
    Split dataset into train/val/test using 'lesion_id' to prevent data leakage.
    Ensures images from the same patient/lesion stay in the same split.
    """
    # 1. Group by lesion_id (one row per lesion) 
    # Use .first() to get the label for that lesion
    lesion_df = df.groupby('lesion_id')['dx'].first().reset_index()
    
    # 2. Separate Test Set
    # Stratify based on disease label to maintain class distribution
    train_val_lesions, test_lesions = train_test_split(
        lesion_df,
        test_size=test_size,
        stratify=lesion_df['dx'],
        random_state=seed
    )
    
    # 3. Separate Validation Set from Train Set
    # Adjust validation size relative to the remaining data
    adj_val_size = val_size / (1 - test_size)
    
    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=adj_val_size,
        stratify=train_val_lesions['dx'],
        random_state=seed
    )
    
    # 4. Create dictionary mapping
    split_map = {}
    for lid in train_lesions['lesion_id']: split_map[lid] = 'train'
    for lid in val_lesions['lesion_id']:   split_map[lid] = 'val'
    for lid in test_lesions['lesion_id']:  split_map[lid] = 'test'
    
    # 5. Map split column to original df 
    df['split'] = df['lesion_id'].map(split_map)
    
    # Check for unassigned rows (sanity check)
    if df['split'].isna().any():
        print(f" Warning: {df['split'].isna().sum()} images were not assigned to any split!")
    
    return df

def get_class_weights(df):
    """
    Calculate Inverse Frequency Class Weights for Focal Loss.
    Use this to set the 'alpha' parameter.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'dx' column (usually training set)
        
    Returns:
        torch.Tensor: Normalized weights for each class
    """
    # 1. Map labels to indices
    if 'target' not in df.columns:
         targets = df['dx'].map(config.CLASS_TO_IDX).values
    else:
         targets = df['target'].values # In case you pre-processed it
         
    # 2. Count samples per class
    # minlength ensures all classes are counted even if count is 0
    class_counts = np.bincount(targets, minlength=config.NUM_CLASSES)
    
    # Avoid division by zero (though unlikely in this dataset)
    class_counts = np.maximum(class_counts, 1) 
    
    # 3. Calculate Weights: Total / (Num_Classes * Count)
    # This formula balances the contribution of each class
    total_samples = len(targets)
    weights = total_samples / (config.NUM_CLASSES * class_counts)
    
    # 4. Convert to Tensor
    return torch.tensor(weights, dtype=torch.float32)

def calculate_accuracy(output, target):
    """
    Simple accuracy calculation helper.
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.max(output, 1)
        correct = (pred == target).sum().item()
        return correct / batch_size