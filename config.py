# adv_skin_cancer/src/config.py
import torch
import os
from pathlib import Path

class Config:
# ========== Project Paths ==========
    # Path main
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    
    # Data structure paths
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed" 
    METRIC_DIR = PROJECT_ROOT / 'reports' / 'metrics'
    FIGURE_DIR = PROJECT_ROOT / 'reports' / 'figures'
    
    # Image folders
    IMAGE_DIR_PART1 = RAW_DATA_DIR / "HAM10000_images_part_1"
    IMAGE_DIR_PART2 = RAW_DATA_DIR / "HAM10000_images_part_2"
    METADATA_PATH = RAW_DATA_DIR / "HAM10000_metadata.csv"

    # Outputs
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    RESULT_DIR = PROJECT_ROOT / "results"
    
# ========== Data Settings ==========
    SEED = 42
    NUM_CLASSES = 7
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Mapping
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
    
    # Split Ratios (Stratified by Lesion ID)
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
# ========== Image Processing ==========
    IMAGE_SIZE = 224           # ResNet Standard
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
# ========== Model Architecture ==========
    # ====== EfficientNet baseline ======
    BASE_MODEL = "efficientnet_b0"
    USE_CBAM = False            
    # ====== ResNet50 + CBAM =======
    #BASE_MODEL = "resnet50"
    #USE_CBAM = True            
    # ====== resnet50_vanilla =======
    #BASE_MODEL = "resnet50_vanilla"
    #USE_CBAM = False  
    # ====== resnet50_tuned =======
    #BASE_MODEL = "resnet50_cbam_tuned"
    #USE_CBAM = True  
    
    PRETRAINED = True
    
# ========== Training Hyperparameters ==========
    BATCH_SIZE = 32
    NUM_WORKERS = 4           
    LEARNING_RATE = 1e-4
    #LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 3e-5
    NUM_EPOCHS = 50
    
    # Transfer Learning Strategy
    WARMUP_EPOCHS = 5          # Freeze backbone
    FINETUNE_EPOCHS = 45       # Unfreeze all
    TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS
    
    EARLY_STOPPING_PATIENCE = 10
    
# ========== Imbalance Handling ==========
    LOSS_TYPE = "focal"        # ResNet50
    #LOSS_TYPE = "ce"        # efficientnet_b0    
    FOCAL_GAMMA = 2.0
    #FOCAL_GAMMA = 3.0
    
# ========== Imbalance Setting ==========
    MINORITY_CLASSES = ['mel', 'bcc', 'bkl', 'akiec', 'vasc', 'df']
    MAJORITY_CLASSES = ['nv']
    
# ========== Hardware ==========
    @property
    def DEVICE(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')  # For Mac M1/M2/M3
        else:
            return torch.device('cpu')
            
# ========== Setup Helper ==========
    @classmethod
    def setup(cls):
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR, cls.CHECKPOINT_DIR, cls.LOG_DIR, cls.RESULT_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        print(f" Directories checked/created at {cls.PROJECT_ROOT}")

# Create instance
config = Config()

if __name__ == "__main__":
    # Test run
    config.setup()
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Device: {config.DEVICE}")
    print(f"Part 1 exists: {config.IMAGE_DIR_PART1.exists()}")
    print(f"Part 2 exists: {config.IMAGE_DIR_PART2.exists()}")