# adv_skin_cancer/src/config.py
import torch
from pathlib import Path

class Config:
    # ========== 1. Project Paths (ค่าคงที่) ==========
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed" 
    METRIC_DIR = PROJECT_ROOT / 'reports' / 'metrics'
    FIGURE_DIR = PROJECT_ROOT / 'reports' / 'figures'
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    RESULT_DIR = PROJECT_ROOT / "results"
    IMAGE_DIR_PART1 = RAW_DATA_DIR / "HAM10000_images_part_1"
    IMAGE_DIR_PART2 = RAW_DATA_DIR / "HAM10000_images_part_2"
    METADATA_PATH = RAW_DATA_DIR / "HAM10000_metadata.csv"

    # ========== 2. Data & Hardware Settings (ค่าคงที่) ==========
    SEED = 42
    NUM_CLASSES = 7
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    IMAGE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    NUM_WORKERS = 0 # แนะนำเป็น 0 สำหรับ Mac
    PRETRAINED = True

    # ========== 3. การเลือกโมเดล (แก้ที่นี่จุดเดียว) ==========
    # เลือก: 'efficientnet_b0', 'resnet50_vanilla', 'resnet50_cbam'
    SELECTED_EXP = 'resnet50_cbam' 

    def __init__(self):
        """
        ตั้งค่าพารามิเตอร์เฉพาะสำหรับแต่ละโมเดลโดยอัตโนมัติ
        """ 
        # --- กองกลางที่ใช้เหมือนกันเบื้องต้น ---
        self.WEIGHT_DECAY = 3e-5
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 50
        self.WARMUP_EPOCHS = 5
        self.FINETUNE_EPOCHS = 45
        self.EARLY_STOPPING_PATIENCE = 10
        self.MINORITY_CLASSES = ['mel', 'bcc', 'bkl', 'akiec', 'vasc', 'df']
        self.MAJORITY_CLASSES = ['nv']

        # --- แยกตามการทดลอง ---
        if self.SELECTED_EXP == 'efficientnet_b0':
            self.BASE_MODEL = "efficientnet_b0"
            self.USE_CBAM = False
            self.LOSS_TYPE = "ce"
            self.AUG_STRATEGY = "standard"
            self.USE_WEIGHTED_SAMPLER = False
            self.LEARNING_RATE = 1e-4

        elif self.SELECTED_EXP == 'resnet50_vanilla':
            self.BASE_MODEL = "resnet50"
            self.USE_CBAM = False
            self.LOSS_TYPE = "ce"
            self.AUG_STRATEGY = "standard"
            self.USE_WEIGHTED_SAMPLER = False
            self.LEARNING_RATE = 1e-4

        elif self.SELECTED_EXP == 'resnet50_cbam':
            # นี่คือโมเดลที่นำเสนอในเปเปอร์ (Proposed CAMAD)
            self.BASE_MODEL = "resnet50"
            self.USE_CBAM = True
            self.LOSS_TYPE = "focal"
            self.AUG_STRATEGY = "class-specific"
            self.USE_WEIGHTED_SAMPLER = True
            self.LEARNING_RATE = 1e-4
            #self.FOCAL_GAMMA = 2.5
            # --- Tuning for False Negative Reduction ---
            self.FOCAL_GAMMA = 2.0
            self.USE_MALIGNANT_BOOST = True            
            self.MALIGNANT_BOOST_FACTOR = 1.3

    @property
    def DEVICE(self):
        if torch.cuda.is_available(): return torch.device('cuda')
        elif torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')

    @classmethod
    def setup(cls):
        dirs = [cls.PROCESSED_DATA_DIR, cls.CHECKPOINT_DIR, cls.METRIC_DIR, cls.FIGURE_DIR, cls.LOG_DIR]
        for d in dirs: d.mkdir(parents=True, exist_ok=True)

# สร้าง instance สำหรับใช้งาน
config = Config()