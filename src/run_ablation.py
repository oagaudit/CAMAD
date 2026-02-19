import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tabulate import tabulate
from pathlib import Path
from datetime import datetime

# Import Modules
from src.config import config
from src.dataset import HAM10000Dataset, get_weighted_dataloader
from src.transforms import get_transforms
from src.utils import seed_everything, get_class_weights
from src.models import get_model
from src.loss import FocalLoss
from src.train import train_one_epoch, evaluate
from src.eval import evaluate_model_performance, get_detailed_metrics

# ==========================================
# 🔧 CONFIG: ใส่ Path ของโมเดลเดิมที่นี่
# ==========================================
CHECKPOINT_BASELINE = "models/checkpoints/best_model_resnet50_vanilla.pth" 
CHECKPOINT_FULL     = "models/checkpoints/best_model_resnet50_cbam.pth"     

# ==========================================
# ✅ FIX: ย้าย CustomDataset ออกมาไว้นอกฟังก์ชัน
# ==========================================
class CustomDataset(HAM10000Dataset):
    """Subclass ที่รับ df โดยตรง ไม่ต้องโหลดจากไฟล์"""
    def __init__(self, df, transform=None):
        # Override __init__ เพื่อรับ DataFrame โดยตรง
        # เราไม่เรียก super().__init__ เพราะมันจะไปโหลด CSV
        self.df = df.copy()
        self.transform = transform
        self.class_to_idx = config.CLASS_TO_IDX

def run_experiment(exp_config, df_train, df_val, df_test, device, save_dir):
    print(f"\n{'='*60}")
    print(f"🚀 Experiment: {exp_config['name']}")
    
    # 1. Setup Config
    config.BASE_MODEL = exp_config['base_model']
    config.USE_CBAM = exp_config['use_cbam']
    config.LOSS_TYPE = exp_config['loss_type']
    config.USE_WEIGHTED_SAMPLER = False 
    
    # 2. Prepare Data
    train_transform = get_transforms(split='train')
    val_transform = get_transforms(split='val')
    
    # เรียกใช้ Class ที่ประกาศไว้ด้านบน
    train_dataset = CustomDataset(df_train, transform=train_transform)
    val_dataset = CustomDataset(df_val, transform=val_transform)
    test_dataset = CustomDataset(df_test, transform=val_transform)
    
    # num_workers > 0 จะทำงานได้แล้ว เพราะ CustomDataset อยู่ Global Scope
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model Setup
    model = get_model(device).to(device)
    class_weights = get_class_weights(df_train).to(device)

    # --- MODE: LOAD EXISTING (สำหรับข้อ 1 และ 4) ---
    if exp_config['mode'] == 'load':
        print(f"  📂 Loading Pre-trained Checkpoint: {exp_config['checkpoint_path']}")
        try:
            checkpoint = torch.load(exp_config['checkpoint_path'], map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("  ✅ Load Complete!")
        except FileNotFoundError:
            print(f"  ❌ Error: Checkpoint not found at {exp_config['checkpoint_path']}")
            return None
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
            return None

    # --- MODE: TRAIN NEW (สำหรับข้อ 2 และ 3) ---
    else:
        print(f"  🔥 Training New Model (Config: {exp_config['settings_desc']})")
        
        # Setup Loss
        if config.LOSS_TYPE == 'focal':
            criterion = FocalLoss(alpha=class_weights if exp_config.get('use_weighted', True) else None, gamma=2.0).to(device)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights if exp_config.get('use_weighted', False) else None).to(device)
            
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        
        best_loss = float('inf')
        best_wts = None
        
        # Train Loop
        epochs = 15
        for ep in range(epochs):
            train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"  Ep {ep+1}/{epochs} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_wts = {k: v.cpu() for k, v in model.state_dict().items()}
        
        model.load_state_dict(best_wts)
        # Save รุ่นกลางทางไว้
        torch.save(model.state_dict(), save_dir / f"{exp_config['id']}_best.pth")

    # 4. Evaluate on TEST SET
    print("  📊 Evaluating on Test Set...")
    model.eval()
    model.to(device)
    y_true, y_pred, _ = evaluate_model_performance(model, test_loader, device=device)
    metrics = get_detailed_metrics(y_true, y_pred)
    
    return {
        'Model': exp_config['name'],
        'Macro F1': metrics['Macro_F1'],
        'Mel Recall': metrics['Mel_Recall'],
        'Mal Avg Recall': metrics['Mal_Recall_Avg'],
        'Ben Prec Avg': metrics['Ben_Prec_Avg']
    }

def main():
    seed_everything(42)
    device = config.DEVICE
    
    save_dir = config.RESULT_DIR / f"ablation_partial_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_train = pd.read_csv(config.PROCESSED_DATA_DIR / "train.csv")
        df_val = pd.read_csv(config.PROCESSED_DATA_DIR / "val.csv")
        df_test = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")
    except:
        print("❌ Run data preparation script first!")
        return

    # =================================================================
    # 📋 ABLATION PLAN
    # =================================================================
    experiments = [
        # 1. Baseline: Load ของเดิม (Standard Aug + CE)
        {
            'id': 'exp1',
            'name': '1. Baseline (ResNet50)',
            'mode': 'load',
            'checkpoint_path': CHECKPOINT_BASELINE,
            'base_model': 'resnet50_standard', 
            'use_cbam': False,
            'loss_type': 'ce'
        },
        # 2. + CSA: เทรนใหม่ (CSA + CE)
        {
            'id': 'exp2',
            'name': '2. + CSA',
            'mode': 'train',
            'settings_desc': 'CE Loss + Class-Specific Aug',
            'base_model': 'resnet50', 
            'use_cbam': False,
            'loss_type': 'ce',
            'use_weighted': True 
        },
        # 3. + WFL: เทรนใหม่ (CSA + WFL)
        {
            'id': 'exp3',
            'name': '3. + WFL',
            'mode': 'train',
            'settings_desc': 'Focal Loss + Class-Specific Aug',
            'base_model': 'resnet50',
            'use_cbam': False,
            'loss_type': 'focal',
            'use_weighted': True
        },
        # 4. Full: Load ของเดิม (CSA + WFL + CBAM)
        {
            'id': 'exp4',
            'name': '4. Full CAMAD',
            'mode': 'load',
            'checkpoint_path': CHECKPOINT_FULL,
            'base_model': 'resnet50',
            'use_cbam': True,
            'loss_type': 'focal'
        }
    ]

    results = []
    for exp in experiments:
        res = run_experiment(exp, df_train, df_val, df_test, device, save_dir)
        if res: results.append(res)

    print("\n" + "="*80)
    print("FINAL ABLATION STUDY RESULTS")
    print("="*80)
    if len(results) > 0:
        print(tabulate(results, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
        pd.DataFrame(results).to_csv(save_dir / 'ablation_final.csv', index=False)
    else:
        print("No results obtained.")

if __name__ == "__main__":
    main()