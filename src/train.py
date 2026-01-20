# adv_skin_cancer/src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy
from pathlib import Path

from .config import config
from .loss import FocalLoss
from .utils import get_class_weights

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total if total > 0 else 0})
        
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, device=config.DEVICE):
    # -----------------------------------------------------------
    # 1. คำนวณ Class Weights สำหรับ Loss Function
    # (ทำงานคู่กับ Sampler: Sampler จัดสมดุล Batch, Loss Weight จัดการ Hard Mining)
    # -----------------------------------------------------------
    print("⚖️  Calculating class weights from training set...")
    train_df = train_loader.dataset.df 
    
    class_weights_tensor = get_class_weights(train_df).to(device)
    print(f"Class Weights: {class_weights_tensor.cpu().numpy().round(2)}")
    
    # -----------------------------------------------------------
    # 2. Setup Loss & Optimizer
    # -----------------------------------------------------------
    if config.LOSS_TYPE == 'focal':
        print(f" Using Focal Loss (gamma={config.FOCAL_GAMMA}) WITHOUT Class Weights")
        # แก้ตรงนี้: alpha=None (ไม่ใช้น้ำหนัก เพราะ Sampler จัดให้แล้ว)
        criterion = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=None) 
    else:
        print(" Using Cross Entropy Loss WITHOUT Class Weights")
        # แก้ตรงนี้: weight=None
        criterion = nn.CrossEntropyLoss(weight=None)
    
    # ใช้ Learning Rate ต่ำๆ ตาม Config (แนะนำ 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    # -----------------------------------------------------------
    # 3. Training Loop
    # -----------------------------------------------------------
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f" Start Training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save Best Model (Based on LOSS)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / f'best_model_{config.BASE_MODEL}.pth')
            save_msg = f" Saved (Val Loss: {val_loss:.4f})"
        else:
            save_msg = ""
            
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Time: {epoch_time:.0f}s | LR: {current_lr:.1e}")
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f} {save_msg}")
        print("-" * 40)
        
        if current_lr < 1e-6:
            print(" Early stopping triggered (Low LR)")
            break

    print(f" Training Complete. Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history