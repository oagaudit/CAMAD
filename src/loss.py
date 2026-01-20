# adv_skin_cancer/src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation strictly following Lin et al.
        Equation: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Tensor of shape [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, Num_Classes] (Logits)
        # targets: [Batch] (Indices)
        
        # 1. คำนวณ CE แบบ "ไม่ใส่ Weight" ก่อน เพื่อหา pt ที่แท้จริง
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. คำนวณ pt (ความน่าจะเป็นที่ทายถูก)
        pt = torch.exp(-ce_loss)
        
        # 3. คำนวณ Focal Term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Apply Alpha (Class Weights) ถ้ามี
        if self.alpha is not None:
            # ตรวจสอบ Device ให้ตรงกัน
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
                
            # เลือกค่า alpha ให้ตรงกับ target ของแต่ละ sample
            # targets view เป็น list เพื่อดึง alpha ทีละตัว
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 5. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            