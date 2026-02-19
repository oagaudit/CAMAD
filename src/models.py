# adv_skin_cancer/src/models.py
import torch
import torch.nn as nn
import torchvision.models as models
from .config import config

# ==========================================
# 1. CBAM Module (Keep existing code)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==========================================
# 2. Model 1: ResNet50 + CBAM (Proposed)
# ==========================================
class SkinCancerResNet(nn.Module):
    def __init__(self, num_classes=7, use_cbam=True, pretrained=True):
        super(SkinCancerResNet, self).__init__()
        
        #  
        # เพื่อให้เห็นภาพว่าเราแทรก CBAM เข้าไปหลังจบแต่ละ Stage
        print(f" Building Model: ResNet50 | CBAM={use_cbam}")
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        self.use_cbam = use_cbam
        
        if self.use_cbam:
            self.cbam1 = CBAMBlock(256)
            self.cbam2 = CBAMBlock(512)
            self.cbam3 = CBAMBlock(1024)
            self.cbam4 = CBAMBlock(2048)
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if self.use_cbam: x = self.cbam1(x)
        
        x = self.backbone.layer2(x)
        if self.use_cbam: x = self.cbam2(x)

        x = self.backbone.layer3(x)
        if self.use_cbam: x = self.cbam3(x)

        x = self.backbone.layer4(x)
        if self.use_cbam: x = self.cbam4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x

# ==========================================
# 3. Model 2: EfficientNet-B0 (Baseline)
# ==========================================
class SkinCancerEfficientNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(SkinCancerEfficientNet, self).__init__()
        
        # 
        # Baseline ที่มีความสมดุลระหว่างความแม่นยำและจำนวน Parameter
        print(f" Building Model: EfficientNet-B0 (Baseline)")
        
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # แก้ไข Classifier Head ของ EfficientNet
        # Structure ของ EfficientNet คือ: model.classifier[1] เป็นตัว Linear สุดท้าย
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2), # Default ของ EffNet คือ 0.2
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 4. Helper Function (Selector)
# ==========================================
def get_model(device=config.DEVICE):
    """
    เลือกโมเดลตามค่า config.BASE_MODEL
    """
    model_name = config.BASE_MODEL.lower()
    
    if 'resnet' in model_name:
        model = SkinCancerResNet(
            num_classes=config.NUM_CLASSES,
            use_cbam=config.USE_CBAM,
            pretrained=config.PRETRAINED
        )
    elif 'efficientnet' in model_name:
        model = SkinCancerEfficientNet(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Check config.py")
        
    return model.to(device)