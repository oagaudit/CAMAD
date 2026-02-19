# src/gradcam.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.config import config

def get_gradcam_map(model, image_tensor, model_type='resnet', target_class=None):
    """
    สร้าง Heatmap ตามสถาปัตยกรรมของโมเดล พร้อมระบบตรวจสอบ Layer อัตโนมัติ
    """
    model.eval()
    
    # 1. ระบุ Target Layer
    if 'resnet' in model_type.lower():
        # สำหรับ ResNet50 (Vanilla & CBAM)
        target_layers = [model.backbone.layer4[-1]]
    elif 'efficientnet' in model_type.lower():
        # สำหรับ EfficientNet: ลองหาชื่อ layer ที่เป็นไปได้
        backbone = model.backbone
        if hasattr(backbone, 'conv_head'):
            target_layers = [backbone.conv_head]
        elif hasattr(backbone, 'features'): # กรณี torchvision
            target_layers = [backbone.features[-1]]
        elif hasattr(backbone, 'act2'): # กรณีบาง version ของ timm
            target_layers = [backbone.act2]
        else:
            # ถ้าหาไม่เจอจริงๆ ให้ใช้ Layer สุดท้ายของโมเดล
            target_layers = [list(backbone.children())[-3]] 
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 2. คำนวณ Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

    # 3. เตรียมรูปภาพ (Denormalize)
    img_viz = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_viz = (np.array(config.IMAGENET_STD) * img_viz) + np.array(config.IMAGENET_MEAN)
    img_viz = np.clip(img_viz, 0, 1)

    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    #return img_viz, visualization
    return img_viz, visualization, grayscale_cam

def plot_triple_comparison(img_original, cam_eff, cam_vanilla, cam_proposed, info):
    """ วาดรูปเปรียบเทียบ 4 ช่องสำหรับ Paper Highlight """
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    
    titles = [
        f"Original Image\n(Actual: {info['label']})",
        f"EfficientNet-B0 (SOTA)\nPred: {info['pred_eff']}",
        f"ResNet50 Vanilla\nPred: {info['pred_van']}",
        f"Proposed CAMAD\nPred: {info['pred_prop']}"
    ]
    
    images = [img_original, cam_eff, cam_vanilla, cam_proposed]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    return fig