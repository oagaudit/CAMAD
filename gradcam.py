# src/gradcam.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from .config import config

def visualize_gradcam(model, image_tensor, target_class=None):
    """
    สร้าง Heatmap ว่าโมเดลโฟกัสตรงไหน
    Args:
        model: โมเดลที่เทรนแล้ว
        image_tensor: รูปภาพ (1, 3, 224, 224)
        target_class: int (ถ้า None จะเอาคลาสที่โมเดลทายมั่นใจสุด)
    """
    model.eval()
    
    # ระบุ Layer สุดท้ายของ ResNet ที่จะดึง Feature (layer4[-1])
    target_layers = [model.backbone.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # ถ้าไม่ระบุคลาส ให้ใช้คลาสที่ทำนายได้
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    # สร้าง Mask (Grayscale Heatmap)
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # เตรียมรูปจริงเพื่อซ้อนทับ (Denormalize)
    img_viz = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    mean = np.array(config.IMAGENET_MEAN)
    std = np.array(config.IMAGENET_STD)
    img_viz = std * img_viz + mean
    img_viz = np.clip(img_viz, 0, 1)

    # ซ้อน Heatmap บนรูปจริง
    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    
    return img_viz, visualization

def plot_gradcam_comparison(original, heatmap, title="Grad-CAM"):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title(title)
    plt.axis('off')
    
    plt.show()