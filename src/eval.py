# src/eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from .config import config

def evaluate_model_performance(model, loader, device=config.DEVICE):
    """
    ฟังก์ชันสำหรับวัดผลโมเดลแบบละเอียด (Classification Report + Confusion Matrix)
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    print("Evaluating Model...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    # 1. Classification Report (Macro F1, Recall per class)
    class_names = [config.IDX_TO_CLASS[i] for i in range(config.NUM_CLASSES)]
    print("\n" + "="*40)
    print("Classification Report")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # 2. Balanced Accuracy
    b_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced Accuracy: {b_acc:.4f}")

    # 3. Specific Metric: Melanoma Recall
    mel_idx = config.CLASS_TO_IDX['mel']
    # หา Recall ของ Mel โดยเฉพาะ (TP / (TP + FN))
    cm = confusion_matrix(y_true, y_pred)
    mel_recall = cm[mel_idx, mel_idx] / cm[mel_idx].sum()
    print(f"Melanoma Recall:   {mel_recall:.4f}")
    
    # 4. Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Balanced Acc: {b_acc:.4f})')
    
    # Save Figure
    save_path = config.FIGURE_DIR / 'confusion_matrix_final.png'
    plt.savefig(save_path)
    print(f"\n Saved Confusion Matrix to {save_path}")
    plt.show()

    return y_true, y_pred, y_prob