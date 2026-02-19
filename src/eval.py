# src/eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
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

def get_detailed_metrics(y_true, y_pred):
    """คำนวณ Metrics แยกกลุ่ม Malignant และ Benign ตามความต้องการของ Paper"""
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0)
    
    # กลุ่มเนื้อร้าย
    mal_classes = ['akiec', 'bcc', 'mel']
    # กลุ่มเนื้อดี
    ben_classes = ['bkl', 'df', 'nv', 'vasc']
    
    mal_recall = np.mean([report[c]['recall'] for c in mal_classes])
    ben_prec = np.mean([report[c]['precision'] for c in ben_classes])
    
    return {
        'Macro_F1': report['macro avg']['f1-score'],
        'Mal_Recall_Avg': mal_recall,
        'Ben_Prec_Avg': ben_prec,
        'Mel_Recall': report['mel']['recall'],
        'Mel_F1': report['mel']['f1-score']
    }

def run_mcnemar_comparison(y_true, y_pred_base, y_pred_prop):
    """คำนวณ McNemar's Test ระหว่างโมเดลพื้นฐานและโมเดลนำเสนอ"""
    y_true = np.array(y_true)
    y_p1 = np.array(y_pred_base)
    y_p2 = np.array(y_pred_prop)

    # สร้าง Contingency Table เฉพาะส่วนที่จำเป็น (Base ถูก/Prop ผิด และ Prop ถูก/Base ผิด)
    b = np.sum((y_p1 == y_true) & (y_p2 != y_true))
    c = np.sum((y_p1 != y_true) & (y_p2 == y_true))
    
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    kappa = cohen_kappa_score(y_p1, y_p2)
    
    return result.pvalue, kappa