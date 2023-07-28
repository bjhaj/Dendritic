import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from src_files.loss_functions.losses import CrossEntropyLS
from src_files.helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings
from src_files.helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib
from torch.cuda.amp import GradScaler, autocast

def binaryROC(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FIXED_FPR = 0.05
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = torch.tensor(y_true)  # Convert y_true to a PyTorch tensor
    y_pred = torch.tensor(y_pred)  # Convert y_pred to a PyTorch tensor

    fprs, tprs, thresholds = roc_curve(y_true, y_pred)

    # Plot the ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fprs, tprs, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    tpr = tprs[fprs < FIXED_FPR][-1]
    fpr = fprs[fprs < FIXED_FPR][-1]
    threshold = thresholds[fprs < FIXED_FPR][-1]

    print("AUC:", roc_auc_score(y_true, y_pred))
    to_pct = lambda x: str(round(x, 4) * 100) + "%"
    print("TPR: ", to_pct(tpr), "\nFPR: ", to_pct(fpr), "\nThreshold: ", round(threshold, 2))


def bin_calculate_precision_recall_f1(y_true, y_pred):
    tp = np.sum(y_true & (y_pred >= 0.5))
    fp = np.sum((~y_true) & (y_pred >= 0.5))
    fn = np.sum(y_true & (y_pred < 0.5))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score



