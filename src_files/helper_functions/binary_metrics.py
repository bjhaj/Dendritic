import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from src_files.loss_functions.losses import CrossEntropyLS
from src_files.helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings
from src_files.helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib
from torch.cuda.amp import GradScaler, autocast

def binaryROC(args, model, dataset):
    # Set the device for computation (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Put the model in evaluation mode
    model.eval()

    # Define the number of classes (for binary classification, it's 1)
    num_classes = 1

    # Create an empty array to store the predicted probabilities for the positive class (class 1)
    y_pred_proba = np.zeros((len(dataset), num_classes))

    # Iterate over the test dataset and generate predictions
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataset):
            inputs = inputs.to(device)
            outputs = model(inputs.unsqueeze(0))
            probabilities = torch.sigmoid(outputs).squeeze(0)  # Use sigmoid activation for binary classification
            y_pred_proba[i] = probabilities.cpu().numpy()

    # Convert true labels to binary format (0 or 1)
    y_true = np.array(dataset.targets)

    # Create a color palette for the line
    color_palette = plt.cm.get_cmap('tab10')

    # Plot ROC curve for the positive class (class 1)
    plt.figure(figsize=(6, 6))
    y_true_class = y_true
    y_pred_class = y_pred_proba[:, 0]  # Positive class probabilities for binary classification

    fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_class)
    auc = roc_auc_score(y_true_class, y_pred_class)

    # Plot the ROC curve for the positive class
    plt.plot(fpr, tpr, color=color_palette(0), label=f'Dendrite 1, AUC = {auc:.2f}')

    # Plot a line representing random chance (AUC = 0.5)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')

    # Set labels and title for the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification')
    plt.legend()
    plt.show()

def bin_calculate_precision_recall_f1(y_true, y_pred):
    tp = np.sum(y_true & (y_pred >= 0.5))
    fp = np.sum((~y_true) & (y_pred >= 0.5))
    fn = np.sum(y_true & (y_pred < 0.5))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def binary_validation_accuracy(args, history, val_loader, model):
    loss_fn = CrossEntropyLS(args.label_smooth)
    model.eval()
    top1 = AverageMeter()
    val_loss_meter = AverageMeter()

    # Number of classes for binary classification is 1
    num_classes = 1

    # Initialize variables to store true positives, false positives, and false negatives for the positive class (class 1)
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # mixed precision
            with autocast():
                logits = model(input).float()

            # Calculate loss and record it with the AverageMeter
            loss = loss_fn(logits, target)
            val_loss_meter.update(loss.item(), input.size(0))

            # Measure accuracy and record accuracy
            acc1 = accuracy(logits, target)
            if num_distrib() > 1:
                acc1 = reduce_tensor(acc1, num_distrib())
                torch.cuda.synchronize()
            top1.update(acc1.item(), input.size(0))

            # Convert target and logits to numpy arrays
            target_np = target.cpu().numpy()
            logits_np = torch.sigmoid(logits).cpu().numpy()  # Use sigmoid activation for binary classification

            # Calculate true positives, false positives, and false negatives for the positive class (class 1)
            tp += np.sum(target_np == 1) & (logits_np >= 0.5)
            fp += np.sum(target_np == 0) & (logits_np >= 0.5)
            fn += np.sum(target_np == 1) & (logits_np < 0.5)

    # Calculate precision, recall, and F1-score for binary classification
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print metrics for binary classification
    print_at_master("Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}"
                    .format(precision, recall, f1_score))

    history['val_acc'].append(top1.avg)
    history['val_loss'].append(val_loss_meter.avg)
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1_score)

    print_at_master("Validation Accuracy: {:.4f}, Validation Loss: {:.4f}"
                    .format(top1.avg, val_loss_meter.avg))

    model.train()
