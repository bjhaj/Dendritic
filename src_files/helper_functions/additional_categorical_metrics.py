import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from src_files.loss_functions.losses import CrossEntropyLS
from src_files.helper_functions.general_helper_functions import accuracy, AverageMeter, silence_PIL_warnings
from src_files.helper_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib
from torch.cuda.amp import GradScaler, autocast


# Assume you have a PyTorch model named 'model' and a test dataset named 'test_dataset'

# Set the device for computation (CPU or GPU)
def printROC(args, model, dataset):
  # Set the device for computation (CPU or GPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Put the model in evaluation mode
  model.eval()

  # Define the number of classes
  num_classes = args.num_classes

  # Create an empty array to store the predicted probabilities for each class
  y_pred_proba = np.zeros((len(dataset), num_classes))

  # Iterate over the test dataset and generate predictions
  with torch.no_grad():
      for i, (inputs, _) in enumerate(dataset):
          inputs = inputs.to(device)
          outputs = model(inputs.unsqueeze(0))
          probabilities = torch.softmax(outputs, dim=1).squeeze(0)
          y_pred_proba[i] = probabilities.cpu().numpy()

  # Convert true labels to one-hot encoded format
  y_true = label_binarize(dataset.targets, classes=list(range(num_classes)))

  # Create a color palette for the lines
  color_palette = plt.cm.get_cmap('tab10')

  # Plot ROC curves for each class
  plt.figure(figsize=(6, 6))
  for class_idx in range(num_classes):
      y_true_class = y_true[:, class_idx]
      y_pred_class = y_pred_proba[:, class_idx]

      fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_class)
      auc = roc_auc_score(y_true_class, y_pred_class)

      # Plot the ROC curve for the current class
      plt.plot(fpr, tpr, color=color_palette(class_idx), label=f'Dendrite {class_idx}, AUC = {auc:.2f}')

  # Plot a line representing random chance (AUC = 0.5)
  plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')

  # Set labels and title for the plot
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curves - Multiclass Classification')
  plt.legend()
  plt.show()

def calculate_precision_recall_f1(y_true, y_pred):
    tp = np.sum(y_true & (y_pred >= 0.5))
    fp = np.sum((~y_true) & (y_pred >= 0.5))
    fn = np.sum(y_true & (y_pred < 0.5))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def validation_accuracy(args, history, val_loader, model):
    loss_fn = CrossEntropyLS(args.label_smooth)
    model.eval()
    top1 = AverageMeter()
    val_loss_meter = AverageMeter()
    num_classes = args.num_classes

    # Initialize dictionaries to store true positives, false positives, and false negatives for each class
    tp_dict = {i: 0 for i in range(num_classes)}
    fp_dict = {i: 0 for i in range(num_classes)}
    fn_dict = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # mixed precision
            with autocast():
                logits = model(input).float()

            # Calculate loss and record it with the AverageMeter
            loss = loss_fn(logits, target)
            val_loss_meter.update(loss.item(), input.size(0))

            # measure accuracy and record accuracy
            acc1 = accuracy(logits, target)
            if num_distrib() > 1:
                acc1 = reduce_tensor(acc1, num_distrib())
                torch.cuda.synchronize()
            top1.update(acc1.item(), input.size(0))

            # Convert target and logits to numpy arrays
            target_np = target.cpu().numpy()
            logits_np = torch.softmax(logits, dim=1).cpu().numpy()

            # One-vs-all calculation for each class
            for class_idx in range(num_classes):
                y_true_class = (target_np == class_idx)
                y_pred_class = logits_np[:, class_idx]

                # Calculate true positives, false positives, and false negatives for the current class
                tp = np.sum(y_true_class & (y_pred_class >= 0.5))
                fp = np.sum((~y_true_class) & (y_pred_class >= 0.5))
                fn = np.sum(y_true_class & (y_pred_class < 0.5))

                # Update dictionaries with the counts for the current class
                tp_dict[class_idx] += tp
                fp_dict[class_idx] += fp
                fn_dict[class_idx] += fn

    # Calculate and print the metrics for each class
    for class_idx in range(num_classes):
        precision, recall, f1_score = calculate_precision_recall_f1(y_true=(target_np == class_idx), y_pred=logits_np[:, class_idx])
        print_at_master("Class {}: Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}"
                        .format(class_idx, precision, recall, f1_score))

    history['val_acc'].append(top1.avg)
    history['val_loss'].append(val_loss_meter.avg)
    history['precision'].append(precision)
    history['recall'].append(recall)
    history['f1'].append(f1_score)

    print_at_master("Validation Accuracy: {:.4f}, Validation Loss: {:.4f}"
                    .format(top1.avg, val_loss_meter.avg))

    model.train()

