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
  plt.figure()
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

def validation_accuracy(args, history, val_loader, model):
    loss_fn = CrossEntropyLS(args.label_smooth)
    model.eval()
    top1 = AverageMeter()
    val_loss_meter = AverageMeter()  # Add an AverageMeter to track validation loss
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # mixed precision
            with autocast():
                logits = model(input).float()

            # Calculate loss and record it with the AverageMeter
            loss = loss_fn(logits, target)
            val_loss_meter.update(loss.item(), input.size(0))

            # measure accuracy and record accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            if num_distrib() > 1:
                acc1 = reduce_tensor(acc1, num_distrib())
                torch.cuda.synchronize()
            top1.update(acc1.item(), input.size(0))

    history['val_acc'].append(top1.avg)
    # Save the validation loss for the current epoch
    history['val_loss'].append(val_loss_meter.avg)

    print_at_master("Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
    .format(val_loss_meter.avg,top1.avg))

    model.train()
