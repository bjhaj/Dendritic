from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

# Plot the false-positive rate of a model compared to the true-positive rate (ROC-Curves)
def b_plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

def b_evaluate(model, dataloader, device, FIXED_FPR=0.05):
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

    fprs, tprs, thresholds = roc_curve(y_true, y_pred)
    plot_roc(fprs, tprs)
    tpr = tprs[fprs < FIXED_FPR][-1]
    fpr = fprs[fprs < FIXED_FPR][-1]
    threshold = thresholds[fprs < FIXED_FPR][-1]

    print("AUC:", roc_auc_score(y_true, y_pred))
    to_pct = lambda x: str(round(x, 4) * 100) + "%"
    print("TPR: ", to_pct(tpr), "\nFPR: ", to_pct(fpr), "\nThreshold: ", round(threshold, 2))

def b_score_keras_model(model):
    # Score the test set
    evaluate(model, dataloaders['test'], device)
