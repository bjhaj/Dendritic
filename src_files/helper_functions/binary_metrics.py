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
