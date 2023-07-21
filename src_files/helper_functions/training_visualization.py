def plot_history(history):
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc.cpu().numpy() for acc in train_acc], 'b', label='Training acc')
    plt.plot(epochs, [acc.cpu().numpy() for acc in val_acc], 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
