import matplotlib.pyplot as plt


def plot_accuracy(history):
    mae = history.history['mae']
    loss = history.history['loss']

    epochs = range(len(mae))

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------

    plt.plot(epochs, mae, 'b', label='Training mae')
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Vs Testing Accuracy')
    plt.legend()
    plt.figure()

    # # ------------------------------------------------
    # # Plot mae and loss per epoch skipping the first 50
    # # ------------------------------------------------
    epochs = range(len(mae) - 50)
    plt.plot(epochs, mae[50:], 'b', label='Training Loss')
    plt.plot(epochs, loss[50:], 'r', label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Vs Testing Losss')
    plt.legend()
