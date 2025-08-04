import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss):
    """
    Genera una gráfica comparando la pérdida de entrenamiento y validación por época.
    
    Parámetros:
    - train_loss: lista o array con pérdida de entrenamiento por época.
    - val_loss: lista o array con pérdida de validación por época.
    """
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    
    plt.title('Comparación de Pérdida por Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
