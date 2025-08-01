import matplotlib.pyplot as plt

def show_image(imagen_2d):
    """
    Renderiza una imagen 2D de forma (28, 28) usando escala de grises.

    Parámetros:
        - imagen_2d: matriz de 28x28 (tipo tensor, ndarray, o lista de listas)
    """
    plt.imshow(imagen_2d, cmap='gray')
    plt.axis('off')
    plt.show()
