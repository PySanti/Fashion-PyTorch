
# Fashion-PyTorch

En este ejercicio resolveremos el dataset *Fashion* (ejercicio que ya resolvimos en el pasado) utilizando la libreria **PyTorch**, libreria que planeamos utilizar los proximos meses.

Se utilizara una arquitectura tipo MLP.

Luego, buscaremos implementar conceptos relacionados con la implementacion avanzada de una red neuronal, vease Normalization, Weigths Initialization (He Initialization), Advanced Activation Functions, LR Scheduling, etc.


# Preprocesamiento

Primeramente, utilizando el siguiente codigo cargamos el conjunto de datos y revisamos el shape 

```
import torch
from torchvision import datasets, transforms
import numpy as np

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
)

# Cargar el dataset de prueba
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
)

print(train_dataset)
print(test_dataset)
```

Resultado:

```
Dataset FashionMNIST
    Number of datapoints: 60000
    Root location: ./data
    Split: Train
Dataset FashionMNIST
    Number of datapoints: 10000
    Root location: ./data
    Split: Test
```

La variable *train_dataset* y *test_dataset* contienen 60.000 y 10.000 tuplas respectivamente con pares (imagen, tag), debemos crear una funcion para convertir las imagenes en su representacion matricial.

Luego, creamos la siguiente funcion para convertir los datasets:

```
import numpy as np
import torch
def convert_dataset(set_):
    X_data = []
    Y_data = []
    print('Convirtiendo imagenes')
    for (img, tag) in set_:
        X_data.append(np.array(img))
        Y_data.append(tag)
    return torch.tensor(np.array(X_data)), torch.tensor(np.array(Y_data)).unsqueeze(1)
```

Y usando el siguiente codigo:

```
from torchvision import datasets
from utils.convert_dataset import convert_dataset

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
)

# Cargar el dataset de prueba
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
)

X_train,Y_train = convert_dataset(train_dataset)
X_test,Y_test = convert_dataset(test_dataset)

print("Shape del x-train")
print(X_train.shape)
print("Shape del y-train")
print(Y_train.shape)

print("Shape del x-test")
print(X_test.shape)
print("Shape del y-test")
print(Y_test.shape)
```

Resultado:

```
Shape del x-train
torch.Size([60000, 28, 28])
Shape del y-train
torch.Size([60000, 1])
Shape del x-test
torch.Size([10000, 28, 28])
Shape del y-test
torch.Size([10000, 1])
```

Conclusion: cada imagen esta representada por una matriz de 28x28, tenemos 60.000 de esas matrices en el conjunto de train y 10.000 en el de test.

Cada imagen tiene la siguiente forma:

```
[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
           0,  13,  73,   0,   0,   1,   4,   0,   0,   0,   0,   1,   1,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   0,
          36, 136, 127,  62,  54,   0,   0,   0,   1,   3,   4,   0,   0,   3],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   6,   0,
         102, 204, 176, 134, 144, 123,  23,   0,   0,   0,   0,  12,  10,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         155, 236, 207, 178, 107, 156, 161, 109,  64,  23,  77, 130,  72,  15],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,  69,
         207, 223, 218, 216, 216, 163, 127, 121, 122, 146, 141,  88, 172,  66],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   0, 200,
         232, 232, 233, 229, 223, 223, 215, 213, 164, 127, 123, 196, 229,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 183,
         225, 216, 223, 228, 235, 227, 224, 222, 224, 221, 223, 245, 173,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 193,
         228, 218, 213, 198, 180, 212, 210, 211, 213, 223, 220, 243, 202,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   3,   0,  12, 219,
         220, 212, 218, 192, 169, 227, 208, 218, 224, 212, 226, 197, 209,  52],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   6,   0,  99, 244,
         222, 220, 218, 203, 198, 221, 215, 213, 222, 220, 245, 119, 167,  56],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   0,   0,  55, 236,
         228, 230, 228, 240, 232, 213, 218, 223, 234, 217, 217, 209,  92,   0],
        [  0,   0,   1,   4,   6,   7,   2,   0,   0,   0,   0,   0, 237, 226,
         217, 223, 222, 219, 222, 221, 216, 223, 229, 215, 218, 255,  77,   0],
        [  0,   3,   0,   0,   0,   0,   0,   0,   0,  62, 145, 204, 228, 207,
         213, 221, 218, 208, 211, 218, 224, 223, 219, 215, 224, 244, 159,   0],
        [  0,   0,   0,   0,  18,  44,  82, 107, 189, 228, 220, 222, 217, 226,
         200, 205, 211, 230, 224, 234, 176, 188, 250, 248, 233, 238, 215,   0],
        [  0,  57, 187, 208, 224, 221, 224, 208, 204, 214, 208, 209, 200, 159,
         245, 193, 206, 223, 255, 255, 221, 234, 221, 211, 220, 232, 246,   0],
        [  3, 202, 228, 224, 221, 211, 211, 214, 205, 205, 205, 220, 240,  80,
         150, 255, 229, 221, 188, 154, 191, 210, 204, 209, 222, 228, 225,   0],
        [ 98, 233, 198, 210, 222, 229, 229, 234, 249, 220, 194, 215, 217, 241,
          65,  73, 106, 117, 168, 219, 221, 215, 217, 223, 223, 224, 229,  29],
        [ 75, 204, 212, 204, 193, 205, 211, 225, 216, 185, 197, 206, 198, 213,
         240, 195, 227, 245, 239, 223, 218, 212, 209, 222, 220, 221, 230,  67],
        [ 48, 203, 183, 194, 213, 197, 185, 190, 194, 192, 202, 214, 219, 221,
         220, 236, 225, 216, 199, 206, 186, 181, 177, 172, 181, 205, 206, 115],
        [  0, 122, 219, 193, 179, 171, 183, 196, 204, 210, 213, 207, 211, 210,
         200, 196, 194, 191, 195, 191, 198, 192, 176, 156, 167, 177, 210,  92],
        [  0,   0,  74, 189, 212, 191, 175, 172, 175, 181, 185, 188, 189, 188,
         193, 198, 204, 209, 210, 210, 211, 188, 188, 194, 192, 216, 170,   0],
        [  2,   0,   0,   0,  66, 200, 222, 237, 239, 242, 246, 243, 244, 221,
         220, 193, 191, 179, 182, 182, 181, 176, 166, 168,  99,  58,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,  40,  61,  44,  72,  41,  35,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
```

Al ser una imagen en escala de grises (con dos canales), no es necesario utilizar *embeddings* avanzados.

Creamos una funcion usando *matplotlib* para poder visualizar las imagenes, son asi:

[imagen no encontrada](./images/img_1.png)

Las etiquetas son las siguientes:

| Etiqueta | Clase         |
|---------:|---------------|
| 0        | T-shirt/top    |
| 1        | Trouser        |
| 2        | Pullover       |
| 3        | Dress          |
| 4        | Coat           |
| 5        | Sandal         |
| 6        | Shirt          |
| 7        | Sneaker        |
| 8        | Bag            |
| 9        | Ankle boot     |

# Construccion de arquitectura


# Entrenamiento Inicial

# Implementacion de tecnicas
