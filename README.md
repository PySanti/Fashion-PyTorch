
# Fashion-PyTorch

En este ejercicio resolveremos el dataset *Fashion* (ejercicio que ya resolvimos en el pasado) utilizando la libreria **PyTorch**, libreria que planeamos utilizar los proximos meses.

Se utilizara una arquitectura tipo MLP.

Luego, buscaremos implementar conceptos relacionados con la implementacion avanzada de una red neuronal, vease Normalization, Weigths Initialization (He Initialization), Advanced Activation Functions, LR Scheduling, etc.


# Preprocesamiento

## Visualizacion del dataset

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

Creamos una funcion usando *matplotlib* para poder visualizar las imagenes. La siguiente imagen es de la clase 4:

![imagen no encontrada](./images/img_1.png)

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

## Escalado

Utilizamos *normalizacion* y *estandarizacion* respectivamente para escalar los datos:

```
import torch
import numpy as np

def convert_dataset(dataset):
    images = torch.stack([torch.tensor(np.array(img).astype('float32')) / 255.0 for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])
    
    mean, std = 0.2860, 0.3530  # Precalculados para FashionMNIST
    images = (images - mean) / std
    
    return images.to('cuda'), labels.unsqueeze(1).to('cuda')
```

De este modo logramos que todas las features (pixeles) esten en la misma escala para todas las muestras.

## Division del conjunto de datos

Utilizamos *scikit-learn* para dividir el conjunto de test en validacion y test:

```
# main.py

from torchvision import datasets
import torch
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')

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

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, stratify=Y_test)

X_train = X_train.to('cuda')
Y_train = Y_train.to('cuda')
X_val  = X_val.to('cuda')
Y_val  = Y_val.to('cuda')
X_test = X_test.to('cuda')
Y_test = Y_test.to('cuda')

```

Ademas, **movimos los tensores a la GPU**.

# Construccion de arquitectura

El codigo utilizado para la definicion de la arquitectura fue la siguiente:

```
import torch
class MLP(torch.nn.Module):
    def __init__(self, input_shape=(28,28)):
        super(MLP, self).__init__()
        self.flat_layer = torch.nn.Flatten()
        self.hl1 = torch.nn.Linear(input_shape[0]*input_shape[1], 120)
        self.hl2 = torch.nn.Linear(120, 72)
        self.hl3 = torch.nn.Linear(72, 48)
        self.out_layer = torch.nn.Linear(48, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.flat_layer(x)
        out = self.hl1(out)
        out = self.relu(out)
        out = self.hl2(out)
        out = self.relu(out)
        out = self.hl3(out)
        out = self.relu(out)
        out = self.out_layer(out)
        return out

```

El codigo en **main.py** fue el siguiente:

```
from numpy import outer
from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
from utils.MLP import MLP
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')

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

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, stratify=Y_test)

X_train = X_train.to('cuda')
Y_train = Y_train.to('cuda')
X_val  = X_val.to('cuda')
Y_val  = Y_val.to('cuda')
X_test = X_test.to('cuda')
Y_test = Y_test.to('cuda')

mlp = MLP().to('cuda')
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.RMSprop(mlp.parameters(), lr=0.1, momentum=0.9)
train_set = MyDataset(X_train, Y_train)
dataloader = torch.utils.data.DataLoader(dataset=train_set,batch_size=64,shuffle=True)

epochs = 100

for ep in range(epochs):

    for (X_train_batch, Y_train_batch) in dataloader:
        X_train_batch = X_train_batch.float()
        Y_train_batch = Y_train_batch
        outputs = mlp(X_train_batch)
        batch_loss = loss(outputs, Y_train_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    print(f'Epoca actual : {ep}/{epochs}')
    print(batch_loss)

```


# Entrenamiento Inicial

Utilizando el siguiente codigo (despues de algunas pruebas previas), obtenemos los posteriores resultados:

```
import numpy as np
from torchvision import datasets
import torch
from utils.Dataset import  MyDataset
from utils.MLP import MLP
from utils.convert_dataset import convert_dataset
from sklearn.model_selection import train_test_split
from utils.plot_loss import plot_loss


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')

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

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42, stratify=Y_test)

X_train = X_train.to('cuda')
Y_train = Y_train.to('cuda')
X_val  = X_val.to('cuda')
Y_val  = Y_val.to('cuda')
X_test = X_test.to('cuda')
Y_test = Y_test.to('cuda')


BATCH_SIZE  = 128
EPOCHS      = 40

# loaders
train_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True)
val_loader = torch.utils.data.DataLoader(
        dataset=MyDataset(X_val, Y_val),
        batch_size=BATCH_SIZE,
        shuffle=True)

mlp = MLP().to('cuda')
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.RMSprop(mlp.parameters(), lr=0.0001, momentum=0.9)

val_losses = np.array([])
train_losses = np.array([])


for ep in range(EPOCHS):

    # entrenamiento

    batches_train_loss = np.array([])
    mlp.train()

    for (X_train_batch, Y_train_batch) in train_loader:
        X_train_batch = X_train_batch.float()
        outputs = mlp(X_train_batch)
        batch_loss = loss(outputs, Y_train_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batches_train_loss = np.append(batches_train_loss, batch_loss.item())


    # validacion
    mlp.eval()
    batches_val_loss = np.array([])

    with torch.no_grad():
        for (X_val_batch, Y_val_batch) in val_loader:
            X_val_batch = X_val_batch.float()
            batches_val_loss = np.append(batches_val_loss, loss(mlp(X_val_batch), Y_val_batch).item())


    print(f'Epoca actual : {ep}/{EPOCHS}')
    print(f"\tTrain batches : {len(batches_train_loss)}")
    print(f'\tTrain loss : {batches_train_loss.mean()}')
    print(f'\tVal loss : {batches_val_loss.mean()}')
    print(f"\tDiff: {((batches_val_loss.mean())*100)/(batches_train_loss.mean())-100}")

    val_losses = np.append(val_losses, batches_val_loss.mean())
    train_losses = np.append(train_losses, batches_train_loss.mean())

print(val_losses)
print(train_losses)
plot_loss(train_losses, val_losses)
```

```
Usando dispositivo NVIDIA GeForce RTX 5070
Epoca actual : 0/40
        Train batches : 469
        Train loss : 0.5160285907386463
        Val loss : 0.43835695944726466
        Diff: -15.051807726428876
Epoca actual : 1/40
        Train batches : 469
        Train loss : 0.36329284389771377
        Val loss : 0.38761087954044343
        Diff: 6.693783280128812
Epoca actual : 2/40
        Train batches : 469
        Train loss : 0.32625732222981035
        Val loss : 0.3953237425535917
        Diff: 21.169308891443706
Epoca actual : 3/40
        Train batches : 469
        Train loss : 0.30254963510580407
        Val loss : 0.3824466414749622
        Diff: 26.407900423088435
Epoca actual : 4/40
        Train batches : 469
        Train loss : 0.28373749153827554
        Val loss : 0.364764466509223
        Diff: 28.5570209744443
Epoca actual : 5/40
        Train batches : 469
        Train loss : 0.2674202574595714
        Val loss : 0.3483842906542122
        Diff: 30.27595364830617
Epoca actual : 6/40
        Train batches : 469
        Train loss : 0.25402771034982924
        Val loss : 0.35781815759837626
        Diff: 40.85792337600259
Epoca actual : 7/40
        Train batches : 469
        Train loss : 0.24276356375230146
        Val loss : 0.3430200483649969
        Diff: 41.2979950792739
Epoca actual : 8/40
        Train batches : 469
        Train loss : 0.23058735733347407
        Val loss : 0.34542085528373717
        Diff: 49.800431072286244
Epoca actual : 9/40
        Train batches : 469
        Train loss : 0.2225358535422445
        Val loss : 0.33681214936077597
        Diff: 51.35185813859792
Epoca actual : 10/40
        Train batches : 469
        Train loss : 0.2106702119938092
        Val loss : 0.34740526527166365
        Diff: 64.90478743234596
Epoca actual : 11/40
        Train batches : 469
        Train loss : 0.20429983657242648
        Val loss : 0.37785576581954955
        Diff: 84.95157517445963
Epoca actual : 12/40
        Train batches : 469
        Train loss : 0.19475546995523388
        Val loss : 0.36280880514532327
        Diff: 86.28940446638947
Epoca actual : 13/40
        Train batches : 469
        Train loss : 0.187456229427603
        Val loss : 0.36133833080530164
        Diff: 92.75877462629387
Epoca actual : 14/40
        Train batches : 469
        Train loss : 0.17829608277026524
        Val loss : 0.3520105991512537
        Diff: 97.43036060125891
Epoca actual : 15/40
        Train batches : 469
        Train loss : 0.1724710003462936
        Val loss : 0.3832379741594195
        Diff: 122.20429718036092
Epoca actual : 16/40
        Train batches : 469
        Train loss : 0.1682966111628994
        Val loss : 0.36959443241357803
        Diff: 119.60895698359391
Epoca actual : 17/40
        Train batches : 469
        Train loss : 0.1615974226199996
        Val loss : 0.3959186911582947
        Diff: 145.00309766035528
Epoca actual : 18/40
        Train batches : 469
        Train loss : 0.15516087173748372
        Val loss : 0.4670820966362953
        Diff: 201.03085359468093
Epoca actual : 19/40
        Train batches : 469
        Train loss : 0.15042928835031574
        Val loss : 0.42649011574685575
        Diff: 183.51534493313352
Epoca actual : 20/40
        Train batches : 469
        Train loss : 0.14564127414656092
        Val loss : 0.42868103124201296
        Diff: 194.3403466867676
Epoca actual : 21/40
        Train batches : 469
        Train loss : 0.140365113788195
        Val loss : 0.40935792922973635
        Diff: 191.63794206546225
Epoca actual : 22/40
        Train batches : 469
        Train loss : 0.13641048774027875
        Val loss : 0.4170353688299656
        Diff: 205.72089854556327
Epoca actual : 23/40
        Train batches : 469
        Train loss : 0.1339985163830745
        Val loss : 0.46532224789261817
        Diff: 247.25925364901542
Epoca actual : 24/40
        Train batches : 469
        Train loss : 0.12621948876773625
        Val loss : 0.4529493160312995
        Diff: 258.85846191692127
Epoca actual : 25/40
        Train batches : 469
        Train loss : 0.12471279528921347
        Val loss : 0.517032890021801
        Diff: 314.57886403939796
Epoca actual : 26/40
        Train batches : 469
        Train loss : 0.12242884076099152
        Val loss : 0.5325593311339617
        Diff: 334.9949961330081
Epoca actual : 27/40
        Train batches : 469
        Train loss : 0.11649491698709505
        Val loss : 0.5003268163651228
        Diff: 329.48381723860746
Epoca actual : 28/40
        Train batches : 469
        Train loss : 0.11351050271479877
        Val loss : 0.4886847872287035
        Diff: 330.5194458142347
Epoca actual : 29/40
        Train batches : 469
        Train loss : 0.10977587009718551
        Val loss : 0.4620428788010031
        Diff: 320.8965762621172
Epoca actual : 30/40
        Train batches : 469
        Train loss : 0.10809904517236549
        Val loss : 0.5582607932388782
        Diff: 416.43452756564443
Epoca actual : 31/40
        Train batches : 469
        Train loss : 0.10479304878346955
        Val loss : 0.5076058279722929
        Diff: 384.3888348177962
Epoca actual : 32/40
        Train batches : 469
        Train loss : 0.10366247808818878
        Val loss : 0.4988673336803913
        Diff: 381.2419526147059
Epoca actual : 33/40
        Train batches : 469
        Train loss : 0.09926912032051889
        Val loss : 0.49196155928075314
        Diff: 395.58367969043525
Epoca actual : 34/40
        Train batches : 469
        Train loss : 0.09696057065527068
        Val loss : 0.5769057869911194
        Diff: 494.9900903968735
Epoca actual : 35/40
        Train batches : 469
        Train loss : 0.09320845270255354
        Val loss : 0.5507992915809155
        Diff: 490.9327701626204
Epoca actual : 36/40
        Train batches : 469
        Train loss : 0.09224395164802893
        Val loss : 0.5336498458404094
        Diff: 478.52014826579955
Epoca actual : 37/40
        Train batches : 469
        Train loss : 0.09088020000074591
        Val loss : 0.5637221157550811
        Diff: 520.2914559502007
Epoca actual : 38/40
        Train batches : 469
        Train loss : 0.08790211446249663
        Val loss : 0.5988429114222527
        Diff: 581.261099444597
Epoca actual : 39/40
        Train batches : 469
        Train loss : 0.0853665280443773
        Val loss : 0.5969214454293251
        Diff: 599.2453120724541
```

Como vemos se empieza a generar overfitting muy rapidamente.

![imagen no encontrada](./images/overfitting1.png)

## L2 y Dropout



# Implementacion de tecnicas
