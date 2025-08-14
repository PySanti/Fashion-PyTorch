import os
from time import sleep
from random import randint
from utils.show_image import show_image
from scipy.sparse import random
from sklearn.externals.array_api_compat.numpy import test
from utils.MLP import MLP
import torch
from utils.train_model import train_model
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from utils.load_base_datasets import load_base_datasets
from load_best_model import load_best_model
from utils.generate_dataloaders import generate_dataloaders
from utils.accuracy import accuracy


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')

train_dataset, val_dataset, test_dataset = load_base_datasets()
target_map = {
        0:	"T-shirt/top",
        1:	"Trouser",
        2:	"Pullover",
        3:	"Dress",
        4:	"Coat",
        5:	"Sandal",
        6:	"Shirt",
        7:	"Sneaker",
        8:	"Bag",
        9:	"Ankle boot"
    }

try:
    print("Intentando cargar resultados previos")
    model, best_trial = load_best_model()
    _, _ ,test_loader = generate_dataloaders(best_trial.config["batch_size"], train_dataset, val_dataset, test_dataset)

    print("Configuracion del mejor modelo")
    print(best_trial.config)
    print(f"Precision en validacion del mejor modelo : {best_trial.metric_analysis['accuracy']['max']}")

    test_acc = []
    for (X_test_batch, Y_test_batch) in test_loader:
        X_test_batch = X_test_batch.to("cuda")
        Y_test_batch = Y_test_batch.to("cuda")
        outputs = model(X_test_batch)
        test_acc.append(accuracy(outputs, Y_test_batch))
    print(f"Precision en test del mejor modelo : {np.mean(test_acc)}")


    model.eval()
    with torch.no_grad():
        while True:
            random_element = randint(0, len(test_dataset[0]))
            # el unsqueeze se utiliza para agregar una dimension en la posicion 0 al tensor
            # teniendo en cuenta que el modelo espera un batch y no un tensor unico
            random_image = test_dataset[0][random_element].to("cuda")
            prediction = torch.max(model(random_image.unsqueeze(0)),1)[1][0]
            target = test_dataset[1][random_element]
            print(f"Modelo : {target_map[int(prediction)]} / Target : {target_map[int(target)]}")
            show_image(random_image.to("cpu"))


except Exception as e:
    search_space = {
        "base_lr": tune.loguniform(5e-4,5e-2),
        "batch_size": tune.choice([512, 1024]),

        "l1_size": tune.randint(250, 350),
        "l1_drop": tune.loguniform(0.25, 0.35),

        "l2_size": tune.randint(450, 650),
        "l2_drop": tune.loguniform(0.05, 0.15),

        "l3_size": tune.randint(450, 650),
        "l3_drop": tune.loguniform(0.25, 0.35),

        "l2_rate": tune.loguniform(5e-3, 5e-5),
        "t_max" : tune.randint(8,15),
        'max_epochs': 80
    }



    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=50,            # Número máximo de épocas/iteraciones por trial
        grace_period=10,      # Mínimo de épocas antes de considerar detener un trial
        reduction_factor=2   # Factor de reducción de recursos entre etapas
    )

    analysis = tune.run(
        tune.with_parameters(train_model, train_dataset=train_dataset, val_dataset=val_dataset),
        config=search_space,
        scheduler=scheduler,
        verbose=10,

        # Número de combinaciones a probar
        num_samples=75,  

        # recursos
        resources_per_trial={"cpu": 5, "gpu": 1},
        max_concurrent_trials=4,

        # almacenamiento
        name="my_tune_exp",
        trial_dirname_creator=lambda trial:f"trial_{trial.trial_id}",
        storage_path=os.path.abspath("./tune_results"),
        resume=True,

        keep_checkpoints_num=2,       # Máximo de checkpoints a conservar
        checkpoint_score_attr="accuracy"  # Métrica para seleccionar los mejores
    )
    best_config = analysis.get_best_config(metric='accuracy', mode='max')
    best_accuracy = analysis.get_best_trial(metric='accuracy', mode='max').last_result['accuracy']
    print("Mejor configuracion")
    print(best_config)
    print("Mejor accuracy")
    print(best_accuracy)

