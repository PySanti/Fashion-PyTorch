import os
import torch
from utils.train_model import train_model
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from utils.load_base_datasets import load_base_datasets


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')

train_dataset, val_dataset, test_dataset = load_base_datasets()


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
    storage_path=os.path.abspath("./tune_results")
)
best_config = analysis.get_best_config(metric='accuracy', mode='max')
best_accuracy = analysis.get_best_trial(metric='accuracy', mode='max').last_result['accuracy']
print("Mejor configuracion")
print(best_config)
print("Mejor accuracy")
print(best_accuracy)

# test accuracy
#test_acc = np.array([])
#for (X_test_batch, Y_test_batch) in test_loader:
#    outputs = mlp(X_test_batch)
#    test_acc = np.append(test_acc, accuracy(outputs, Y_test_batch))
#print(f"Test acc : {test_acc.mean()}")

