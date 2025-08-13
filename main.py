import os
import torch
from utils.train_model import train_model
from ray import tune
from ray.tune.schedulers import ASHAScheduler


print(f'Usando dispositivo {torch.cuda.get_device_name(0)}')


search_space = {
    "base_lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
    "batch_size": tune.choice([256, 512]),

    "l1_size": tune.choice([64, 128, 256, 512]),
    "l1_drop": tune.choice([0.1, 0.2, 0.3, 0.4]),

    "l2_size": tune.choice([64, 128, 256, 512]),
    "l2_drop": tune.choice([0.1, 0.2, 0.3, 0.4]),

    "l3_size": tune.choice([64, 128, 256, 512]),
    "l3_drop": tune.choice([0.1, 0.2, 0.3, 0.4]),

    "l2_rate": tune.choice([1e-3, 1e-2, 1e-1, 1e-4]),
    "t_max" : tune.choice([3,5,7,10]),
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
    train_model,
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
    resume=True
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

