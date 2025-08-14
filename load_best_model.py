import torch
from utils.MLP import MLP  # Importa tu clase de modelo
from ray.tune import ExperimentAnalysis
import os
from utils.MLP import MLP  # Asegúrate de importar tu modelo
from ray.train.torch import TorchCheckpoint

def load_best_model(experiment_path="~/ray_results/my_tune_exp"):
    experiment_path = os.path.abspath("./tune_results/my_tune_exp/")
    os.makedirs(experiment_path, exist_ok=True)
    analysis = ExperimentAnalysis(experiment_path)
    best_trial = analysis.get_best_trial(metric="accuracy", mode="max", scope="all")
    
    if not best_trial.checkpoint:
        raise ValueError("El mejor trial no tiene checkpoint")
    
    model = MLP(
        hidden_sizes=[
            best_trial.config["l1_size"],
            best_trial.config["l2_size"],
            best_trial.config["l3_size"]
        ],
        dropout_rates=[
            best_trial.config["l1_drop"],
            best_trial.config["l2_drop"],
            best_trial.config["l3_drop"]
        ]
    )
    
    with best_trial.checkpoint.as_directory() as checkpoint_dir:
        model_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
        model.load_state_dict(model_state_dict)

    model = model.to("cuda")
    return model, best_trial

