import os
import wandb


def init_wandb(project_name, experiment_name, wandb_api_key):
    if project_name is not None and experiment_name is not None:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(project=project_name, name=experiment_name)
