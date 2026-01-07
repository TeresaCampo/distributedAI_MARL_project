import wandb
import pandas as pd
from pathlib import Path
import sys, os
from pathlib import Path

def backup_wandb_project(entity, project_name, metrics):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}")
    
    # Crea la directory se non esiste
    file_path = Path(__file__).resolve()
    current_dir = file_path.parent    
    
    project_path = Path(current_dir) / project_name
    project_path.mkdir(parents=True, exist_ok=True)
    
    for run in runs:
        history = run.history(keys=metrics)
        history["run_id"] = run.id
        history["run_name"] = run.name
        history["run_group"] = run.group
        history.to_parquet(project_path / f"{run.name}_{run.group}.parquet")
        
    
    print(f"Backup completato per {project_name}: {project_path}")

if __name__=="__main__":
    USERNAME = "299011-unimore"
    PROJECT_STRICT = "gridworld_strictplay_2agents"
    PROJECT_LOOSE = "gridworld_looseplay_2agents"
    metrics_of_interest = ["Episodes", "Testing_steps_per_episode", "Testing_success_rate_per_episode", "Testing_reward_per_episode"]
    backup_wandb_project(USERNAME, PROJECT_STRICT, metrics_of_interest)
    backup_wandb_project(USERNAME, PROJECT_LOOSE, metrics_of_interest)

    PROJECT_STRICT = "gridworld_strictplay_4agents"
    PROJECT_LOOSE = "gridworld_looseplay_4agents"
    backup_wandb_project(USERNAME, PROJECT_STRICT, metrics_of_interest)
    backup_wandb_project(USERNAME, PROJECT_LOOSE, metrics_of_interest)