import gc
import os

import numpy as np
import optuna
import pandas as pd
import wandb
from optuna.integration import WeightsAndBiasesCallback
from sdv.tabular import CTGAN, TVAE

from augmentations.objectives import CTGANObjective, TVAEObjective
from utils.datasets import CommonDataset
import warnings


def run_objectives(dataset: CommonDataset, best_model_path,
                   project_str="optuna-wandb-augs-release_local",
                   n_trials=100, n_jobs=2,
                   percentage_selection=None):
    np.random.seed(7575)
    warnings.filterwarnings('ignore')
    os.makedirs(best_model_path, exist_ok=True)
    dataset_name = dataset.dataset_name
    wandb_kwargs = {"project": project_str, 'name': dataset_name}
    data = pd.DataFrame(np.hstack((dataset.X, np.expand_dims(dataset.y, axis=1))),
                        columns=list(map(str, np.arange(start=0, stop=len(dataset.X[0]) + 1))))
    del dataset
    gc.collect()
    if percentage_selection is not None:
        N = int(percentage_selection * len(data))
        data = data.iloc[np.random.choice(len(data), size=N, replace=False)]
    ######## CTGAN ########
    wandbc = WeightsAndBiasesCallback(metric_name=f"evaluation_ctgan_{dataset_name}",
                                      wandb_kwargs=wandb_kwargs)
    ctgan_objective = CTGANObjective(data)
    study = optuna.create_study(direction='maximize')
    study.optimize(ctgan_objective, n_trials=n_trials, callbacks=[wandbc], n_jobs=n_jobs)
    wandb.log({f'best_params_ctgan_{dataset_name}': study.best_params})

    ctgan = CTGAN(**CTGANObjective.params_to_kwargs(study.best_params))
    ctgan.fit(data)
    ctgan.save(best_model_path + '/ctgan_best.pkl')

    ######## TVAE #########
    wandbc = WeightsAndBiasesCallback(metric_name=f"evaluation_tvae_{dataset_name}", wandb_kwargs=wandb_kwargs)
    tvae_objective = TVAEObjective(data)
    study = optuna.create_study(direction='maximize')
    study.optimize(tvae_objective, n_trials=n_trials,
                   callbacks=[wandbc], n_jobs=n_jobs)
    wandb.log({f'best_params_tvae_{dataset_name}': study.best_params})

    tvae = TVAE(**TVAEObjective.params_to_kwargs(study.best_params))
    tvae.fit(data)
    tvae.save(best_model_path + '/tvae_best.pkl')
    wandb.finish()
