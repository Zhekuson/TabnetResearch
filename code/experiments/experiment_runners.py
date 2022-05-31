import itertools

import numpy as np
import pandas as pd
import wandb

from augmentations.augmentations import Augmentator
from experiments.augmentation_experiments import AugmentationExperiment
from pipelines.common_pipeline import CommonPipeline


def get_proportions_iterable(n: int):
    prop = np.arange(start=0, stop=6) / 10
    iterable = itertools.product(prop, repeat=n)
    return iterable


def run_pipeline_no_augs(pipeline: CommonPipeline,
                         metric_names_mapping: dict[str, (str, str)]):
    pipeline.launch_full_cv(None, None, metric_names_mapping)


def run_full_pipeline_with_augs(pipeline: CommonPipeline,
                                best_augmentators: dict[str, Augmentator],
                                metric_names_mapping: dict[str, (str, str)],
                                warm_start=None):
    augs = list(best_augmentators.keys())
    proportions = get_proportions_iterable(len(augs))

    for proportion in proportions:
        augmentations_proportions = {}
        if warm_start is not None:
            if list(proportion) < warm_start:
                print(f"skip {proportion}")
                continue
        for i, item in enumerate(augs):
            augmentations_proportions[item] = proportion[i]
        experiment = AugmentationExperiment(augmentations_proportions, best_augmentators,
                                            metric_names_mapping,
                                            pipeline)
        experiment.start()


def get_summary(wandb_obj, project_name, path_to_save=None):
    api = wandb_obj.Api()
    entity = "zhekuson"  # set to your entity and project
    runs = api.runs(entity + "/" + project_name)
    summary_list, name_list = [], []
    for run in runs:
        if 'summary' not in run.name:
            continue
        summary_list.append(
            {k: list(v) for k, v in run.history().items()
             if not k.startswith('_')})
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "name": name_list
    })

    if path_to_save is None:
        path_to_save = project_name + '.csv'
    runs_df.to_csv(path_to_save)
