import sys

import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from xgboost import XGBClassifier, XGBRegressor

import wandb as wandb
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import wandb
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from augmentations.augs_runners import run_objectives
from experiments.augmentation_experiments import Augmentator
from experiments.experiment_runners import run_full_pipeline_with_augs, run_pipeline_no_augs
from pipelines.catboost_pipelines import CatboostClassificationPipeline, CatboostRegressionPipeline
from pipelines.lgbm_pipelines import LGBMClassificationPipeline, LGBMRegressionPipeline
from pipelines.tabnet_pipelines import TabNetClassificationPipeline, TabNetRegressionPipeline, TabNetPretrainingPipeline
from pipelines.xgboost_pipelines import XGBClassificationPipeline, XGBRegressionPipeline
from utils.config_utils import common_setup_and_config
from utils.datasets import CommonDataset
from utils.tabnet_hyperparams import get_model_with_hyperparams, _get_kwargs

if __name__ == "__main__":
    args = sys.argv
    SEED = 7575
    config = common_setup_and_config(args)
    task_type = config['task_type']

    if task_type == 'augs':
        dataset_name = config['dataset_name']
        dataset_path = config['dataset_path']
        best_path = config['best_path']
        percentage_selection = config['percentage_selection']
        dataset = CommonDataset(path=dataset_path,
                                dataset_type='classification',
                                dataset_name=dataset_name)
        run_objectives(dataset, percentage_selection=percentage_selection,
                       best_model_path=best_path,
                       project_str=f"optuna-wandb-augs-docker-{dataset_name}")
    elif task_type == 'learn':
        dataset_name = config['dataset_name']
        dataset_path = config['dataset_path']
        tvae_best_path = config['tvae_best_path']
        ctgan_best_path = config['ctgan_best_path']
        model_type = config['model_type']
        dataset_type = config['dataset_type']

        ###
        project_name = f'{model_type}-docker-launch-{dataset_name}'
        dataset = CommonDataset(path=dataset_path,
                                dataset_type=dataset_type,
                                dataset_name=dataset_name)
        if dataset_type == 'classification':

            if model_type == 'tabnet':
                model, kwargs = get_model_with_hyperparams(dataset_name)
                pipeline = TabNetClassificationPipeline(model, dataset,
                                                        wandb, project_name,
                                                        None, **kwargs)
                if 'syn' in dataset_name:
                    mapping = {'val_0_auc': ('max', 'auc')}
                else:
                    mapping = {'val_0_accuracy': ('max', 'accuracy')}

            elif model_type == 'catboost':
                model = CatBoostClassifier(eval_metric='AUC' if 'syn' in dataset_name else 'Accuracy')
                kwargs = {}
                pipeline = CatboostClassificationPipeline(model, dataset,
                                                          wandb, project_name,
                                                          None, **kwargs)
                if 'syn' in dataset_name:
                    mapping = {'AUC': ('max', 'auc')}
                else:
                    mapping = {'Accuracy': ('max', 'accuracy')}

            elif model_type == 'lgbm':
                model = LGBMClassifier()
                kwargs = {}
                pipeline = LGBMClassificationPipeline(model, dataset,
                                                      wandb, project_name,
                                                      None, **kwargs)
                if 'syn' in dataset_name:
                    mapping = {'auc': ('max', 'auc')}
                else:
                    mapping = {'accuracy': ('max', 'accuracy')}

            elif model_type == 'xgb':
                model = XGBClassifier()
                kwargs = {}
                pipeline = XGBClassificationPipeline(model, dataset,
                                                     wandb, project_name,
                                                     None, **kwargs)
                if 'syn' in dataset_name:
                    mapping = {'auc': ('max', 'auc')}
                else:
                    mapping = {'accuracy': ('max', 'accuracy')}


        elif dataset_type == 'regression':

            if model_type == 'tabnet':
                model, kwargs = get_model_with_hyperparams(dataset_name)
                pipeline = TabNetRegressionPipeline(model, dataset,
                                                    wandb, project_name,
                                                    None, **kwargs)
                mapping = {'val_0_mse': ('min', 'mse')}
            elif model_type == 'catboost':
                model = CatBoostRegressor(eval_metric='RMSE')
                kwargs = {}
                pipeline = CatboostRegressionPipeline(model, dataset,
                                                      wandb, project_name,
                                                      None, **kwargs)
                mapping = {'RMSE': ('min', 'rmse')}
            elif model_type == 'lgbm':
                model = LGBMRegressor()
                kwargs = {}
                pipeline = LGBMRegressionPipeline(model, dataset,
                                                  wandb, project_name,
                                                  None, **kwargs)
                mapping = {'l2': ('min', 'mse')}

            elif model_type == 'xgb':
                model = XGBRegressor()
                kwargs = {}
                pipeline = XGBRegressionPipeline(model, dataset,
                                                 wandb, project_name,
                                                 None, **kwargs)

                mapping = {'rmse': ('min', 'rmse')}

        best_augmentators = {
            'SMOTE': Augmentator(None, aug_type='SMOTE'),  # всегда первым
            'CTGAN': Augmentator(ctgan_best_path, aug_type='CTGAN'),
            'TVAE': Augmentator(tvae_best_path, aug_type='TVAE')
        }
        warm_start = None
        if 'warm_start' in config.keys():
            warm_start = config['warm_start']
        print("Warm start:", warm_start)
        run_full_pipeline_with_augs(pipeline, best_augmentators,
                                    mapping, warm_start)
    elif task_type == 'pretraining':

        dataset_name = config['dataset_name']
        dataset_path = config['dataset_path']
        project_name = f'pretraining-docker-launch-{dataset_name}'
        dataset = CommonDataset(path=dataset_path,
                                dataset_type='classification',
                                dataset_name=dataset_name)
        kwargs = _get_kwargs(dataset_name)
        kwargs.pop('eval_metric')

        pipeline = TabNetPretrainingPipeline(dataset, wandb,
                                             project_name, f'pretraining/{dataset_name}',
                                             **kwargs)
        print('launching pretraining')
        pipeline.launch_pretraining()
    elif task_type == 'with_pretraining':
        dataset_name = config['dataset_name']
        dataset_path = config['dataset_path']
        dataset_type = config['dataset_type']
        path_to_pretrained = config['path_to_pretrained']
        project_name = f'tabnet_with_pretraining-docker-launch-{dataset_name}'
        dataset = CommonDataset(path=dataset_path,
                                dataset_type=dataset_type,
                                dataset_name=dataset_name)
        model, kwargs = get_model_with_hyperparams(dataset_name)
        pretrained_model = TabNetPretrainer()
        pretrained_model.load_model(path_to_pretrained)
        print(pretrained_model)
        if pretrained_model == None:
            wandb.init()
            wandb.alert('NO LOADED PRETRAINED MODEL')
        kwargs['from_unsupervised'] = pretrained_model
        if dataset_type == 'classification':
            pipeline = TabNetClassificationPipeline(model, dataset,
                                                    wandb, project_name,
                                                    None, **kwargs)
            if 'syn' in dataset_name:
                mapping = {'val_0_auc': ('max', 'auc')}
            else:
                mapping = {'val_0_accuracy': ('max', 'accuracy')}
        else:
            pipeline = TabNetRegressionPipeline(model, dataset,
                                                wandb, project_name,
                                                None, **kwargs)
            mapping = {'val_0_mse': ('min', 'mse')}
        run_pipeline_no_augs(pipeline, mapping)
