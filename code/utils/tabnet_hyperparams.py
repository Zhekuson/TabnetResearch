import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

dataset_names = ['forest_cover', 'higgs', 'pokerhand', 'rossmann_stores_sales', 'sarcos',
                 'syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6', 'airfoil', 'dexter',
                 'p53', 'stars', 'slice_localization']


def _get_kwargs(dataset_name: str):
    if dataset_name == 'higgs':
        return {'eval_metric': ['accuracy'],  # 'auc'
                'batch_size': 1024, #16384,
                'virtual_batch_size': 512,
                'patience': 8,
                'max_epochs': 80
                }
    elif dataset_name == 'sarcos':
        return {'eval_metric': ['mse'],
                'batch_size': 4096,
                'virtual_batch_size': 512,
                'patience': 8,
                'max_epochs': 120
                }
    elif dataset_name == 'pokerhand':
        return {'eval_metric': ['accuracy'],
                'batch_size': 4096,
                'virtual_batch_size': 1024,
                'patience': 30,
                'max_epochs': 120
                }
    elif dataset_name == 'forest_cover':
        return {'eval_metric': ['accuracy'],
                'batch_size': 2048, #16384,
                'virtual_batch_size': 512,
                'patience': 10,
                'max_epochs': 100
                }
    elif dataset_name == 'rossmann_stores_sales':
        return {'eval_metric': ['mse'],
                'batch_size': 4096,
                'virtual_batch_size': 512,
                'patience': 3,
                'max_epochs': 100
                }
    elif 'syn' in dataset_name:
        return {'eval_metric': ['auc'],
                'batch_size': 3000,
                'virtual_batch_size': 100,
                'patience': 15,
                'max_epochs': 150
                }
    elif dataset_name == 'stars':
        return {'eval_metric': ['accuracy'],
                'batch_size': 64,
                'virtual_batch_size': 32,
                'patience': 15,
                'max_epochs': 150
                }
    elif dataset_name == 'airfoil':
        return {
            'eval_metric': ['mse'],
            'batch_size': 128,
            'virtual_batch_size': 32,
            'patience': 8,
            'max_epochs': 125
        }
    elif dataset_name == 'slice_localization':
        return {
            'eval_metric': ['mse'],
            'batch_size': 4096,
            'virtual_batch_size': 512,
            'patience': 3,
            'max_epochs': 100
        }


def get_model_with_hyperparams(dataset_name: str, SEED=7575):
    kwargs = _get_kwargs(dataset_name)
    if dataset_name == 'higgs':
        optimizer_params = {'lr': 0.02}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.9, 'step_size': 20}
        #n_d=24, n_a=26 n_steps=5
        model = TabNetClassifier(n_d=16, n_a=16, lambda_sparse=0.000001, n_steps=4, gamma=1.5, momentum=0.6,
                                 optimizer_params=optimizer_params,
                                 scheduler_params=scheduler_params,
                                 scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'stars':
        optimizer_params = {'lr': 0.03}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 5}
        model = TabNetClassifier(n_d=8, n_a=8, lambda_sparse=0.0001, n_steps=3, gamma=1.2, momentum=0.6,
                                 optimizer_params=optimizer_params,
                                 scheduler_params=scheduler_params,
                                 scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'sarcos':
        optimizer_params = {'lr': 0.01}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 6}
        model = TabNetRegressor(n_d=8, n_a=8, lambda_sparse=0.0001,
                                n_steps=3, gamma=1.2, momentum=0.9,
                                optimizer_params=optimizer_params,
                                scheduler_params=scheduler_params,
                                scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'pokerhand':
        optimizer_params = {'lr': 0.02}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 20}
        model = TabNetClassifier(n_d=16, n_a=16, lambda_sparse=0.000001,
                                 n_steps=4, gamma=1.5, momentum=0.95,
                                 optimizer_params=optimizer_params,
                                 scheduler_params=scheduler_params,
                                 scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'forest_cover':
        optimizer_params = {'lr': 0.02}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 20}
        # n_d=64, n_a=64 n_steps=5
        model = TabNetClassifier(n_d=16, n_a=16, lambda_sparse=0.0001,
                                 n_steps=5, gamma=1.5, momentum=0.7,
                                 optimizer_params=optimizer_params,
                                 scheduler_params=scheduler_params,
                                 scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'rossmann_stores_sales':
        optimizer_params = {'lr': 0.02}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 20}
        model = TabNetRegressor(n_d=32, n_a=32, lambda_sparse=0.001,
                                n_steps=5, gamma=1.2, momentum=0.8,
                                optimizer_params=optimizer_params,
                                scheduler_params=scheduler_params, scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'airfoil':
        optimizer_params = {'lr': 0.03}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 20}
        model = TabNetRegressor(n_d=4, n_a=4, lambda_sparse=0.001,
                                n_steps=5, gamma=1.2, momentum=0.8,
                                optimizer_params=optimizer_params,
                                scheduler_params=scheduler_params, scheduler_fn=scheduler, seed=SEED)
    elif 'syn' in dataset_name:
        optimizer_params = {'lr': 0.01}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.7, 'step_size': 20}
        momentum = 0.7
        if dataset_name == 'syn1':
            L = 0.02
            gamma = 2
            n_steps = 4
        elif dataset_name == 'syn2' or dataset_name == 'syn3':
            L = 0.01
            gamma = 2
            n_steps = 4
        else:
            L = 0.005
            gamma = 1.5
            n_steps = 5

        model = TabNetClassifier(n_d=16, n_a=16, lambda_sparse=L,
                                 n_steps=n_steps, gamma=gamma, momentum=momentum,
                                 optimizer_params=optimizer_params,
                                 scheduler_params=scheduler_params, scheduler_fn=scheduler, seed=SEED)
    elif dataset_name == 'slice_localization':
        optimizer_params = {'lr': 0.0001}
        scheduler = torch.optim.lr_scheduler.StepLR
        scheduler_params = {'gamma': 0.95, 'step_size': 20}
        model = TabNetClassifier(n_d=8, n_a=8, lambda_sparse=0.001,
                                 n_steps=4, gamma=1.2, momentum=0.8,
                                 optimizer_params=optimizer_params,
                                 scheduler_params=scheduler_params, scheduler_fn=scheduler, seed=SEED)

    return model, kwargs
