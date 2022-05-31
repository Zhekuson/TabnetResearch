import numpy as np
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import StratifiedKFold, KFold
from torch import nn
import wandb
from pytorch_tabnet.callbacks import Callback

from augmentations.augmentations import Augmentator
from pipelines.common_pipeline import CommonPipeline
from utils.datasets import CommonDataset


class WandbTabNetCallback(Callback):
    def __init__(self, wandb_obj, params):
        super().__init__()
        self.wandb = wandb_obj
        wandb.config.update({
            'parameters': params
        })

    def set_params(self, params):
        super().set_params(params)

    def set_trainer(self, model):
        super().set_trainer(model)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        if logs is not None:
            wandb.log(logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if logs is not None:
            wandb.log(logs)

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if logs is not None:
            wandb.log(logs)


class TabNetClassificationPipeline(CommonPipeline):

    def __init__(self, model, dataset: CommonDataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs):
        super().__init__(model, dataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs)

    def launch_full_cv(self,
                       augmentation_types_with_proportions: dict[str, float],
                       augmentators: dict[str, Augmentator],
                       metric_names_mapping: dict[str, (str, str)],
                       n_splits=5,
                       seed=7575):
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        results = self._launch_cv(kf, augmentation_types_with_proportions, augmentators)
        self._log_summary(augmentation_types_with_proportions, results, metric_names_mapping)
        return results

    def _train_model(self, train_idx, test_idx):
        wandb_callback = WandbTabNetCallback(self.wandb_obj, self.model.get_params())
        self.model_copy.fit(
            self.dataset[train_idx][0], self.dataset[train_idx][1],
            eval_set=[(self.dataset[test_idx][0], self.dataset[test_idx][1])],
            loss_fn=nn.CrossEntropyLoss(),
            callbacks=[wandb_callback],
            **self.kwargs
        )
        return self.model_copy.history

    def _save_model_copy(self, path):
        self.model_copy.save_model(path)


class TabNetRegressionPipeline(CommonPipeline):

    def __init__(self, model, dataset: CommonDataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs):
        super().__init__(model, dataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs)

    def launch_full_cv(self,
                       augmentation_types_with_proportions: dict[str, float],
                       augmentators: dict[str, Augmentator],
                       metric_names_mapping: dict[str, str],
                       n_splits=5,
                       seed=7575):
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        results = self._launch_cv(kf, augmentation_types_with_proportions, augmentators)
        self._log_summary(augmentation_types_with_proportions, results, metric_names_mapping)
        return results

    def _train_model(self, train_idx, test_idx):
        wandb_callback = WandbTabNetCallback(self.wandb_obj, self.model.get_params())
        self.model_copy.fit(
            self.dataset[train_idx][0], self.dataset[train_idx][1],
            eval_set=[(self.dataset[test_idx][0], self.dataset[test_idx][1])],
            loss_fn=nn.MSELoss(),
            callbacks=[wandb_callback],
            **self.kwargs
        )
        return self.model_copy.history

    def _save_model_copy(self, path):
        self.model_copy.save_model(path)


class TabNetPretrainingPipeline:
    def __init__(self, dataset: CommonDataset, wandb_obj,
                 wandb_project_name, model_save_path, **kwargs):
        self.model = None
        self.model_save_path = model_save_path
        self.wandb_project_name = wandb_project_name
        self.wandb_obj = wandb_obj
        self.dataset = dataset
        self.kwargs = kwargs

    def launch_pretraining(self, pretraining_ratio=0.8, SEED=7575):
        self.wandb_obj.init(self.wandb_project_name, 'fdcf')
        np.random.seed(SEED)
        length = len(self.dataset.X)
        N_valid = int(length * 0.2)
        valid_idx = np.random.choice(length, N_valid, replace=False)
        X_train = np.delete(self.dataset.X, valid_idx, axis=0)
        X_valid = self.dataset.X[valid_idx]
        self.model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax"
        )
        wandb_callback = WandbTabNetCallback(self.wandb_obj, self.model.get_params())
        self.model.fit(
            X_train=X_train,
            eval_set=[X_valid],
            pretraining_ratio=pretraining_ratio,
            callbacks=[wandb_callback],
            **self.kwargs
        )
        self.model.save_model(self.model_save_path)

