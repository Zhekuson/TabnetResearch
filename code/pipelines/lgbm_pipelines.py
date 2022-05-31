from typing import Tuple, List

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, auc, roc_auc_score
import numpy as np
from augmentations.augmentations import Augmentator
from pipelines.common_pipeline import CommonPipeline
from utils.datasets import CommonDataset
from wandb.lightgbm import wandb_callback


def accuracy_for_lgbm(
        labels: np.ndarray, preds: np.ndarray
) -> Tuple[str, float, bool]:
    preds = preds.reshape(preds.shape[0] // labels.shape[0], -1)
    preds = preds.argmax(axis=0)
    acc = accuracy_score(labels, preds)
    # # eval_name, eval_result, is_higher_better
    return 'accuracy', acc, True


def lgbm_acc_bin(labels, preds):
    preds = preds > 0.5
    preds = preds.astype(int)
    return 'accuracy', accuracy_score(labels, preds), True


def auc_lgbm(
        labels: np.ndarray, preds: np.ndarray
):
    sorted_index = np.argsort(preds)
    preds = preds[sorted_index]
    labels = labels[sorted_index]
    return 'auc', roc_auc_score(labels, preds), True


class LGBMClassificationPipeline(CommonPipeline):

    def __init__(self, model: LGBMClassifier, dataset: CommonDataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER,
                 **kwargs):
        super().__init__(model, dataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs)

    def _log_summary(self, augmentation_types_with_proportions: dict[str, float],
                     results: List[dict],
                     metric_names_mapping: dict[str, (str, str)]):
        run = self.wandb_obj.init(project=self.wandb_project_name,
                                  name=self._get_summary_name_wandb(augmentation_types_with_proportions))
        for metric_name in metric_names_mapping.keys():
            self.wandb_obj.define_metric(metric_names_mapping[metric_name][1])

        for result in results:
            log_dict = {}
            for metric_name in metric_names_mapping.keys():
                if metric_names_mapping[metric_name][0] == 'max':
                    best_value = max(result['valid_0'][metric_name])
                else:
                    best_value = min(result['valid_0'][metric_name])
                log_dict[metric_names_mapping[metric_name][1]] = best_value
            self.wandb_obj.log(log_dict)
        run.finish()
        pass

    def launch_full_cv(self,
                       augmentation_types_with_proportions: dict[str, float],
                       augmentators: dict[str, Augmentator],
                       metric_names_mapping: dict[str, (str, str)],
                       n_splits=5, seed=7575):
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        results = self._launch_cv(kf, augmentation_types_with_proportions, augmentators)
        self._log_summary(augmentation_types_with_proportions, results, metric_names_mapping)
        return results

    def _train_model(self, train_idx, test_idx):
        if self.dataset.dataset_name == 'higgs':
            metric = lgbm_acc_bin
        else:
            metric = auc_lgbm if 'syn' in self.dataset.dataset_name else accuracy_for_lgbm
        res = self.model_copy.fit(self.dataset[train_idx][0],
                                  self.dataset[train_idx][1],
                                  eval_set=(self.dataset[test_idx][0],
                                            self.dataset[test_idx][1]),
                                  eval_metric=metric,
                                  callbacks=[wandb_callback()],
                                  **self.kwargs
                                  )
        return res.evals_result_

    def _save_model_copy(self, path):
        super()._save_model_copy(path)


class LGBMRegressionPipeline(CommonPipeline):

    def __init__(self, model: LGBMRegressor, dataset: CommonDataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER,
                 **kwargs):
        super().__init__(model, dataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs)

    def _log_summary(self, augmentation_types_with_proportions: dict[str, float],
                     results: List[dict],
                     metric_names_mapping: dict[str, (str, str)]):
        run = self.wandb_obj.init(project=self.wandb_project_name,
                                  name=self._get_summary_name_wandb(augmentation_types_with_proportions))
        for metric_name in metric_names_mapping.keys():
            self.wandb_obj.define_metric(metric_names_mapping[metric_name][1])

        for result in results:
            log_dict = {}
            for metric_name in metric_names_mapping.keys():
                if metric_names_mapping[metric_name][0] == 'max':
                    best_value = max(result['valid_0'][metric_name])
                else:
                    best_value = min(result['valid_0'][metric_name])
                log_dict[metric_names_mapping[metric_name][1]] = best_value
            self.wandb_obj.log(log_dict)
        run.finish()
        pass

    def launch_full_cv(self,
                       augmentation_types_with_proportions: dict[str, float],
                       augmentators: dict[str, Augmentator],
                       metric_names_mapping: dict[str, (str, str)],
                       n_splits=5, seed=7575):
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        results = self._launch_cv(kf, augmentation_types_with_proportions, augmentators)
        self._log_summary(augmentation_types_with_proportions, results, metric_names_mapping)
        return results

    def _train_model(self, train_idx, test_idx):
        res = self.model_copy.fit(self.dataset[train_idx][0],
                                  self.dataset[train_idx][1],
                                  eval_set=(self.dataset[test_idx][0],
                                            self.dataset[test_idx][1]),
                                  eval_metric='mse',
                                  callbacks=[wandb_callback()],
                                  **self.kwargs
                                  )
        return res.evals_result_

    def _save_model_copy(self, path):
        super()._save_model_copy(path)
