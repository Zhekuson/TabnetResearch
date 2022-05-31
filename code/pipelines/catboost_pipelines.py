import copy
from typing import List

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from wandb.integration.catboost import WandbCallback

from augmentations.augmentations import Augmentator
from pipelines.common_pipeline import CommonPipeline
from utils.datasets import CommonDataset


class CatboostClassificationPipeline(CommonPipeline):

    def __init__(self, model: CatBoostClassifier,
                 dataset: CommonDataset,
                 wandb_obj,
                 wandb_project_name,
                 BEST_MODELS_FOLDER,
                 **kwargs):
        super().__init__(model, dataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs)

    def _log_summary(self, augmentation_types_with_proportions: dict[str, float],
                     results: List[dict],
                     metric_names_mapping: dict[str, (str, str)]):

        """

        :param augmentation_types_with_proportions:
        :param results:
        :param metric_names_mapping: кривое имя -> (min/max, нормальное имя)
        :rtype: object
        """
        run = self.wandb_obj.init(project=self.wandb_project_name,
                                  name=self._get_summary_name_wandb(augmentation_types_with_proportions))
        for metric_name in metric_names_mapping.keys():
            self.wandb_obj.define_metric(metric_names_mapping[metric_name][1])

        for result in results:
            log_dict = {}
            for metric_name in metric_names_mapping.keys():
                best_value = result['validation'][metric_name]
                log_dict[metric_names_mapping[metric_name][1]] = best_value
            self.wandb_obj.log(log_dict)
        run.finish()
        pass

    def launch_full_cv(self,
                       augmentation_types_with_proportions: dict[str, float],
                       augmentators: dict[str, Augmentator],
                       metric_names_mapping: dict[str, (str, str)],
                       n_splits=5, seed=7575, ):
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        results = self._launch_cv(kf, augmentation_types_with_proportions, augmentators)
        self._log_summary(augmentation_types_with_proportions, results, metric_names_mapping)
        return results

    def _train_model(self, train_idx, test_idx):
        self.model_copy.fit(self.dataset[train_idx][0],
                            self.dataset[train_idx][1],
                            eval_set=(self.dataset[test_idx][0],
                                      self.dataset[test_idx][1]),
                            use_best_model=True,
                            callbacks=[WandbCallback()],
                            **self.kwargs)
        return self.model_copy.get_best_score()

    def _save_model_copy(self, path):
        self.model_copy.save_model(path)


class CatboostRegressionPipeline(CommonPipeline):

    def __init__(self, model: CatBoostRegressor,
                 dataset: CommonDataset,
                 wandb_obj,
                 wandb_project_name,
                 BEST_MODELS_FOLDER, **kwargs):
        super().__init__(model, dataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs)

    def launch_full_cv(self,
                       augmentation_types_with_proportions: dict[str, float],
                       augmentators: dict[str, Augmentator],
                       metric_names_mapping: dict[str, (str, str)],
                       n_splits=5, seed=7575):
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        results = self._launch_cv(kf, augmentation_types_with_proportions, augmentators)
        self._log_summary(augmentation_types_with_proportions, results, metric_names_mapping)
        return results

    def _log_summary(self, augmentation_types_with_proportions: dict[str, float],
                     results: List[dict],
                     metric_names_mapping: dict[str, (str, str)]):

        """

        :param augmentation_types_with_proportions:
        :param results:
        :param metric_names_mapping: кривое имя -> (min/max, нормальное имя)
        :rtype: object
        """
        run = self.wandb_obj.init(project=self.wandb_project_name,
                                  name=self._get_summary_name_wandb(augmentation_types_with_proportions))
        for metric_name in metric_names_mapping.keys():
            self.wandb_obj.define_metric(metric_names_mapping[metric_name][1])

        for result in results:
            log_dict = {}
            for metric_name in metric_names_mapping.keys():
                best_value = result['validation'][metric_name]
                log_dict[metric_names_mapping[metric_name][1]] = best_value
            self.wandb_obj.log(log_dict)
        run.finish()
        pass

    def _train_model(self, train_idx, test_idx):
        self.model_copy.fit(self.dataset[train_idx][0],
                            self.dataset[train_idx][1],
                            eval_set=(self.dataset[test_idx][0],
                                      self.dataset[test_idx][1]),
                            use_best_model=True,
                            callbacks=[WandbCallback()],
                            **self.kwargs)
        return self.model_copy.get_best_score()

    def _save_model_copy(self, path):
        self.model_copy.save_model(path)
