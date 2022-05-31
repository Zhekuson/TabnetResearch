import copy
from typing import List

import numpy as np
import wandb
from sklearn.model_selection import BaseCrossValidator

from utils.datasets import CommonDataset


class CommonPipeline:
    def __init__(self, model, dataset: CommonDataset, wandb_obj, wandb_project_name, BEST_MODELS_FOLDER, **kwargs):
        self.kwargs = kwargs
        self.model_copy = None
        self.model = model
        self.dataset = dataset
        self.wandb_project_name = wandb_project_name
        self.BEST_MODELS_FOLDER = BEST_MODELS_FOLDER
        self.wandb_obj = wandb_obj
        pass

    def launch_full_cv(self, *args, **kwargs):
        pass

    def _train_model(self, *args):
        pass

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
            wandb.define_metric(metric_names_mapping[metric_name][1])

        for result in results:
            log_dict = {}
            for metric_name in metric_names_mapping.keys():
                if metric_names_mapping[metric_name][0] == 'max':
                    best_value = max(result[metric_name])
                else:
                    best_value = min(result[metric_name])
                log_dict[metric_names_mapping[metric_name][1]] = best_value
            wandb.log(log_dict)
        run.finish()
        pass

    def _launch_cv(self, kf: BaseCrossValidator, augmentation_types_with_proportions, augmentators):
        try:
            results = []
            counter = 1
            for train, test in kf.split(self.dataset.X, self.dataset.y):
                run = self.wandb_obj.init(project=self.wandb_project_name,
                                          name=self._get_run_name_wandb(
                                              counter,
                                              augmentation_types_with_proportions)
                                          )
                self.wandb_obj.alert("Info", f"Started experiment \n" +
                                     f'Dataset:{self.dataset.dataset_name}'
                                     f"Configuration: {augmentation_types_with_proportions}")
                self.model_copy = copy.deepcopy(self.model)
                if augmentation_types_with_proportions is not None:
                    train_idx, total_N = self._add_augmentations(train,
                                                                 augmentation_types_with_proportions,
                                                                 augmentators)
                else:
                    train_idx, total_N = train, 0
                # train model
                result = self._train_model(train_idx, test)

                # get result
                self.wandb_obj.alert("Info", f"split num {counter} finished" +
                                     f"Here is what we have: {result}")
                results.append(result)

                # save model
                if self.BEST_MODELS_FOLDER is not None:
                    name = self._get_filename_for_saving(counter, augmentation_types_with_proportions)
                    self._save_model_copy(name)

                # after training

                if total_N > 0:
                    self.dataset.remove_augmentations(total_N)
                counter += 1
                run.finish()
            return results

        except BaseException as e:
            # in case something happened
            self.wandb_obj.alert("Error happened", str(e), level='ERROR')
            raise e

    def _get_filename_for_saving(self, counter: int,
                                 augmentation_types_with_proportions: dict[str, float]) -> str:
        aug_info = '_'.join([x[0] + '_' + str(x[1]) for x in augmentation_types_with_proportions.items()])
        return self.BEST_MODELS_FOLDER + f'/best_{self.dataset.dataset_name}_split_{counter}_' + aug_info

    def _get_run_name_wandb(self, counter: int,
                            augmentation_types_with_proportions: dict[str, float]):
        if augmentation_types_with_proportions is not None:
            aug_info = '_'.join([x[0] + '_' + str(x[1]) for x in augmentation_types_with_proportions.items()])
        else:
            aug_info = ''
        return f'{type(self).__name__}_{self.dataset.dataset_name}_split_{counter}_' + aug_info

    def _get_summary_name_wandb(self, augmentation_types_with_proportions: dict[str, float]):
        if augmentation_types_with_proportions is not None:
            aug_info = '_'.join([x[0] + '_' + str(x[1]) for x in augmentation_types_with_proportions.items()])
        else:
            aug_info = ""
        return f'{type(self).__name__}_{self.dataset.dataset_name}_' + aug_info + '_summary'

    def _add_augmentations(self, train, augmentation_types_with_proportions, augmentators):
        # augmentations
        train_idx = train
        train_length = len(train)
        total_N = 0
        for i, augmentator_name in enumerate(augmentators):
            proportion = augmentation_types_with_proportions[augmentator_name]
            augmentator = augmentators[augmentator_name]
            N = int(proportion * train_length)
            if N > 0:
                if augmentator_name != 'SMOTE':
                    idx = np.arange(len(self.dataset.X), len(self.dataset.X) + N)
                    train_idx = np.append(train, idx)
                    self.dataset.add_augmentations(augmentator, N)
                    total_N += N
                else:
                    self.dataset.add_augmentations(augmentator, N)
        return train_idx, total_N

    def _save_model_copy(self, path):
        pass
