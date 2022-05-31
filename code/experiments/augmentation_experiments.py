from augmentations.augmentations import Augmentator
from pipelines.common_pipeline import CommonPipeline


class AugmentationExperiment:
    def __init__(self, augmentation_types_with_proportions: dict[str, float],
                 augmentators: dict[str, Augmentator],
                 metric_names_mapping: dict[str, str],
                 pipeline: CommonPipeline):
        self.metric_names_mapping = metric_names_mapping
        self.augmentation_types_with_proportions = augmentation_types_with_proportions
        self.augmentators = augmentators
        self.pipeline = pipeline

    def start(self):
        return self.pipeline.launch_full_cv(self.augmentation_types_with_proportions,
                                            self.augmentators,
                                            self.metric_names_mapping)
