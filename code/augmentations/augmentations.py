from sdv.tabular import CTGAN, TVAE
from sdv.tabular.base import DISABLE_TMP_FILE
import torch
from pytorch_tabnet.utils import define_device
import numpy as np


class Augmentator:
    aug_types = ['CTGAN', 'TVAE', 'SMOTE']

    def __init__(self, path_to_model, aug_type):
        """

        @param path_to_model:
        @param aug_type:
        """
        self.aug_type = aug_type
        if aug_type not in Augmentator.aug_types:
            raise ValueError('Wrong type')
        elif aug_type == 'CTGAN':
            self.model = CTGAN.load(path_to_model)
        elif aug_type == 'TVAE':
            self.model = TVAE.load(path_to_model)
        else:
            self.model = ClassificationSMOTE()

    def gen_samples(self, N, X=None, y=None, dataset_type='classification'):
        if self.aug_type == 'TVAE' or self.aug_type == 'CTGAN':
            return self.model.sample(N, output_file_path=DISABLE_TMP_FILE)
        else:
            assert X is not None
            if dataset_type == 'classification':
                self.model = ClassificationSMOTE(p=N / len(X))
            else:
                self.model = RegressionSMOTE(p=N / len(X))
            return self.model(X, y)


class RegressionSMOTE():
    """
    Apply SMOTE
    This will average a percentage p of the elements in the batch with other elements.
    The target will be averaged as well (this might work with binary classification
    and certain loss), following a beta distribution.
    """

    def __init__(self, device_name="auto", p=0.8, alpha=0.5, beta=0.5, seed=0):
        ""
        self.seed = seed
        self._set_seed()
        self.device = define_device(device_name)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        if (p < 0.) or (p > 1.0):
            raise ValueError("Value of p should be between 0. and 1.")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return

    def __call__(self, X, y):
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        batch_size = X.shape[0]
        random_values = torch.rand(batch_size, device=self.device)
        idx_to_change = random_values < self.p

        # ensure that first element to switch has probability > 0.5
        np_betas = np.random.beta(self.alpha, self.beta, batch_size) / 2 + 0.5
        random_betas = torch.from_numpy(np_betas).to(self.device).float()
        index_permute = torch.randperm(batch_size, device=self.device)
        X = X.float()
        X[idx_to_change] = random_betas[idx_to_change, None] * X[idx_to_change]
        X[idx_to_change] += (1 - random_betas[idx_to_change, None]) * X[index_permute][idx_to_change].view(
            X[idx_to_change].size())  # noqa

        y[idx_to_change] = random_betas[idx_to_change, None] * y[idx_to_change]
        y[idx_to_change] += (1 - random_betas[idx_to_change, None]) * y[index_permute][idx_to_change].view(
            y[idx_to_change].size())  # noqa

        return X.numpy(), y.numpy()


class ClassificationSMOTE():
    """
    Apply SMOTE for classification tasks.
    This will average a percentage p of the elements in the batch with other elements.
    The target will stay unchanged and keep the value of the most important row in the mix.
    """

    def __init__(self, device_name="auto", p=0.8, alpha=0.5, beta=0.5, seed=0):
        ""
        self.seed = seed
        self._set_seed()
        self.device = define_device(device_name)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        if (p < 0.) or (p > 1.0):
            raise ValueError("Value of p should be between 0. and 1.")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return

    def __call__(self, X, y):
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        batch_size = X.shape[0]
        random_values = torch.rand(batch_size, device=self.device)
        idx_to_change = random_values < self.p

        # ensure that first element to switch has probability > 0.5
        np_betas = np.random.beta(self.alpha, self.beta, batch_size) / 2 + 0.5
        random_betas = torch.from_numpy(np_betas).to(self.device).float()
        index_permute = torch.randperm(batch_size, device=self.device)
        X = X.float()
        X[idx_to_change] = random_betas[idx_to_change, None] * X[idx_to_change]
        X[idx_to_change] += (1 - random_betas[idx_to_change, None]) * X[index_permute][idx_to_change].view(
            X[idx_to_change].size())  # noqa

        return X.numpy(), y.numpy()
