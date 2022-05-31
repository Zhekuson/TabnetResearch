import gc

import numpy as np
import pandas as pd
from mat4py import loadmat
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from augmentations.augmentations import Augmentator


class CommonDataset(Dataset):
    """
    Available datasets
    """

    dataset_names = ['forest_cover', 'higgs', 'pokerhand', 'rossmann_stores_sales', 'sarcos',
                     'syn1', 'syn2', 'syn3', 'syn4', 'syn5', 'syn6', 'airfoil', 'dexter',
                     'p53', 'stars', 'slice_localization']

    def __init__(self, path: str, dataset_type: str, dataset_name: str, immediate_load=True):
        """
        @param path: путь
        @param dataset_type: тип(регрессия/классификация)
        @param dataset_name: название
        @param immediate_load: загрузить при инициализации или отложить
        """
        self.copy_y = None
        self.copy_X = None
        self.y = None
        self.X = None
        self._loaded = False
        self.path = path

        if dataset_type in ['classification', 'regression']:
            self.type = dataset_type
        else:
            raise ValueError('Wrong dataset type')

        if dataset_name in CommonDataset.dataset_names:
            self.dataset_name = dataset_name
        else:
            raise ValueError('Unsupported dataset name')

        if immediate_load:
            self._load_data()

    def _load_data(self):
        self.special_test_set = False
        if self.dataset_name == 'forest_cover':
            data = pd.read_csv(self.path)
            self.y = data['Cover_Type'].apply(lambda x: x-1).to_numpy()
            self.X = data.drop(columns=['Cover_Type']).to_numpy()

        elif self.dataset_name == 'higgs':
            data = pd.read_csv(self.path, names=list(np.arange(start=0, stop=29)))
            self.y = data[0].to_numpy()
            self.X = data.drop(columns=[0]).to_numpy()

        elif self.dataset_name == 'sarcos':
            self.special_test_set = True
            data = loadmat(self.path + '/sarcos_inv.mat')
            data = pd.DataFrame(data['sarcos_inv'])
            self.y = data[21].to_numpy().reshape(-1, 1)
            self.X = data.drop(columns=np.arange(start=21, stop=len(data.columns), step=1)).to_numpy()

            data = loadmat(self.path + '/sarcos_inv_test.mat')
            data = pd.DataFrame(data['sarcos_inv_test'])
            self.special_test_y = data[21].to_numpy().reshape(-1, 1)
            self.special_test_X = data.drop(columns=np.arange(start=21, stop=len(data.columns), step=1)).to_numpy()

        elif self.dataset_name == 'pokerhand':
            self.special_test_set = True
            data = pd.read_csv(self.path + '/poker-hand-training-true.csv', names=np.arange(0, 11))
            self.y = data[10].to_numpy()
            self.X = data.drop(columns=[10]).to_numpy()

            data = pd.read_csv(self.path + '/poker-hand-testing.csv', names=np.arange(0, 11))
            self.special_test_y = data[10].to_numpy()
            self.special_test_X = data.drop(columns=[10]).to_numpy()

        elif self.dataset_name == 'rossmann_stores_sales':
            sales_train_df = pd.read_csv(self.path + '/train.csv')
            store_info_df = pd.read_csv(self.path + '/store.csv')
            sales_train_all_df = pd.merge(sales_train_df, store_info_df, how='inner', on='Store')
            self.y = sales_train_all_df['Sales'].to_numpy()
            self.X = sales_train_all_df.drop(columns=['Sales']).to_numpy()

        elif self.dataset_name == 'airfoil':
            data = pd.read_csv(self.path)
            self.y = data['SSPL'].to_numpy().reshape(-1, 1)
            self.X = data.drop(columns=['SSPL']).to_numpy()

        elif self.dataset_name == 'stars':
            data = pd.read_csv(self.path)
            self.y = data['Type'].to_numpy().astype('int')
            self.X = data.drop(columns=['Type']).to_numpy()

            self.ohe_color = OneHotEncoder(sparse=False)
            transformed = self.ohe_color.fit_transform(self.X[:, 4].reshape(-1, 1))
            self.X = np.delete(np.hstack((self.X, transformed)), 4, 1)

            self.ohe_class = OneHotEncoder(sparse=False)
            transformed = self.ohe_class.fit_transform(self.X[:, 4].reshape(-1, 1))
            self.X = np.delete(np.hstack((self.X, transformed)), 4, 1).astype('float')

        elif self.dataset_name == 'slice_localization':
            data = pd.read_csv(self.path)
            self.y = data['reference'].to_numpy()
            self.X = data.drop(columns=['reference']).to_numpy()

        elif self.dataset_name == 'p53':
            data = pd.read_csv(self.path + '/K8.data', names=np.arange(0, 5410))
            self.y = data[5408].apply(lambda x: 0 if x == 'inactive' else 1).to_numpy()
            self.X = data.drop(columns=[5408]).to_numpy()

        elif 'syn' in self.dataset_name:
            data = pd.read_csv(self.path)
            self.y = data['11'].to_numpy()
            self.X = data.drop(columns=['11']).to_numpy()

        self._loaded = True
        gc.collect()
        pass

    def _split_gen_samples(self, data):
        if self.dataset_name == 'forest_cover':
            y = data['54'].apply(lambda x: x-1).to_numpy()
            X = data.drop(columns=['54']).to_numpy()
        elif self.dataset_name == 'higgs':
            y = data['28'].to_numpy()
            X = data.drop(columns=['28']).to_numpy()
        elif self.dataset_name == 'sarcos':
            y = data['21'].to_numpy().reshape(-1,1)
            X = data.drop(columns=['21']).to_numpy()
        elif self.dataset_name == 'pokerhand':
            y = data['10'].to_numpy()
            X = data.drop(columns=['10']).to_numpy()
        elif self.dataset_name == 'rossmann_stores_sales':
            y = data['3'].to_numpy()
            X = data.drop(columns=['3']).to_numpy()

        elif self.dataset_name == 'airfoil':
            y = data['5'].to_numpy().reshape(-1, 1)
            X = data.drop(columns=['5']).to_numpy()

        elif self.dataset_name == 'stars':
            y = data['6'].to_numpy().astype('int')
            X = data.drop(columns=['6']).to_numpy()
            transformed = self.ohe_color.transform(X[:, 4].reshape(-1, 1))
            X = np.delete(np.hstack((X, transformed)), 4, 1)
            transformed = self.ohe_class.transform(X[:, 4].reshape(-1, 1))
            X = np.delete(np.hstack((X, transformed)), 4, 1).astype('float')

        elif self.dataset_name == 'slice_localization':
            y = data['384'].to_numpy()
            X = data.drop(columns=['384']).to_numpy()

        elif self.dataset_name == 'p53':
            y = data['5409'].to_numpy()
            X = data.drop(columns=['5409']).to_numpy()

        elif 'syn' in self.dataset_name:
            y = data['11'].to_numpy()
            X = data.drop(columns=['11']).to_numpy()
        gc.collect()
        return X, y

    def __getitem__(self, index) -> T_co:
        item = (self.X[index], self.y[index])
        return item

    def __len__(self):
        return len(self.X)

    def add_augmentations(self, augmentator: Augmentator, N:int):
        """
        :param augmentator: аугментирующая модель
        :param N: число образцов
        """
        if augmentator.aug_type != 'SMOTE':
            aug_samples = augmentator.gen_samples(N)
            if len(aug_samples) > 0:
                aug_X, aug_y = self._split_gen_samples(aug_samples)
                self.X = np.append(self.X, aug_X, axis=0)
                self.y = np.append(self.y, aug_y, axis=0)
        else:
            self.copy_X = self.X.copy()
            self.copy_y = self.y.copy()
            self.X, self.y = augmentator.gen_samples(N, self.X, self.y, self.type)

    def remove_augmentations(self, N):
        """
        :param N: исходное количество добавленных образцов
        """
        if self.copy_X is not None:
            self.X = self.copy_X
            self.y = self.copy_y
            self.copy_X = None
            self.copy_y = None
        else:
            self.X = np.delete(self.X, np.arange(len(self.X) - N, len(self.X)), axis=0)
            self.y = np.delete(self.y, np.arange(len(self.y) - N, len(self.y)), axis=0)

