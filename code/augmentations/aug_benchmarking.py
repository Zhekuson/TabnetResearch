import numpy as np
import pandas as pd
import sdv.evaluation
import warnings
from augmentations import Augmentator
from utils.datasets import CommonDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def run_single_comparison(dataset: CommonDataset, augmentator: Augmentator, percentage=1):
    N = int(percentage * len(dataset.X))
    idx = np.random.choice(len(dataset.X), size=N, replace=False)
    y = dataset.y[idx] if len(dataset.y[idx].shape) == 2 else np.expand_dims(dataset.y[idx], axis=1)
    data = pd.DataFrame(np.hstack((dataset.X[idx], y)),
                        columns=list(map(str, np.arange(start=0, stop=len(dataset.X[0]) + 1))))
    syn_data = augmentator.gen_samples(N)
    aug_X, aug_y = dataset._split_gen_samples(syn_data)
    aug_y = aug_y if len(aug_y.shape) == 2 else np.expand_dims(aug_y, axis=1)
    syn_data = pd.DataFrame(np.hstack((aug_X, aug_y)),
                            columns=list(map(str, np.arange(start=0, stop=len(dataset.X[0]) + 1))))
    return sdv.evaluation.evaluate(syn_data, real_data=data)


def run_comparisons(dataset, augmentator, K):
    eval_results = []
    for _ in tqdm(range(K)):
        eval_results.append(run_single_comparison(dataset, augmentator))
    eval_results = np.array(eval_results)
    mu = np.mean(eval_results)
    sigma = np.std(eval_results)
    return mu, sigma


if __name__ == "__main__":
    dataset_names = []
    paths_to_datasets = []

    paths_to_augs = [f'./best_aug_models/{dataset_name}_best_models'
                     for dataset_name in dataset_names]

    ctgan_results = []
    tvae_results = []
    K = 10
    for i, name in enumerate(dataset_names):
        print(f'dataset:{name}')
        dataset = CommonDataset(dataset_name=name, path=paths_to_datasets[i],
                                dataset_type='regression')
        augmentator = Augmentator(path_to_model=paths_to_augs[i] + '/ctgan_best.pkl', aug_type='CTGAN')
        mu, sigma = run_comparisons(dataset, augmentator, K)
        ctgan_results.append((mu, sigma))
        print((mu, sigma))
        augmentator = Augmentator(path_to_model=paths_to_augs[i] + '/tvae_best.pkl', aug_type='TVAE')
        mu, sigma = run_comparisons(dataset, augmentator, K)
        tvae_results.append((mu, sigma))
        print((mu, sigma))
    df = pd.DataFrame({'CTGAN': ctgan_results, 'TVAE': tvae_results}, index=dataset_names)
    print(df)
    df.to_csv('augs_benchmark.csv')
