################################################################################################################################################################################################
#  Feature selection - syn data generation
################################################################################################################################################################################################
"""
Based on the work of 'Jinsung Yoon': https://github.com/jsyoon0823/INVASE/blob/master/data_generation.py
Written by Jinsung Yoon
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "IINVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu
---------------------------------------------------
Generating Synthetic Data for Synthetic Examples
There are 6 Synthetic Datasets
X ~ N(0,I) where d = 100
Y = 1/(1+logit)
- Syn1: logit = exp(X1 * X2)
- Syn2: logit = exp(X3^2 + X4^2 + X5^2 + X6^2 -4)
- Syn3: logit = -10 sin(2 * X7) + 2|X8| + X9 + exp(-X10) - 2.4
- Syn4: If X11 < 0, Syn1, X11 >= Syn2
- Syn5: If X11 < 0, Syn1, X11 >= Syn3
- Syn6: If X11 < 0, Syn2, X11 >= Syn3
"""
import errno
import os

import numpy as np
import pandas as pd

N_SAMPLES = 10000
N_DIM = 10
SEED = 1348
np.random.seed(SEED)


# %% Basic Label Generation (Syn1, Syn2, Syn3)
def Basic_Label_Generation(X, data_type):
    # number of samples
    n = len(X[:, 0])

    # Logit computation
    # 1. Syn1
    if data_type == 'Syn1':
        logit = np.exp(X[:, 0] * X[:, 1])

    # 2. Syn2
    elif data_type == 'Syn2':
        logit = np.exp(np.sum(X[:, 2:6] ** 2, axis=1) - 4.0)

    # 3. Syn3
    elif data_type == 'Syn3':
        logit = np.exp(-10 * np.sin(0.2 * X[:, 6]) + abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4)

    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape((1 / (1 + logit)), [n, 1])
    prob_0 = np.reshape((logit / (1 + logit)), [n, 1])

    # Probability output
    prob_y = np.concatenate((prob_0, prob_1), axis=1)

    # Sampling from the probability
    y = np.zeros([n, 2])
    y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n, ])
    y[:, 1] = 1 - y[:, 0]

    return y[:, 1]


# %% Complex Label Generation (Syn4, Syn5, Syn6)
def Complex_Label_Generation(X, data_type):
    # number of samples
    n = len(X[:, 0])

    # Logit generation
    # 1. Syn4
    if data_type == 'Syn4':
        logit1 = np.exp(X[:, 0] * X[:, 1])
        logit2 = np.exp(np.sum(X[:, 2:6] ** 2, axis=1) - 4.0)

    # 2. Syn5
    elif data_type == 'Syn5':
        logit1 = np.exp(X[:, 0] * X[:, 1])
        logit2 = np.exp(-10 * np.sin(0.2 * X[:, 6]) + abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4)

    # 3. Syn6
    elif data_type == 'Syn6':
        logit1 = np.exp(np.sum(X[:, 2:6] ** 2, axis=1) - 4.0)
        logit2 = np.exp(-10 * np.sin(0.2 * X[:, 6]) + abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4)

    # Based on X[:,10], combine two logits
    idx1 = (X[:, 10] < 0) * 1
    idx2 = (X[:, 10] >= 0) * 1

    logit = logit1 * idx1 + logit2 * idx2

    # P(Y=1|X) & P(Y=0|X)
    prob_1 = np.reshape((1 / (1 + logit)), [n, 1])
    prob_0 = np.reshape((logit / (1 + logit)), [n, 1])

    # Probability output
    prob_y = np.concatenate((prob_0, prob_1), axis=1)

    # Sampling from the probability
    y = np.zeros([n, 2])
    y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n, ])
    y[:, 1] = 1 - y[:, 0]

    return y[:, 1]


# %% Ground truth Variable Importance
def Ground_Truth_Mask_Generation(n_features, data_type):
    # mask initialization
    m = np.zeros(n_features)

    # For each data_type
    # Simple
    if data_type in ['Syn1', 'Syn2', 'Syn3']:
        if data_type == 'Syn1':
            m[:2] = 1
        elif data_type == 'Syn2':
            m[2:6] = 1
        elif data_type == 'Syn3':
            m[6:10] = 1

    # Complex
    if data_type in ['Syn4', 'Syn5', 'Syn6']:
        if data_type == 'Syn4':
            m[:2] = 1
            m[2:6] = 1
        elif data_type == 'Syn5':
            m[:2] = 1
            m[6:10] = 1
        elif data_type == 'Syn6':
            m[2:6] = 1
            m[6:10] = 1
        m[10] = 1
    return m


def create_dir(path):
    if os.path.exists(path):
        return

    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# %% Generate X and Y
def generate_data(n=10000, d=11, data_type='Syn1', seed=1, output_dir='.'):
    """
    :param output_dir:
    :param n: Number of samples
    :param d: input dimension
    :param data_type: the name of the syn dataset
    :param seed: random seed for numpy
    """

    np.random.seed(seed)

    # X generation
    X = np.random.randn(n, d)

    # Y generation
    if data_type in ['Syn1', 'Syn2', 'Syn3']:
        Y = Basic_Label_Generation(X, data_type)

    elif data_type in ['Syn4', 'Syn5', 'Syn6']:
        Y = Complex_Label_Generation(X, data_type)

    data = np.concatenate([X, np.expand_dims(Y, axis=1)], axis=1)
    output_dir = os.path.join(output_dir, '{}_{}'.format(data_type, str(d)))
    create_dir(output_dir)
    output_path = os.path.join(output_dir, 'data.csv')
    pd.DataFrame(data=data).to_csv(output_path, index=False)


if __name__ == "__main__":
    datasets = ['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']
    for set_type in datasets:
        generate_data(n=N_SAMPLES,
                      data_type=set_type,
                      seed=SEED,
                      output_dir='datasets/synthetic/{}'.format(set_type))
