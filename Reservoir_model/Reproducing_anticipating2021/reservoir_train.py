# -*- coding: utf-8 -*- 
# @Time : 2021/11/28 13:43 
# @Author : lepold
# @File : reservoir_train.py


import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import echotorch.nn as etnn
import echotorch.utils
import echotorch.utils.matrix_generation as mg
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


# Customized Dataset
class ChaoticDataset(Dataset):
    """
    Coupled Chaotic map dataset
    """

    def __init__(self, root_dir, sample_len, n_sample):
        self.root_dir = root_dir
        self.sample_len = sample_len
        self.n_sample = n_sample

    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        data_path = os.path.join(self.root_dir, "train_data.npy")
        data = np.load(data_path)
        return data, data


if __name__ == '__main__':

    spectral_radius = 1.
    leaky_rate = 1.0
    input_dim = 2
    n_hidden = 100

    use_cuda = True
    use_cuda = torch.cuda.is_available() if use_cuda else False

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(BASE_DIR, "data")
    data = np.load(os.path.join(root_dir, "train_data.npy"))

    # coupled_chaotic_dataset = ChaoticDataset(root_dir, train_sample_length,
                                             # n_train_samples)  # train_sample_length and n_train samples is invalid here!
    # trainloader = DataLoader(coupled_chaotic_dataset, batch_size=1, shuffle=False, num_workers=2)

    w_generator = mg.matrix_factory.get_generator(
        "uniform",
        connectivity=0.2,
        spectral_radius=1e-5,
        apply_spectral_radius=True,
        scale=1.0,
        input_set=None,
        minimum_edges=0,
        min=-1.0,
        max=1.0
    )
    win_generator = mg.matrix_factory.get_generator(
        "uniform",
        connectivity=None,
        spectral_radius=0.99,
        apply_spectral_radius=False,
        scale=1.0,
        input_set=None,
        minimum_edges=0,
        min=-1.0,
        max=1.0
    )
    wbias_generator = mg.matrix_factory.get_generator(
        "uniform",
        connectivity=None,
        spectral_radius=0.99,
        apply_spectral_radius=False,
        scale=1.,
        input_set=None,
        minimum_edges=0,
        min=-1.0,
        max=1.0
    )

    esn = etnn.CLiESN(
        input_dim=input_dim,
        hidden_dim=n_hidden,
        output_dim=2,
        learning_algo='inv',
        leaky_rate=leaky_rate,
        w_generator=w_generator,
        win_generator=win_generator,
        wbias_generator=wbias_generator,
        washout=1000,
        memory=1,
        with_bias=False,
        ridge_param=1e-5,
    )
    if use_cuda:
        esn.cuda()

    # Train phase
    # Inputs and outputs
    inputs, targets = data, data
    targets = targets[:, :-1]
    inputs = torch.from_numpy(inputs[np.newaxis, :,:].astype(np.float32))
    targets = torch.from_numpy(targets[np.newaxis, :,:].astype(np.float32))

    # To variable
    inputs, targets = Variable(inputs), Variable(targets)
    if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

    # Accumulate xTx and xTy
    esn(inputs, targets)
    # end for

    # Finalize training
    esn.finalize()

    # output during train phase
    # vout = esn(inputs)
    vout = inputs
    # Test phase

    epsilon_ls = [0.2, 0.22, 0.24, 0.26]
    test_ls = []
    fig, ax = plt.subplots(2, 4, figsize=(10, 5), dpi=300)
    ax = ax.flatten()
    for idx, epsilon in enumerate(epsilon_ls):
        vvout = vout[:, :2000, :]
        tt = vvout.size(1)
        test_u = torch.stack([vvout[:,:,0], vvout[:,:,1], torch.ones((1, tt)) * epsilon], dim=2)
        test_u= Variable(test_u)
        if use_cuda: test_u = test_u.cuda()
        y_predicted = esn(test_u)
        y_predicted = y_predicted.cpu().numpy().squeeze()
        test_ls.append(y_predicted)
        ax[idx].scatter(vvout[0, -500:-1, 0], vvout[0, -499:, 0], s=0.5, marker=",", color="b")
        ax[idx].scatter(y_predicted[-500:-1, 0], y_predicted[-499:, 0], s=0.5, marker=",", color="r")
        ax[idx].set_xlabel(r'$x_{1}(n)$')
        ax[idx].set_ylabel(r'$x_{1}(n+1)$')
        ax[idx].set_xlim([0, 1])
        ax[idx].set_ylim([0, 1])
        ax[idx].text(0.4, 0.1, r'$\epsilon$=%.2f' % epsilon_ls[idx], )
        ax[idx].set_aspect("equal")

        ax[idx + 4].scatter(vvout[0, -500:, 0], vvout[0, -500:, 1], s=0.5, marker=",", color="b")
        ax[idx + 4].scatter(y_predicted[-500:, 0], y_predicted[-500:, 1], s=0.5, marker=",", color="r")
        ax[idx + 4].set_xlabel(r'$x_{1}$')
        ax[idx + 4].set_ylabel(r'$x_{2}$')
        ax[idx + 4].set_xlim([0, 1])
        ax[idx + 4].set_ylim([0, 1])
        ax[idx + 4].text(0.8, 0.1, r'$\epsilon$=%.2f' % epsilon_ls[idx], )
        ax[idx + 4].set_aspect("equal")

    test_ls = np.stack(test_ls, axis=0)
    plt.show()
    # np.save(os.path.join(root_dir, 'predicted_data.npy'), test_ls)