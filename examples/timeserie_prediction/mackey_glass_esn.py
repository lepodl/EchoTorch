# -*- coding: utf-8 -*-
#
# File : examples/SwitchAttractor/switch_attractor_esn
# Description : Attractor switching task with ESN.
# Date : 26th of January, 2018
#
# This file is part of EchoTorch.  EchoTorch is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti <nils.schaetti@unine.ch>


# Imports
import numpy as np
import torch
from echotorch.datasets.MackeyGlassDataset import MackeyGlassDataset
import echotorch.nn as etnn
import echotorch.utils
import echotorch.utils.matrix_generation as mg
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

# Dataset params
train_sample_length = 5000
test_sample_length = 2000
n_train_samples = 1
n_test_samples = 1
spectral_radius = 0.9
leaky_rate = 1.0
input_dim = 1
n_hidden = 100
washout = 1000

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

# Mackey glass dataset
mackey_glass_train_dataset = MackeyGlassDataset(train_sample_length, n_train_samples, tau=30)
mackey_glass_test_dataset = MackeyGlassDataset(test_sample_length, n_test_samples, tau=30)

# Data loader
trainloader = DataLoader(mackey_glass_train_dataset, batch_size=1, shuffle=False, num_workers=2)
testloader = DataLoader(mackey_glass_test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Random weight generator
w_generator = mg.matrix_factory.get_generator(
    "normal",
    mean=0.0,
    std=1.0
)

# ESN cell
esn = etnn.LiESN(
    input_dim=input_dim,
    hidden_dim=n_hidden,
    output_dim=1,
    spectral_radius=spectral_radius,
    learning_algo='inv',
    leaky_rate=leaky_rate,
    w_generator=w_generator,
    win_generator=w_generator,
    wbias_generator=w_generator,
    washout=washout,
    with_bias=True,
)
if use_cuda:
    esn.cuda()
# end if

# For each batch
if __name__ == '__main__':
    for data in trainloader:
        # Inputs and outputs
        inputs, targets = data

        # To variable
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # Accumulate xTx and xTy
        esn(inputs, targets)
    # end for

    # Finalize training
    esn.finalize()

    # Train MSE
    fig = plt.figure(figsize=(10, 7), dpi=300)
    axes1 = fig.add_subplot(2, 1, 1)
    dataiter = iter(trainloader)
    train_u, train_y = dataiter.next()
    train_u, train_y = Variable(train_u), Variable(train_y)
    if use_cuda: train_u, train_y = train_u.cuda(), train_y.cuda()
    y_predicted = esn(train_u)
    time = y_predicted.size()[1]
    time = np.arange(time)
    axes1.plot(time, train_y.data.squeeze()[washout:], color="r")
    axes1.plot(time, y_predicted.data.squeeze(), color="b")
    axes1.set_title("train phase")

    print(u"Train MSE: {}".format(echotorch.utils.mse(y_predicted.data, train_y[:,washout:, :].data)))
    print(u"Train NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, train_y[:,washout:, :].data)))
    print(u"")


    # Test MSE
    dataiter = iter(testloader)
    test_u, test_y = dataiter.next()
    test_u, test_y = Variable(test_u), Variable(test_y)
    if use_cuda: test_u, test_y = test_u.cuda(), test_y.cuda()
    y_predicted = esn(test_u)
    print(u"Test MSE: {}".format(echotorch.utils.mse(y_predicted.data, test_y[:, washout:, :].data)))
    print(u"Test NRMSE: {}".format(echotorch.utils.nrmse(y_predicted.data, test_y[:, washout:, :].data)))
    print(u"")
    time = y_predicted.size()[1]
    time = np.arange(time)
    axes2 = fig.add_subplot(2, 1, 2)
    axes2.plot(time, test_y.data.squeeze()[washout:], color="r")
    axes2.plot(time, y_predicted.data.squeeze(), color="b")
    axes2.set_title("test phase")
    plt.show()
