# -*- coding: utf-8 -*- 
# @Time : 2021/11/26 14:42 
# @Author : lepold
# @File : coupled_chaotic_maps.py
import os

import numpy as np
import unittest
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool


# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)


class Coupled_logistic_maps:
    def __init__(self, epsilon, num=1):
        # initial state
        self.x1 = np.random.rand(num)
        self.x2 = np.random.rand(num)
        self.epsilon = epsilon
        self.num = num

    @staticmethod
    def customized_func(x):
        return 4 * x * (1 - x)

    def run(self, time=1):
        x1list = np.empty((time, self.num), dtype=np.float32)
        x2list = np.empty((time, self.num), dtype=np.float32)
        for i in range(time):
            temp1 = self.customized_func(self.x2) - self.customized_func(self.x1)
            temp2 = self.customized_func(self.x1) - self.customized_func(self.x2)

            self.x1 = self.customized_func(self.x1) + self.epsilon * temp1
            self.x2 = self.customized_func(self.x2) + self.epsilon * temp2
            x1list[i] = self.x1
            x2list[i] = self.x2
        return x1list.squeeze(), x2list.squeeze()


class TestLogisticMaps(unittest.TestCase):

    @staticmethod
    def initialize():
        np.random.seed()

    def _test_system_behavior(self):
        epsilon_list = [0.20, 0.22, 0.24, 0.26]
        systems = [Coupled_logistic_maps(epsilon, 1) for epsilon in epsilon_list]

        def _run(i):
            return systems[i].run(time=3000)

        with ThreadPool(4, initializer=self.initialize, ) as p:
            data_list = p.map(_run, range(4))

        fig, ax = plt.subplots(2, 4, figsize=(10, 5), dpi=300)
        ax = ax.flatten()
        for i in range(4):
            ax[i].scatter(data_list[i][0][-1000:-1], data_list[i][0][-999:], label=r'$\epsilon$=%.2f' % epsilon_list[i],
                          s=0.5, marker=",")
            ax[i].legend(loc="best", frameon=False)
            ax[i].set_xlabel(r'$x_{1}(n)$')
            ax[i].set_ylabel(r'$x_{1}(n+1)$')
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim([0, 1])
            ax[i].set_aspect("equal")
        for i in np.arange(4, 8):
            ax[i].scatter(data_list[i - 4][0][-1000:], data_list[i - 4][1][-1000:],
                          label=r'$\epsilon$=%.2f' % epsilon_list[i - 4], s=0.5, marker=",")
            ax[i].legend(loc="best", frameon=False)
            ax[i].set_xlabel(r'$x_{1}$')
            ax[i].set_ylabel(r'$x_{2}$')
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim([0, 1])
            ax[i].set_aspect("equal")

        fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
        fig.savefig("./system_scatter.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax1 = fig.add_subplot(1, 1, 1, frameon=False)
        ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax1.set_xlabel('Iteration')

        for i in range(4):
            ax = fig.add_subplot(4, 1, i + 1)
            ax.plot(data_list[i][0][-100:], '*-', label='x1', c='b', linewidth=1)
            ax.plot(data_list[i][1][-100:], '*-', label='x2', c="r", linewidth=1)
            ax.set_xlim([0, 110])
            ax.set_ylim([0, 1])
            plt.yticks([])
            plt.xticks([0, 50, 100], [0, 50, 100])
            # ax.spines['right'].set_color('none')
            # ax.spines['top'].set_color('none')
            # ax.spines['left'].set_color('none')
            # ax.spines['right'].set_color('none')
            # ax.legend(loc="best")

            ax.text(100, 0.7, r'$\epsilon$=%.2f' % epsilon_list[i], )
            # ax.tick_params(axis='x', labelcolor='cornflowerblue')
        fig.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.1)
        fig.savefig("./system_evolution.png")
        plt.close(fig)

    def test_generate_train_test_data(self):
        epsilon_list = [0.20, 0.22, 0.24, 0.26]
        ll = len(epsilon_list)
        systems = [Coupled_logistic_maps(epsilon, 1) for epsilon in epsilon_list]

        def _run(i):
            return systems[i].run(time=3000)

        with ThreadPool(4, initializer=self.initialize, ) as p:
            data_list = p.map(_run, range(ll))

        train_data = np.stack([np.stack(data_list[i], axis=0) for i in range(3)], axis=0)
        train_data = train_data[:, :, 1000:].transpose(1, 0, 2)
        train_data = train_data.reshape((2, -1))
        print(train_data[0].shape)
        train_para = np.concatenate([np.ones(2000) * epsilon_list[i] for i in range(3)])
        train_data = np.stack([train_data[0], train_data[1],  train_para], axis=0)
        os.makedirs("./data", exist_ok=True)
        np.save("./data/train_data.npy", train_data)

    def _test_reservoir_predict(self):
        # TODO(2021.11.28): It may be supposed to implement in an Independent file.
        pass
