# -*- coding: utf-8 -*-
#
# File : echotorch/nn/LiESNCell.py
# Description : An Leaky-Integrated Echo State Network layer.
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

"""
Created on 26 January 2018
@author: Nils Schaetti
"""

import torch
import torch.sparse
import torch.nn as nn
from torch.autograd import Variable
from echotorch.nn.reservoir.ESNCell import ESNCell


# Leak-Integrated Echo State Network layer
class LiESNCell(ESNCell):
    """
    Leaky-Integrated Echo State Network layer
    """

    # Constructor
    def __init__(self, leaky_rate=1.0, *args, **kwargs):
        """
        Constructor
        :param leaky_rate: Reservoir's leaky rate (default 1.0, normal ESN)
        """
        super(LiESNCell, self).__init__(*args, **kwargs)

        # Param
        self._leaky_rate = leaky_rate

    # end __init__

    #####################
    # PUBLIC
    #####################

    #####################
    # OVERLOAD
    #####################

    # Compute post nonlinearity hook
    def _post_nonlinearity(self, x):
        """
        Compute post nonlinearity hook
        :param x: Reservoir state at time t
        :return: Reservoir state
        """
        return self.hidden.mul(1.0 - self._leaky_rate) + x.view(self.output_dim).mul(self._leaky_rate)

    # end _post_nonlinearity

    def _reservoir_layer(self, u_win, x_w, u_clue, epsilon_k=1, epsilon_bias=0):
        """
        Compute reservoir layer
        :param u_win: Processed inputs
        :param x_w: Processed states
        :param u_clue: Processed clues
        :return: States before non-linearity
        """
        if self._noise_generator is None:
            return u_win + x_w + self.w_bias * (u_clue +  epsilon_bias) * epsilon_k
        else:
            return u_win + x_w + self.w_bias * (u_clue +  epsilon_bias) * epsilon_k + self._noise_generator(self._output_dim)
        # end if
    # end _reservoir_layer

    #############################################################################################################
    def forward(self, u, reset_state=True):
        """
        Forward pass function
        :param u: Input signal
        :param reset_state: Reset state at each batch ?
        :return: Resulting hidden states
        """
        # Time length
        time_length = int(u.size()[1])

        # Number of batches
        n_batches = int(u.size()[0])

        # Outputs
        outputs = Variable(torch.zeros(n_batches, time_length, self.output_dim, dtype=self.dtype))
        outputs = outputs.cuda() if self.hidden.is_cuda else outputs

        # divide u into two parts along the last dimension: data and clues
        u_clue = u[:, :, -1]
        u = u[:, :, :2]

        # For each sample
        for b in range(n_batches):
            # Reset hidden layer
            if reset_state:
                self.reset_hidden()
            # end if

            # Pre-update hook
            u[b, :] = self._pre_update_hook(u[b, :], self._forward_calls, b)

            # Observe inputs
            self.observation_point('U', u[b, :])

            # For each steps
            for t in range(time_length):
                # Current input
                ut = u[b, t] * self._input_scaling

                # Pre-hook
                ut = self._pre_step_update_hook(ut, self._forward_calls, b, t)

                # Compute input layer
                u_win = self._input_layer(ut)

                # Apply W to x
                x_w = self._recurrent_layer(self.hidden)

                # Add everything
                x = self._reservoir_layer(u_win, x_w, u_clue[b, t])

                # Apply activation function
                x = self.nonlin_func(x)

                # Post nonlinearity
                x = self._post_nonlinearity(x)

                # Post-hook
                x = self._post_step_update_hook(x.view(self.output_dim), ut, self._forward_calls, b, t)

                # Neural filter
                for neural_filter_handler in self._neural_filter_handlers:
                    x = neural_filter_handler(x, ut, self._forward_calls, b, t, t < self._washout)
                # end if

                # New last state
                self.hidden.data = x.data

                # Add to outputs
                outputs[b, t] = self.hidden
            # end for

            # Post-update hook
            outputs[b, :] = self._post_update_hook(outputs[b, :], u[b, :], self._forward_calls, b)

            # Post states update handlers
            for handler in self._post_states_update_handlers:
                handler(outputs[b, self._washout:], u[b, self._washout:], self._forward_calls, b)
            # end for

            # Observe states
            self.observation_point('X', outputs[b, self._washout:])
        # end for

        # Count calls to forward
        self._forward_calls += 1

        return outputs[:, self._washout:]

    ###################################################################################################################

    # Extra-information
    def extra_repr(self):
        """
        Extra-information
        """
        s = super(LiESNCell, self).extra_repr()
        s += ', leaky-rate={_leaky_rate}'
        return s.format(**self.__dict__)
    # end extra_repr

# end LiESNCell
