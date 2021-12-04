# Imports
import torch
import torch.sparse
from torch.autograd import Variable

import echotorch.utils
from echotorch.utils.visualisation import Observable
from .LiESNCell import LiESNCell


# Echo State Network layer
# with Feedback
# TODO: Test
class CLiESNCell(LiESNCell, Observable):
    """
    Echo State Network layer
    Basis cell for ESN
    """

    # Constructor
    def __init__(self, feedback_noise, epsilon_k, epsilon_bias, *args, **kwargs):
        """
        Constructor
        :param args: Arguments
        :param kwargs: Positional arguments
        """
        # Superclass
        super(CLiESNCell, self).__init__(*args, **kwargs)

        # Feedbacks matrix
        self._feedback_noise = feedback_noise
        self._epsilon_k = epsilon_k
        self._epsilon_bias = epsilon_bias
        self._w_fdb = None
        self._u_clue = None

    # end __init__

    ######################
    # PROPERTIES
    ######################

    ######################
    # PUBLIC
    ######################

    # Set feedbacks
    def set_feedbacks(self, w_fdb):
        """
        Set feedbacks
        :param w_fdb: Feedback matrix (reservoir x
        """
        self._w_fdb = w_fdb

    # end set_feedbacks

    ######################
    # PRIVATE
    ######################

    ######################
    # OVERRIDE
    ######################

    # Hook which gets executed before the update state equation for every timesteps.
    def _pre_step_update_hook(self, inputs, forward_i, sample_i, t):
        """
        Hook which gets executed before the update equation for every timesteps
        :param inputs: Input signal.
        :param forward_i: Index of forward call
        :param sample_i: Position of the sample in the batch.
        :param t: Timestep.
        """
        if self._w_fdb is not None:
            if callable(self._feedback_noise):
                return torch.mv(self._w_fdb, self.hidden) + self._feedback_noise(self._output_dim)
            else:
                return torch.mv(self._w_fdb, self.hidden)
        else:
            return inputs
        # end if

    # end _pre_step_update_hook

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
                # define _u_cule in time t in batch b
                self._u_clue = u[b, t, -1]

                # Current input
                ut = u[b, t] * self._input_scaling

                # Pre-hook
                ut = self._pre_step_update_hook(ut, self._forward_calls, b, t)

                # Compute input layer
                u_win = self._input_layer(ut)

                # Apply W to x
                x_w = self._recurrent_layer(self.hidden)

                # Add everything
                x = self._reservoir_layer(u_win, x_w)

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

    def _reservoir_layer(self, u_win, x_w):
        """
        Compute reservoir layer
        :param u_win: Processed inputs
        :param x_w: Processed states
        :return: States before non-linearity
        """
        if self._noise_generator is None:
            return u_win + x_w + self.w_bias * (self._u_clue + self._epsilon_bias) * self._epsilon_k
        else:
            return u_win + x_w + self.w_bias * (self._u_clue + self._epsilon_bias) * self._epsilon_k + self._noise_generator(
                self._output_dim)
        # end if
    # end _reservoir_layer

    def extra_repr(self):
        """
        Extra-information
        """
        s = super(CLiESNCell, self).extra_repr()
        s += ', leaky-rate={_leaky_rate}'
        return s.format(**self.__dict__)
    # end extra_repr
