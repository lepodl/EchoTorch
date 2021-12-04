# -*- coding: utf-8 -*- 
# @Time : 2021/12/3 13:58 
# @Author : lepold
# @File : CLiESN.py

import torch
from echotorch.nn.linear.RRCell import RRCell
from .CLiESNCell import CLiESNCell
from ..Node import Node
from echotorch.nn.reservoir.ESN import ESN


# Li-ESN with Feedbacks
class CLiESN(ESN):
    """
    Li-ESN with Feedbacks, and input with clues
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, leaky_rate, w_generator, win_generator, wbias_generator,
                 bias_scaling=1.0, input_scaling=1.0, nonlin_func=torch.tanh, learning_algo='inv',
                 ridge_param=0.0, with_bias=True, softmax_output=False, washout=0, memory=0, debug=Node.NO_DEBUG, test_case=None,
                 epsilon_k=1.0, epsilon_bias=0.,
                 dtype=torch.float32):
        """
        Constructor
        :param input_dim: Input feature space dimension
        :param hidden_dim: Reservoir hidden space dimension
        :param output_dim: Output space dimension
        :param leaky_rate: Leaky-rate
        :param spectral_radius: Spectral radius
        :param bias_scaling: Bias scaling
        :param input_scaling: Input scaling
        :param w_generator: Internal weight matrix generator
        :param win_generator: Input-reservoir weight matrix generator
        :param wbias_generator: Bias weight matrix generator
        :param nonlin_func: Non-linear function
        :param learning_algo: Learning algorithm (inv, pinv)
        :param ridge_param: Regularisation parameter
        :param with_bias: Add a bias to output ?
        :param softmax_output: Add a softmax layer at the outputs ?
        :param washout: Length of the washout period ?
        :param debug: Debug mode
        :param test_case: Test case to call for test
        :param dtype: Data type
        """
        super(CLiESN, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            w_generator=w_generator,
            win_generator=win_generator,
            wbias_generator=wbias_generator,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            learning_algo=learning_algo,
            ridge_param=ridge_param,
            softmax_output=softmax_output,
            washout=washout,
            with_bias=with_bias,
            create_rnn=False,
            create_output=False,
            debug=debug,
            test_case=test_case,
            dtype=dtype,
        )

        # Generate matrices
        w, w_in, w_bias = self._generate_matrices(w_generator, win_generator, wbias_generator)

        # Recurrent layer
        self._esn_cell = CLiESNCell(
            leaky_rate=leaky_rate,
            input_dim=input_dim,
            output_dim=hidden_dim,
            input_scaling=input_scaling,
            w=w,
            w_in=w_in,
            w_bias=w_bias,
            nonlin_func=nonlin_func,
            washout=washout,
            memory=memory,
            debug=debug,
            test_case=test_case,
            dtype=torch.float32,
            epsilon_k=epsilon_k,
            epsilon_bias=epsilon_bias,
            feedback_noise=None
        )

        self._output = RRCell(
            input_dim=hidden_dim,
            output_dim=output_dim,
            ridge_param=ridge_param,
            with_bias=with_bias,
            learning_algo=learning_algo,
            softmax_output=softmax_output,
            normalize_output=False,
            debug=debug,
            test_case=test_case,
            dtype=dtype,
            handle_output=True
        )
        # Trainable elements
        self.add_trainable(self._output)   # not needed is ok.
    # end __init__

    ###################
    # PROPERTIES
    ###################

    ###################
    # PUBLIC
    ###################

    # Finish training
    def finalize(self):
        """
        Finish training
        """
        # Train output
        self._output.finalize()

        # Set feedback matrix
        self._esn_cell.set_feedbacks(self._output.w_out)

        # In eval mode
        self.train(False)
    # end finalize

    ###################
    # PRIVATE
    ###################

# end ESNCell
