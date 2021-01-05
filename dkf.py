"""
Inspired by the Deep Markov Model from Pyro: https://pyro.ai/examples/dmm.html
"""

import argparse
from os.path import exists

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from tqdm import tqdm


class Emitter(nn.Module):
    '''
    Parameterizes the normal observation likelihood `p(y_t | z_t)`
    '''
    
    def __init__(self, width, height, input_channels, z_dim, emission_channels, kernel_size):
        super().__init__()
        kernel_size = 4
        padding = int((kernel_size - 1) / 2)
        n_layers = 2
        stride = 2
        self.feature_to_cnn_dim = min(width, height) // 2 ** n_layers
        self.feature_to_cnn_shape = (emission_channels[0], self.feature_to_cnn_dim, self.feature_to_cnn_dim)

        # Return to original dimension using ConvCNNs
        self.lin_z_to_hidden = nn.Linear(
            z_dim, np.prod(self.feature_to_cnn_shape))
        self.lin_hidden_to_hidden = nn.ConvTranspose2d(
            emission_channels[0], emission_channels[1], kernel_size, stride, padding, bias=True)
        self.lin_hidden_to_input_loc = nn.ConvTranspose2d(
            emission_channels[1], input_channels, kernel_size, stride, padding, bias=True)
        self.lin_hidden_to_input_scale = nn.ConvTranspose2d(
            emission_channels[1], input_channels, kernel_size, stride, padding, bias=True)

        # Non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, z_t):
        """
        Given z_t, calculate mean and variance to parameterize the normal distribution `p(y_t | z_t)`
        """
        
        batch_size = z_t.shape[0]
        h1 = self.relu(self.lin_z_to_hidden(z_t)).view(batch_size, *self.feature_to_cnn_shape)
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        loc = self.tanh(self.lin_hidden_to_input_loc(h2))
        scale = self.softplus(self.lin_hidden_to_input_scale(h2)).clamp(min=1e-4)
        return loc, scale


class GatedTransition(nn.Module):
    """
    Parameterizes the normal latent transition probability `p(z_t | z_{t-1})`
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()

        # Initialize gate and proposed mean for split linearity 
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        # Non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given z_{t-1}, calculate mean and variance to parameterize the
        (diagonal) normal distribution `p(z_t | z_{t-1})`
        """
        # Compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))

        # Compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)

        # Assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean

        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean))).clamp(min=1e-4)

        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, y_{t:T})`, which is the basic building block
    of the inference model. The dependence on `y_{t:T}` is
    through the hidden state of the RNN
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        # Non-linearities
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given z_{t-1} and the hidden RNN state h(y_{t:T}), calculate mean and variance tp
        parameterize the (diagonal) normal distribution `q(z_t | z_{t-1}, y_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)

        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)

        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))

        # return loc, scale which can be fed into Normal
        return loc, scale


class Flattener(nn.Module):
    """
    Flatten the input data
    """
    
    def __init__(self, width, height, input_channels, rnn_dim, flatten_channels, kernel_size):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        n_layers = 2
        stride = 2
        self.feature_width = width // 2 ** len(flatten_channels)
        self.feature_height = height // 2 ** len(flatten_channels)
        self.feature_dim = flatten_channels[-1] * self.feature_width * self.feature_height

        # Two-layered convolution with a fully connected layer at last 
        self.conv_y_to_hidden = nn.Conv2d(input_channels, flatten_channels[0], 
                                          kernel_size, stride, padding, bias=True)
        self.conv_hidden_to_hidden = nn.Conv2d(flatten_channels[0], flatten_channels[1], 
                                               kernel_size, stride, padding, bias=True)
        self.lin_hidden_to_rnn = nn.Linear(self.feature_dim, rnn_dim)

        # Non-linearities 
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, z_t):
        """
        Return the flattened input to RNN 
        """
        batch_size = z_t.shape[0]
        h1 = self.relu(self.conv_y_to_hidden(z_t))
        h2 = self.relu(self.conv_hidden_to_hidden(h1)).view(batch_size, -1)
        rnn_input = self.tanh(self.lin_hidden_to_rnn(h2))

        return rnn_input


class DKF(nn.Module):
    """
    Generative model and inference model for the Deep Kalman Filter 
    """

    def __init__(self, input_channels=1, z_channels=16, emission_channels=[32, 16],
                 transition_channels=32, flatten_channels=[16, 32], rnn_input_dim=32, 
                 rnn_channels=32, kernel_size=3, height=100, width=100, 
                 num_layers=1, rnn_dropout_rate=0.0, num_iafs=0, iaf_dim=50, use_cuda=False):
        super().__init__()
        self.input_channels = input_channels
        self.rnn_input_dim = rnn_input_dim
        self.height = height
        self.width = width

        # Call functions 
        self.emitter = Emitter(width, height, input_channels, z_channels, emission_channels, kernel_size)
        self.trans = GatedTransition(z_channels, transition_channels)
        self.combiner = Combiner(z_channels, rnn_channels)
        self.flatten = Flattener(width, height, input_channels, rnn_input_dim, flatten_channels, kernel_size)

        # Instantiate RNN
        if use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Setup RNN
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_channels,
                           batch_first=True, bidirectional=False, num_layers=num_layers,
                           dropout=rnn_dropout_rate)
        
        # Normalizing flows, Inverse Autoregressive Flows 
        self.iafs = [affine_autoregressive(z_channels, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        
        # Initiate parameters z_0 and z_q_0 to build the probability
        # distributions p(z_1) and q(z_1)
        self.z_0 = nn.Parameter(torch.zeros(z_channels))
        self.z_q_0 = nn.Parameter(torch.zeros(z_channels))

        # Initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_channels))

        # If we are on GPU
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, mini_batch_reversed, annealing_factor=1.0):
        """
        The generative model p(y_{1:T} | z_{1:T}) p(z_{1:T})
        """
        
        # Number of time steps
        T_max = mini_batch.size(1)

        # Register PyTorch modules with Pyro
        pyro.module("dkf", self)

        # Set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # We enclose all the sample statements in the model in a plate 
        # for conditional independence
        with pyro.plate("z_test", len(mini_batch), dim=-3):
            locs = torch.zeros((mini_batch.size(0), T_max, 1, self.height, self.width))
            scales = torch.zeros((mini_batch.size(0), T_max, 1, self.height, self.width))
            
            # Sample the latents z and observations y
            for t in range(1, T_max + 1):
                # Sample z_t ~ p(z_t | z_{t-1})

                # Mean and variance for normal distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # Sample from dist.Normal(z_loc, z_scale)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample("z_%d" % t, dist.Normal(z_loc, z_scale).to_event(1))

                # Mean and variance for normal distribution p(y_t | z_t)
                emission_loc_t, emission_scale_t = self.emitter(z_t)
                locs[:, t - 1, :, :, :] = emission_loc_t
                scales[:, t - 1, :, :, :] = emission_scale_t
                
                # Sample from dist.Normal(emission_loc_t, emission_scale_t)
                pyro.sample("obs_y_%d" % t,
                            dist.Normal(emission_loc_t, emission_scale_t).to_event(1),
                            obs=mini_batch[:, t - 1, :, :, :])

                # Update time step 
                z_prev = z_t
                
    def guide(self, mini_batch, mini_batch_reversed, annealing_factor=1.0):
        """
        The inference model q(z_{1:T} | y_{1:T})
        """
        
        # Number of time steps through mini-batch
        T_max = mini_batch.size(1)

        # Register all PyTorch modules with Pyro
        pyro.module("dkf", self)
        
        # Contiguous hidden state 
        h_0 = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        
        # Flatten and reverse input y 
        batch_size = mini_batch_reversed.shape[0]
        seq_len = mini_batch_reversed.shape[1]
        flat_mini_batch_reversed = torch.zeros(
            batch_size, seq_len, self.rnn_input_dim).to(self.device)
        for t in range(seq_len):
            flat_mini_batch_reversed[:, t, :] = self.flatten(mini_batch_reversed[:, t, :, :, :])

        # Feed y through RNN;
        rnn_output, _ = self.rnn(flat_mini_batch_reversed, h_0)

        # Backwards to take future observations into account 
        rnn_output = reversed_input(rnn_output)

        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # We enclose all the sample statements in the model in a plate 
        # for conditional independence
        with pyro.plate("z_test", len(mini_batch)):
            # Sample the latents z 
            for t in range(1, T_max + 1):
                # Mean and variance for the distribution q(z_t | z_{t-1}, y_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # If we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else: 
                    z_dist = dist.Normal(z_loc, z_scale)
                    
                # Sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        z_t = pyro.sample("z_%d" % t, z_dist)
                    else:
                        # When no normalizing flow is used, ".to_event(1)" 
                        # indicates latent dimensions are independent
                        z_t = pyro.sample("z_%d" % t, z_dist.to_event(1))

                # Update time step 
                z_prev = z_t

        return z_t


def reversed_input(rnn_output):
    """
    Reverse order for RNN-input
    """

    T = rnn_output.size(1)
    time_slices = torch.arange(T - 1, -1, -1, device=rnn_output.device)

    return rnn_output.index_select(1, time_slices)