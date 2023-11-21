
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh

# Class for storing LSTM outputs in LSTMWithStates()
class LSTMStates(object):

    """class for storing the output states for a single layer in the
    LSTMWithStates() module

    """

    def __init__(self, layer):

        self.layer = layer
        self.step = []

        self.h = []   # hidden state
        self.c = []   # cell state
        self.i = []   # input gate
        self.f = []   # forget gate
        self.o = []   # output gate

    def __repr__(self):

        """print an informative object representation in the console with layer
        and the dimensions stored arrays.
        """

        return "LSTMOutput(L{}, ({}, {}))".format(self.layer+1,
                                                  len(self.h),
                                                  self.h[0].shape)

    def __len__(self):

        return len(self.hidden)

    def to_numpy_arrays(self):

        """helper function to convert attributes from lists
        to numpy arrays
        """

        self.h = np.asarray(self.h)
        self.c = np.asarray(self.c)
        self.i = np.asarray(self.i)
        self.f = np.asarray(self.f)
        self.o = np.asarray(self.o)

        return self


# subclass the base RNNModel class
class LSTMWithStates(nn.LSTM):

    """custom class that subclasses the nn.LSTM pytorch module and implements own
    .forward() method in order to output hidden activations and gating
    variables.

    Parameters:
    ----------

    ninp : int
        dimension of input encoding embeddings
    nhid : int
        the dimension of hidden layers
    nlayers: int,
        the number of LSTM hidden layers
    dropout: float,
        dropout parameter
    store_states: bool (default = True)
        whether or not to store states

    Attributes:
    ----------
    store_states : bool
        stores value set by the `store_states` input argument
    states : list of LSTMStates()
        for each layer it stores the outputus of the layer in the attributes of
        LSTMStates() class

    """

    def __init__(self, ninp, nhid, nlayers, dropout=0.5,
                       store_states=True):

        super(LSTMWithStates, self).__init__(ninp, nhid, nlayers, dropout=0.5)


        self.store_states = store_states
        self.states = None

    def init_state_logger(self):

        self.states = [LSTMStates(layer=i) for i in range(self.num_layers)]


    def forward(self, inputs, hidden):

        """override the RNNModel.forward by computing and returning
        intermediate states

        Parameters:
        ----------

        inputs : tensor (seq_len, batch, input_size),
            an array containing token indices

        Returns:
        -------
        logits

        """

        # unpack hidden tuple
        # these are states from past time step
        # each variable holds states for two layers (layers are in the first dim)
        h_past, c_past = hidden

        # these store hidden states for next time step
        h_next = []
        c_next = []

        # loop over hidden (LSTM) layers
        for k in range(self.num_layers):

            # get weights for the current LSTM layer
            w_ih, w_hh, b_ih, b_hh = self.all_weights[k]

            # compute forward pass for this layer
            h_t, c_t, gates = self.lstm_step(x_t=inputs[0, ...],
                                             h_past=h_past[k, ...],
                                             c_past=c_past[k, ...],
                                             w_ih=w_ih,
                                             w_hh=w_hh,
                                             b_ih=b_ih,
                                             b_hh=b_hh)

            # append state variables to lists if specified
            if self.store_states:

                self.states[k].layer = k
                self.states[k].h.append(h_t.detach().numpy())
                self.states[k].c.append(c_t.detach().numpy())

                self.states[k].i.append(gates["i"].detach().numpy())
                self.states[k].f.append(gates["f"].detach().numpy())
                self.states[k].o.append(gates["o"].detach().numpy())


            # update states for next time step
            inputs = h_t # add additional dimension (N, *, n_out)
            h_next.append(h_t)
            c_next.append(c_t)

        # add dimension that will fit the .decoder module
        inputs = inputs.unsqueeze(0)

        # concatenate output hidden across layer dimension (make sure it has batch dim)
        # h_next.shape = (n_layer, batch_size, hidden_size)
        h_next = torch.cat(h_next, 0).view(self.num_layers, *h_next[0].size())
        c_next = torch.cat(c_next, 0).view(self.num_layers, *c_next[0].size())

        return inputs, (h_next, c_next)


    def lstm_step(self, x_t, h_past, c_past, w_ih, w_hh, b_ih, b_hh):

        """lstm layer forward pass, effectively similar to what nn.LSTMCell
        does in pytorch.

        Parameters:
        ----------

        x_t : tensor (batch, input_size)
            input vector (observation)
        h_t : tensor (hidden_dim,)
            hidden state vector
        c_t : tensor (hidden_dim,)
            cell state vector
        w_ih : tensor (4*hidden_dim, hidden_dim)
            input weight matrix
        w_hh : tensor (4*hidden_dim, hidden_dim)
            recurrent weight matrix
        b_ih : tensor (4*hidden_dim,)
            input bias term
        b_hh : tensor (4*hidden_dim,)
            recurrent bias term

        Returns
        -------

        h : tensor
            new hidden state vector
        c : tensor
            new cell state vector
        gates : tuple of (batch, hidden_dim) tensors
            a tuple containting input, forget, and output gate vectors

        """

        # compute gate values (the respective weights are stacked, so we compute
        # all gates in a single go)
        # gates.shape = (1, batch, dim*4)
        gates = F.linear(x_t, w_ih, b_ih) + F.linear(h_past, w_hh, b_hh)

        # unpack the computed gate vectors, the ordering as per pytorch
        # nn.LSTM docs: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # (see .LSTM.weight attribute)
        i, f, c_hat, o = gates.chunk(4, 1)

        # nonlinearities
        i = sigmoid(i)
        f = sigmoid(f)
        c_hat = tanh(c_hat)
        o = sigmoid(o)

        # compute new cell state (aka memory state)
        c = (f * c_past) + (i * c_hat)

        # compute new hidden state
        h = o * tanh(c)

        return h, c, {'i': i, 'f': f, 'o': o}