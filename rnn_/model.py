# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:27:30 2021

@RNNModel from: vansky/neural-complexity-master
@author: karmeni1
"""

import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh
import logging


logging.basicConfig(format="[%(name)s INFO] %(message)s", level=logging.INFO)

# ===== CORE inherited RNN class that will the pretrained weights =====
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 embedding_file=None, dropout=0.5, tie_weights=False, freeze_embedding=False,
                 store_states=True):
        
        super(RNNModel, self).__init__()
       
        # dropout layer
        self.drop = nn.Dropout(dropout)
        
        # input layer
        if embedding_file:
            # Use pre-trained embeddings
            embed_weights = self.load_embeddings(embedding_file, ntoken, ninp)
            self.encoder = nn.Embedding.from_pretrained(embed_weights)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        
        # hidden layer
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type == 'LSTMWithStates':
            
            logging.info("Using {} LSTMWithState modules as LSTM layers.".format(nlayers))
            
            # we use custom LSTMWithStates class which is a subclass of torch.nn.RNN
            # input arguments are the same as for torch.nn.RNN
            self.rnn = LSTMWithStates(ninp, nhid, nlayers, dropout=dropout, 
                                      store_states=store_states)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        
        # output layer
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights(freeze_embedding)
        if freeze_embedding:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2017)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers:
        # A Loss Framework for Language Modeling" (Inan et al. 2017)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, freeze_embedding):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def zero_parameters(self):
        """ Set all parameters to zero (likely as a baseline) """
        self.encoder.weight.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.fill_(0)
        for weight in self.rnn.parameters():
            weight.data.fill_(0)

    def random_parameters(self):
        """ Randomly initialize all RNN parameters but not the encoder or decoder """
        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def load_embeddings(self, embedding_file, ntoken, ninp):
        """ Load pre-trained embedding weights """
        weights = np.empty((ntoken, ninp))
        with open(embedding_file, 'r') as in_file:
            ctr = 0
            for line in in_file:
                weights[ctr, :] = np.array([float(w) for w in line.strip().split()[1:]])
                ctr += 1
        return(torch.tensor(weights).float())

    def forward(self, observation, hidden):
        emb = self.drop(self.encoder(observation))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        """ Initialize a fresh hidden state """
        weight = next(self.parameters()).data
        if self.rnn_type in ['LSTM', 'LSTMWithStates']:
            return (torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def set_parameters(self,init_val):
        for weight in self.rnn.parameters():
            weight.data.fill_(init_val)
        self.encoder.weight.data.fill_(init_val)
        self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):
        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)
            

# Class for storing LSTM outputs
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

        
        
    
    
    
