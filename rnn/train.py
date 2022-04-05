"""module containing the Trainer() class and associated functions"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh
import pytorch_lightning as pl

# own modules
from layers import LSTMWithStates
from data import WT103DataModule, Dictionary

# bookkeeping etc.
import logging

# ===== LSTM CLASS ===== #

class NeuralLM(pl.LightningModule):
    """own version of the RNNModule() class
    """
    def __init__(self, 
                 rnn_type: str, 
                 ntoken: int, 
                 ninp: int, 
                 nhid: int, 
                 nlayers: int,
                 batch_first=True,
                 embedding_file=None,
                 dropout=0.5, 
                 tie_weights=False, 
                 freeze_embedding=False,
                 store_states=True, 
                 truncated_bptt_steps=0,
                 loss_fct=nn.NLLLoss):

        super(NeuralLM, self).__init__()

        # basic properties
        self.truncated_bptt_steps = truncated_bptt_steps
        self.loss_fct = loss_fct
        
        self.drop = nn.Dropout(dropout) # dropout layer


        # set self.encoder attr, input layer
        if embedding_file:
            # Use pre-trained embeddings
            embed_weights = self.load_embeddings(embedding_file, ntoken, ninp)
            self.encoder = nn.Embedding.from_pretrained(embed_weights)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)

        # set self.rnn attr, hidden layers
        if rnn_type in ['LSTM', 'GRU']:

            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, batch_first=batch_first, dropout=dropout)
        
        elif rnn_type == 'LSTMWithStates':

            logging.info("Using {} LSTMWithState modules as LSTM layers.".format(nlayers))

            # we use custom LSTMWithStates class which is a subclass of torch.nn.RNN
            # input arguments are the same as for torch.nn.RNN
            self.rnn = LSTMWithStates(ninp, nhid, nlayers, batch_first=batch_first, dropout=dropout,
                                      store_states=store_states)

        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, batch_first=batch_first, dropout=dropout)

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

    def init_hidden(self, bsz):
        """ Initialize a fresh hidden state """

        weight = next(self.parameters()).data
        
        if self.rnn_type in ['LSTM', 'LSTMWithStates']:

            return (torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        
        else:

            return torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def set_parameters(self, init_val):

        for weight in self.rnn.parameters():
            weight.data.fill_(init_val)

        self.encoder.weight.data.fill_(init_val)
        self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):

        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=1e-5)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return [optimizer], [lr_scheduler]

    def forward(self, observation, hidden):

        emb = self.drop(self.encoder(observation))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))

        return (decoded.reshape(output.size(0), output.size(1), decoded.size(1)), hidden)

    def training_step(self, batch, batch_idx, hiddens=None):

        # inputs.shape = (batch, sequence_len_bptt, n_features)
        inputs, targets = batch

        # initialize hiddens somehow
        if batch_idx == 0:
            hiddens = self.init_hidden(bsz=inputs.shape[0])  # assuming batch_first = True, so dim 0 is batch size

        # does the forward pass for all tokens in the sequence
        lm_logits, hiddens = self.forward(observation=inputs, hidden=hiddens)

        # targets are shifted inside WT103Dataset() class upon .setup()
        loss = self.loss_fct(lm_logits, targets)

        self.log("train_loss", loss, prog_bar=True, on_step=True)

        # make sure to detach hidden state
        return {"loss": loss, "hiddens": hiddens.detach()}

    def validation_step(self, batch, batch_idx, hiddens=None):

        # targets are created in WT103DataModule.val_dataloader()
        inputs, targets = batch
        
        # initialize hiddens somehow
        if batch_idx == 0:
            hiddens = self.init_hidden(bsz=inputs.shape[0])  # assuming batch_first = True, so dim 0 is batch size

        lm_logits, hiddens = self.forward(observation=inputs, hidden=hiddens)

        loss = self.loss_fct(lm_logits, targets)

        self.log("val loss", loss, prog_bar=True, on_step=True)

        return {"loss": loss, "hiddens": hiddens}

    def test_step(self, batch, batch_idx, hiddens=None):

        # targets are created in WT103DataModule.val_dataloader()
        inputs, targets = batch
        
        # initialize hiddens somehow
        if batch_idx == 0:
            hiddens = self.init_hidden(bsz=inputs.shape[0])  # assuming batch_first = True, so dim 0 is batch size

        lm_logits, hiddens = self.forward(observation=inputs, hidden=hiddens)

        loss = self.loss_fct(lm_logits, targets)

        self.log("test loss", loss, prog_bar=True, on_step=True)

        return {"loss": loss, "hiddens": hiddens}


if __name__ == "__main__":

    model_config = {

        "n_vocab": 50000,
        "n_inp": 50,
        "n_hid": 100,
        "n_layers": 4,
        "truncated_bptt_steps": 20, 

    }

    # initialize main model class
    model = NeuralLM(rnn_type='LSTM',
                     ntoken=model_config['n_vocab'],
                     ninp=model_config['n_inp'],
                     nhid=model_config['n_hid'],
                     nlayers=model_config['n_layers'],
                     batch_first=True,
                     store_states=False,
                     loss_fct=nn.NLLLoss(reduction='mean'))


    # set up dictionary and load up vocabulary from file
    vocab = "C:\\Users\\karmeni1\\project\\lm-mem\\src\\rnn\\vocab.txt"
    dictionary = Dictionary()
    dictionary.load_dict(path=vocab)

    # set up dataset configuration
    p = "C:\\Users\\karmeni1\\project\\lm-mem\\data\\wikitext-103"
    cfg = {
            "train_bs": 64, 
            "valid_bs": 64, 
            "test_bs": 5, 
            "per_batch_seq_len": 1000,  # sequence len per batch, this is in memory for forward pass
            "bptt_len": 20              # detach gradients every 20 tokens, pl.Trainer takes care of this
            }

    # initialize data module instance
    dataset = WT103DataModule(data_dir = p,
                              train_fname = "wiki.train.tokens",
                              valid_fname = "wiki.valid.tokens",
                              test_fname = "wiki.test.tokens",
                              dictionary = dictionary,
                              config = cfg)

    # prepare training and validation datasets
    dataset.setup(stage="fit")

    # initialize trainer module
    trainer = pl.Trainer()

    # start training
    trainer.fit(model=model, 
                train_dataloaders=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader())

    # test the model
    dataset.setup(stage="test")
    trainer.test(model=model, dataloaders=dataset.test_dataloader())