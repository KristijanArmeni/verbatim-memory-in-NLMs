"""module containing the Trainer() class and associated functions"""

import os, argparse, json
from datetime import datetime
import numpy as np
from sklearn import get_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
import wandb

# own modules
from layers import LSTMWithStates
from data import WT103DataModule, Dictionary
from utils import get_configs_for_dev, load_json_config

# bookkeeping etc.
import logging
from typing import Union, Tuple

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)

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
                 nonlinearity: str,
                 batch_first=True,
                 embedding_file=None,
                 lr=None,
                 beta1=None,
                 beta2=None,
                 dropout=0.2, 
                 tie_weights=False, 
                 freeze_embedding=False,
                 store_states=True, 
                 truncated_bptt_steps=0,
                 example_input_array=True,
                 loss_fct=nn.NLLLoss,
                 device="cpu"):

        super(NeuralLM, self).__init__()

        # architecture params
        nonlin = {"relu": nn.ReLU, "tanh": nn.Tanh}
        self.act_fct = nonlin[nonlinearity]()
        self.call_act_fct = self.apply_activation_function if rnn_type in ["LSTM", "LSTMWithStates"] else self.apply_activation_function_dummy
        self.weight_initrange = 0.1

        # Adam parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # pytorch lightning options
        self.truncated_bptt_steps = truncated_bptt_steps
        self.example_input_array = example_input_array
        self.loss_fct = loss_fct
        self.device_type = device

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

            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, batch_first=batch_first, 
                                            dropout=dropout)
        
        elif rnn_type == 'LSTMWithStates':

            logging.info("Using {} LSTMWithState modules as LSTM layers.".format(nlayers))

            # we use custom LSTMWithStates class which is a subclass of torch.nn.RNN
            # input arguments are the same as for torch.nn.RNN
            self.rnn = LSTMWithStates(ninp, nhid, nlayers, batch_first=batch_first, 
                                      dropout=dropout, store_states=store_states)

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
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2017) https://arxiv.org/abs/1608.05859 
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2017)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
    
    def init_weights(self, freeze_embedding):
        """ Initialize encoder and decoder weights to uniform distritbution """

        logging.info(f"Initializing weights to uniform distribution with range {-self.weight_initrange} to {self.weight_initrange}")
        logging.info(f"Initializing bias to {0}")

        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-self.weight_initrange, self.weight_initrange)
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-self.weight_initrange, self.weight_initrange)

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

        if self.rnn_type in ['LSTM', 'LSTMWithStates']:

            return (torch.zeros(self.nlayers, bsz, self.nhid).to(self.device_type),
                    torch.zeros(self.nlayers, bsz, self.nhid).to(self.device_type))
        
        else:

            return torch.zeros(self.nlayers, bsz, self.nhid).to(self.device_type)

    def detach_hidden(self, hiddens: Union[torch.Tensor, Tuple]):

        #if lstm, unpack hidden tuple and return a tuple
        if self.rnn_type in ["LSTM", "LSTMWithStates"]:
            hidden, cellstate = hiddens
            hiddens_detached = (hidden.detach(), cellstate.detach())
        else:
            hiddens_detached = hidden.detach()

        return hiddens_detached

    def set_parameters(self, init_val):

        for weight in self.rnn.parameters():
            weight.data.fill_(init_val)

        self.encoder.weight.data.fill_(init_val)
        self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):

        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def count_params(self):

        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        unit = 1e6
        
        logging.info(f"Model params\n" +
                     f"Total params (M): {round(total_params/unit, 1)}\n" +
                     f"Trainable params (M): {round(train_params/unit, 1)}\n"
                     f"Non-trainable params (M): {round(total_params/unit, 1) - round(train_params/unit, 1)}")

        return {"total": total_params, "trainable": train_params}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr,
                                     betas=(self.beta1, self.beta2))

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/3, end_factor=1.0, total_iters=5)

        return [optimizer], [lr_scheduler]

    def apply_activation_function(self, input_tuple):
        """
        activation function for LSTM inputs as nn.LSTM doesn't accept activation function
        """
        h, m = input_tuple
        outputs = (self.act_fct(h), self.act_fct(m))

        return outputs

    def apply_activation_function_dummy(inputs):
        """ Dummy identity function, because nn.RNN applies activation function internaly"""
        return inputs

    def forward(self, observation, hidden):

        emb = self.drop(self.encoder(observation))
        output, hidden = self.rnn(emb, hidden)
        #hidden = self.call_act_fct(hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))

        # reshape to (batch_size, vocab_size, sequence_len)
        logits = decoded.reshape(output.size(0), decoded.size(1), output.size(1))

        return (logits, hidden)

    def training_step(self, batch, batch_idx, hiddens=None):

        # inputs.shape = (batch, sequence_len_bptt, n_features)
        inputs, targets = batch

        # initialize hiddens at the start of the epoch
        if batch_idx == 0:
            hiddens = self.init_hidden(bsz=inputs.shape[0])  # assuming batch_first = True, so dim 0 is batch size

        # does the forward pass for all tokens in the sequence
        lm_logits, hiddens = self.forward(observation=inputs, hidden=hiddens)

        # targets are shifted inside WT103Dataset() class upon .setup()
        loss = self.loss_fct(lm_logits, targets)

        self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True)

        # detach hidden state (.detach_hidden() unpacks tuple under the hood)
        hiddens_detached = self.detach_hidden(hiddens)

        # make sure to detach hidden state
        return {"loss": loss, "hiddens": hiddens_detached}

    def validation_step(self, batch, batch_idx, hiddens=None):

        # targets are created in WT103DataModule.val_dataloader()
        inputs, targets = batch
        
        # initialize hiddens somehow
        if batch_idx == 0:
            hiddens = self.init_hidden(bsz=inputs.shape[0])  # assuming batch_first = True, so dim 0 is batch size

        lm_logits, hiddens = self.forward(observation=inputs, hidden=hiddens)

        loss = self.loss_fct(lm_logits, targets)

        self.log("val_loss", loss, prog_bar=True, on_step=True, logger=True)

        return {"loss": loss, "hiddens": hiddens}

    def test_step(self, batch, batch_idx, hiddens=None):

        # targets are created in WT103DataModule.test_dataloader()
        inputs, targets = batch
        
        # initialize hiddens somehow
        if batch_idx == 0:
            hiddens = self.init_hidden(bsz=inputs.shape[0])  # assuming batch_first = True, so dim 0 is batch size

        lm_logits, hiddens = self.forward(observation=inputs, hidden=hiddens)

        loss = self.loss_fct(lm_logits, targets)

        self.log("test_loss", loss, prog_bar=True, on_step=True, logger=True)

        return {"loss": loss, "hiddens": hiddens}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument("--data_config", type=str, default="")
    parser.add_argument("--trainer_config", type=str, default="")
    parser.add_argument("--global_seed", type=int, default=9876543)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # use pytorch lightning function to set global random seed for numpy, pytorch and python built-ins
    seed_everything(args.global_seed)

    # check if config paths are passed, else load minimum defaults
    if args.model_config == "":
        model_config = get_configs_for_dev("model_config")
    else:
        model_config = load_json_config(args.model_config) 

    # data config
    if args.data_config == "": 
        data_config = get_configs_for_dev("data_config")
    else: 
        data_config = load_json_config(args.data_config)

    # set up dictionary and load up vocabulary from file
    dictionary = Dictionary()
    dictionary.load_dict(path=data_config["vocab_path"])

    # initialize main model class
    model = NeuralLM(rnn_type='LSTM',
                     ntoken=model_config['n_vocab'],
                     ninp=model_config['n_inp'],
                     nhid=model_config['n_hid'],
                     nlayers=model_config['n_layers'],
                     nonlinearity=model_config["nonlinearity"],
                     truncated_bptt_steps=model_config['truncated_bptt_steps'],
                     batch_first=True,
                     store_states=False,
                     lr=5e-5,
                     beta1=0.7,
                     beta2=0.1,
                     dropout=model_config["dropout"],
                     loss_fct=nn.CrossEntropyLoss(reduction='mean'),
                     example_input_array=model_config["example_input_array"],
                     device=device)

    model_config_str = f"model config:\n {json.dumps(model_config, indent=4)}"
    data_config_str = f"data config:\n {json.dumps(data_config, indent=4)}"
    logging.info(model_config_str)
    logging.info(data_config_str)

    print(model)

    params = model.count_params()

    #############################
    # initialize dataset module #
    #############################

    dataset = WT103DataModule(data_dir = data_config["datadir"],
                              train_fname = "wiki.train.tokens",
                              valid_fname = "wiki.valid.tokens",
                              test_fname = "wiki.test.tokens",
                              dictionary = dictionary,
                              config = data_config)


    #dataset.find_num_workers(num_workers_range = range(0, 6, 1), n_epochs=5)

    # prepare training and validation datasets
    dataset.setup(stage="fit")

    # print batch just to make sure all is good
    sequence_id = np.random.randint(len(dataset.wiki_train))
    _, _ = dataset.print_sample(idx=sequence_id, numel=7)

    ###########################
    # initialize wandb logger #
    ###########################
    if args.trainer_config == "":
        trainer_config = get_configs_for_dev("trainer_config")
    else:
        trainer_config = load_json_config(args.trainer_config)

    # for debug, log this manually, don't version this code snippet
    os.environ["WANDB_API_KEY"] = "d8b913bbe52746e27b67249e76bbedb6c49ea85b"

    now = datetime.now().strftime("%H-%M-%S")
    today = datetime.today().strftime("%Y-%m-%d")
    idstr = today + "_" + now

    if trainer_config["wandb_project"] == "":
        trainer_config["wandb_project"] = "lm-mem"
    if trainer_config["wandb_id"] == "":
        trainer_config["wandb_id"] = f"test-id-{idstr}"

    wandb.finish()  # clear any runs that might be hanging in there
    mode = "run"
    logger = WandbLogger(project=trainer_config["wandb_project"],
                         name=trainer_config["wandb_name"] + f"_{idstr}",
                         group=trainer_config['wandb_group'],
                         notes=None,
                         save_dir=None,
                         id=trainer_config["wandb_id"],
                         mode=mode,
                         )

    #############################
    # initialize trainer module #
    #############################

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    gpus = None if accelerator == "cpu" else [0]

    # save model params
    # params_fname = os.path.join(trainer_config["root_dir"],
    #                            trainer_config["wandb_name"])

    # early stopping and learning rate monitor
    callbacks = [EarlyStopping(monitor="val_loss", mode="min", min_delta=0.02, patience=4),
                 LearningRateMonitor(logging_interval='step', log_momentum=True)]

    trainer = pl.Trainer(default_root_dir=trainer_config["root_dir"],
                         accelerator=accelerator,
                         gpus=gpus,
                         fast_dev_run=False,
                         log_every_n_steps=50,
                         accumulate_grad_batches=1,
                         num_sanity_val_steps=2,
                         check_val_every_n_epoch=1,  # check validation loss during an epoch
                         val_check_interval=50,     # do validation check
                         callbacks=callbacks,
                         enable_model_summary=False,
                         auto_lr_find=True,
                         weights_save_path=None,
                         logger=logger)

    # print trainer config
    trainer_config_str = f"trainer config:\n {json.dumps(trainer_config, indent=4)}"
    logging.info(trainer_config_str)

    ################
    # run training #
    ################
    trainer.fit(model=model, 
                train_dataloaders=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader())

    trainer.validate(model=model, dataloaders=dataset.val_dataloader())

    #########################
    # test-time performance #
    #########################
    dataset.setup(stage="test")
    trainer.test(dataloaders=dataset.test_dataloader(), ckpt_path="best")

    logger.close()