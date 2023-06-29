import sys
from test_data import awd_lstm_dataset, awd_lstm_weights_and_criterion
from models.awd_lstm.model import RNNModel as AWD_RNNModel
from wm_suite.experiment import Experiment
import torch

sys.path.append("/home/ka2773/project/lm-mem/src/models/awd_lstm")

def test_awd_lstm(awd_lstm_dataset, awd_lstm_weights_and_criterion):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = awd_lstm_dataset

    model, criterion = awd_lstm_weights_and_criterion

    model.rnn = None  #TEMP, add this dummy attribute to make code below run

    model.eval()

    # use a copy of evaluate from awd_lstm.main.py to get logprobs for QRNN/AWD_LSTM
    eval_batch_size = 1

    exp = Experiment(model=model.to(device), 
                     criterion=criterion.to(device), 
                     rnn_type="AWD_LSTM", 
                     store_states=False, 
                     dictionary=ds.dictionary, 
                     device=device)

    outputs = exp.run(input_set=ds.seq_ids, metainfo=ds.meta.__dict__)