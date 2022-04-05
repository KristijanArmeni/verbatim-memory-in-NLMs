
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import os, json
from typing import Optional, List
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def isfloat(instr):
    """ Reports whether a string is floatable """
    try:
        _ = float(instr)
        return(True)
    except:
        return(False)


class Dictionary(object):
    """ Maps between observations and indices

    KA: from github.com/vansky/neural-complexity-master
    """

    def __init__(self):

        self.word2idx = {}
        self.idx2word = []

    def add_word_get_idx(self, word):
        """ Adds a new obs to the dictionary if needed """

        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def __len__(self):

        return len(self.idx2word)


    def tokens_to_ids(self, sequence: List, check_unkns=True) -> List:

        logging.info("Converting tokens to ids ...")

        ids = []

        for i, word in enumerate(tqdm(sequence, desc="sequence: ")):

            # code OOV as <unk> and numerals to <num>
            if word not in self.word2idx:

                ids.append(self.add_word_get_idx("<unk>"))

            elif isfloat(word) and '<num>' in self.word2idx:

                ids.append(self.add_word_get_idx("<num>"))

            else:
                ids.append(self.word2idx[word])

        # convert to tensor, add batch and feature dimension
        #idstens = (torch.tensor(ids,  dtype=torch.int64)
        #                        .unsqueeze(-1)
        #                        .unsqueeze(-1))

        return ids

    def load_dict(self, path):
        """ Loads dictionary from disk """

        logging.info(f"Loading vocabulary from {path}...")

        assert os.path.exists(path), "Bad path: %s" % path
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'rb') as file_handle:
                fdata = torch.load(file_handle, pickle_module=dill)
                if isinstance(fdata, tuple):
                    # Compatibility with old pytorch LM saving
                    self = fdata[3]
                self = fdata
        else:
            # Assume dict is plaintext
            with open(path, 'r', encoding='utf-8') as file_handle:
                for line in file_handle:
                    self.add_word_get_idx(line.strip())

        logging.info("Added Dictionary() class to .dictionary attribute...")


def load_tokens(path: str) -> List:

    logging.info(f"load_text_strip_split() loading {path}...")

    with open(path, "r", encoding="utf-8") as fh:
        text = " ".join([l.lower() for l in fh.readlines() if len(l.split()) != 0]).split()

    return text



class WT103Dataset(Dataset):

    def __init__(self, samples, seq_len):

        # drop any trailing samples, based on seq_len

        ids, targets = self.make_batch_sequences_and_targets(samples, sequence_len=seq_len)

        self.ids = torch.LongTensor(ids)
        self.targets = torch.LongTensor(targets)
        self.seq_len = seq_len

    def __len__(self):

        return len(self.ids)

    def __getitem__(self, idx):
        """
        returns the input sequence and the shifted sequence that gives the target tokens
        """
        return (self.ids[idx], self.targets[idx])


    def make_batch_sequences_and_targets(self, samples: List, sequence_len: int):
        """
        make_batch_sequences_and_targets() will split input list into chunks of length <sequence_len>
        and will return targets list with values shifted, to be used in nn.NLLLoss
        This method is called upon initialization of the class.
        """
        chunk_onsets = np.arange(0, len(samples), sequence_len)
        target_onsets = chunk_onsets + 1

        # see if there's trailing tokens (all chunks should be the same length)
        mod = len(samples)%sequence_len

        # if mod > 0, drop the last chunk that has trailing tokens
        if mod == 0:
            n_good_size = len(chunk_onsets)
            n_bad_size = 0
        elif mod > 0:
            chunk_onsets = chunk_onsets[0:-1] # drop the last chunk onset as that one has fewer tokens
            n_good_size = len(chunk_onsets)
            n_bad_size = 1

        logging.info(f"{type(self).__name__}: chunking sequence of {len(samples)} elements into {n_good_size} chunks of size {sequence_len}" +
                    f" and dropping {n_bad_size} chunk of size {mod}")

        # do the chunking
        chunks = []
        target_chunks = []
        for onset, target_onset in zip(chunk_onsets, target_onsets):

            chunks.append(samples[onset:onset+sequence_len])
            target_chunks.append(samples[target_onset:target_onset+sequence_len])

        return chunks, target_chunks


class WT103DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, train_fname: str, valid_fname: str, test_fname: str, 
                dictionary: Dictionary, config: dict):
        super().__init__()

        # string pointing to folder with pretokenized WT103 datasets
        self.data_dir = data_dir
    
        # strings pointing to pre-tokenized WT103 dataset
        self.train_fname = train_fname
        self.valid_fname = valid_fname
        self.test_fname = test_fname
    
        # attributes holding the WT103Dataset() classes (populated in self.setup())
        self.wiki_train = None
        self.wiki_valid = None
        self.wiki_test = None

        self.dictionary = dictionary
        self.cfg = config

    def __repr__(self) -> str:
        
        cfgprint = json.dumps(self.cfg, indent=4)

        string = f"""WikiText103()\nself.data_dir = {self.data_dir}\nself.cfg = {cfgprint}"""

        return string

    def prepare_data(self) -> None:

        pass

    def setup(self, stage: Optional[str] = None):

        if stage in (None, "fit"):

            train_fname_ids = os.path.join(self.data_dir, self.train_fname.replace("tokens", "ids.npy"))
            valid_fname_ids = os.path.join(self.data_dir, self.valid_fname.replace("tokens", "ids.npy"))
            
            # check if indices already exist, otherwise create and save to disk
            if os.path.isfile(train_fname_ids):
                logging.info(f"Loading {train_fname_ids}")
                train_ids = np.load(train_fname_ids).tolist()

            else:
                train_tokens = load_tokens(os.path.join(self.data_dir, self.train_fname))
                train_ids = self.dictionary.tokens_to_ids(train_tokens)

                logging.info(f"Saving {train_fname_ids}")
                np.save(train_fname_ids, np.array(train_ids))

            # same for validation indices
            if os.path.isfile(valid_fname_ids):
                logging.info(f"Loading {valid_fname_ids}")
                valid_ids = np.load(valid_fname_ids).tolist()

            else:
                valid_tokens = load_tokens(os.path.join(self.data_dir, self.valid_fname))
                valid_ids = self.dictionary.tokens_to_ids(valid_tokens)
                
                logging.info(f"Saving {valid_fname_ids}")
                np.save(valid_fname_ids, np.array(valid_ids))

            # train_seqs = chunk_into_sequences(tokens=train_ids, sequence_len=)
            self.wiki_train = WT103Dataset(samples=train_ids, seq_len=self.cfg["per_batch_seq_len"])
            
            # valid_seqs = chunk_into_sequences(tokens=valid_ids, sequence_len=self.cfg["bptt_len"])
            self.wiki_valid = WT103Dataset(samples=valid_ids, seq_len=self.cfg["per_batch_seq_len"])

        elif stage in (None, "test"):

            test_fname_ids = os.path.join(self.data_dir, self.test_fname.replace("tokens", "ids.npy"))

            if os.path.isfile(test_fname_ids):
                logging.info(f"Loading {test_fname_ids}")
                test_ids = np.load(test_fname_ids).tolist()

            else:
                test_tokens = load_tokens(os.path.join(self.data_dir, self.test_fname))
                test_ids = self.dictionary.tokens_to_ids(test_tokens)
                
                logging.info(f"Saving {test_fname_ids}")
                np.save(test_fname_ids, np.array(test_ids))

            #test_seqs = chunk_into_sequences(tokens=test_ids, sequence_len=len(test_ids))
            self.wiki_test = WT103Dataset(samples=test_ids, seq_len=self.cfg["per_batch_seq_len"])

    def train_dataloader(self):

        return DataLoader(dataset=self.wiki_train, batch_size=self.cfg["train_bs"], shuffle=False)

    def val_dataloader(self):
        
        return DataLoader(dataset=self.wiki_valid, batch_size=self.cfg["valid_bs"], shuffle=False)

    def test_dataloader(self):

        return DataLoader(dataset=self.wiki_test, batch_size=self.cfg["test_bs"], shuffle=False)

