""" Module for handling data files """

import gzip
import os
import torch

from nltk import sent_tokenize, word_tokenize

def isfloat(instr):
    """ Reports whether a string is floatable """
    try:
        _ = float(instr)
        return(True)
    except:
        return(False)

class Dictionary(object):
    """ Maps between observations and indices """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """ Adds a new obs to the dictionary if needed """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class SentenceCorpus(object):
    """ Loads train/dev/test corpora and dictionary """
    def __init__(self, path, vocab_file, test_flag=False, interact_flag=False,
                 checkpoint_flag=False, predefined_vocab_flag=False, lower_flag=False,
                 collapse_nums_flag=False, multisentence_test_flag=False, sentences_in_blocks=False,
                 generate_flag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt',
                 markersfname='markers.txt'):

        self.lower = lower_flag
        self.collapse_nums = collapse_nums_flag
        self.sentences_in_blocks = sentences_in_blocks  # whether or not to allow multiple sents per line
        if not (test_flag or interact_flag or checkpoint_flag or predefined_vocab_flag or generate_flag):
            # training mode
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, trainfname))
            self.valid = self.tokenize_with_unks(os.path.join(path, validfname))
            try:
                # don't require a test set at train time,
                # but if there is one, get a sense of whether unks will be required
                self.test = self.tokenize_with_unks(os.path.join(path, testfname))
            except:
                pass
            self.save_dict(vocab_file)
        else:
            # load pretrained model
            if vocab_file[-3:] == 'bin':
                self.load_dict(vocab_file)
            else:
                self.dictionary = Dictionary()
                self.load_dict(vocab_file)
            if test_flag:
                # test mode
                if multisentence_test_flag:
                    self.test = self.tokenize_with_unks(os.path.join(path, testfname))
                else:
                    self.test = self.sent_tokenize_with_unks(os.path.join(path, testfname))
            elif checkpoint_flag or predefined_vocab_flag:
                # load from a checkpoint
                self.train = self.tokenize_with_unks(os.path.join(path, trainfname))
                self.valid = self.tokenize_with_unks(os.path.join(path, validfname))
        self.markers = self.read_marker_file(os.path.join(path, markersfname))

    def save_dict(self, path):
        """ Saves dictionary to disk """
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'wb') as file_handle:
                torch.save(self.dictionary, file_handle, pickle_module=dill)
        else:
            # Assume dict is plaintext
            with open(path, 'w') as file_handle:
                for word in self.dictionary.idx2word:
                    file_handle.write(word+'\n')

    def load_dict(self, path):
        """ Loads dictionary from disk """
        assert os.path.exists(path), "Bad path: %s" % path
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'rb') as file_handle:
                fdata = torch.load(file_handle, pickle_module=dill)
                if isinstance(fdata, tuple):
                    # Compatibility with old pytorch LM saving
                    self.dictionary = fdata[3]
                self.dictionary = fdata
        else:
            # Assume dict is plaintext
            with open(path, 'r', encoding='utf-8') as file_handle:
                for line in file_handle:
                    self.dictionary.add_word(line.strip())

    def tokenize(self, path):
        """ Tokenizes a text file. """
        assert os.path.exists(path), "Bad path: %s" % path
        # Add words to the dictionary
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        if self.lower:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word.lower())
                        else:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word)

            # Tokenize file content
            with gzip.open(path, 'rb') as file_handle:
                ids = torch.LongTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word.lower())
                                token += 1
                        else:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word)
                                token += 1
        else:
            with open(path, 'r') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        if self.lower:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word.lower())
                        else:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as file_handle:
                ids = torch.LongTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word.lower())
                                token += 1
                        else:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word)
                                token += 1
        return ids

    def tokenize_with_unks(self, path):
        """ Tokenizes a text file, adding unks if needed. """
        assert os.path.exists(path), "Bad path: %s" % path
        if path[-2:] == 'gz':
            # Determine the length of the corpus
            with gzip.open(path, 'rb') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with gzip.open(path, 'rb') as file_handle:
                ids = torch.LongTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                # Convert OOV to <unk>
                                if word.lower() not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word.lower()]
                                token += 1
                        else:
                            for word in words:
                                # Convert OOV to <unk>
                                if word not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word]
                                token += 1
        else:
            # Determine the length of the corpus
            with open(path, 'r') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with open(path, 'r') as file_handle:
                ids = torch.LongTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                # Convert OOV to <unk>
                                if word.lower() not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word.lower()]
                                token += 1
                        else:
                            for word in words:
                                # Convert OOV to <unk>
                                if word not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word]
                                token += 1
        return ids

    def read_marker_file(self, path):

        # read markers.txt line by line and store the contents to markers
        with open(path, 'r') as file_handle:
            
            # read variable names from header (first row)
            # we are currently assuming at least three columns below
            colnames = file_handle.readline().strip("\n").split("\t")
            markers = {key: [] for key in colnames}
            
            for line in file_handle:
                # read the line containing marker values and prompt labels
                # and make it a list (e.g [0, 0, 0, 0, 1, 1, 1, 1])
                # first (markers, condition_label1, condition_label2)
                row_values = line.strip("\n").split("\t")
                
                tmp = [int(el) for el in row_values[0].strip("[]").split(",")]
                markers[colnames[0]].append(tmp) #these are the markers
                markers[colnames[1]].append(row_values[1])  # this codes ex. condition
                markers[colnames[2]].append(row_values[2])  # this codes ex. condition

        return markers

    def sent_tokenize_with_unks(self, path):
        """ Tokenizes a text file into sentences, adding unks if needed. """
        assert os.path.exists(path), "Bad path: %s" % path
        all_ids = []
        sents = []
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as file_handle:
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        ids = self.convert_to_ids(words)
                        all_ids.append(ids)
        else:
            with open(path, 'r') as file_handle:
                for fchunk in file_handle:

                    if self.sentences_in_blocks:
                        # treat all sentences on one line as a block of sentences
                        blocks = [fchunk]
                    else:
                        # otherwise, split line content into several sentences
                        blocks = sent_tokenize(fchunk)

                    for line in blocks:
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        ids = self.convert_to_ids(words)
                        all_ids.append(ids)
        return (sents, all_ids)

    def online_tokenize_with_unks(self, line):
        """ Tokenizes an input sentence, adding unks if needed. """
        all_ids = []
        sents = [line.strip()]

        words = ['<eos>'] + line.strip().split() + ['<eos>']

        ids = self.convert_to_ids(words)
        all_ids.append(ids)
        return (sents, all_ids)

    def convert_to_ids(self, words, tokens=None):
        if tokens is None:
            tokens = len(words)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        if self.lower:
            for word in words:
                # Convert OOV to <unk>
                if word.lower() not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                token += 1
        else:
            for word in words:
                # Convert OOV to <unk>
                if word not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word]
                token += 1
        return(ids)