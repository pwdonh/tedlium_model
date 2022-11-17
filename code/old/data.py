import os
import torch
import numpy as np
from torch.autograd import Variable

def data_shuffle(datafile, is_shuffle=True):
    data_words = datafile['data_words']
    data_durs = datafile['data_durs']
    storynames = datafile['talknames']
    vocab_size = len(datafile['vocab'])
    story_list = np.arange(len(storynames))
    if is_shuffle:
        np.random.shuffle(story_list)
    data_w = []
    data_d = []
    for (ii,(words,durs)) in enumerate(zip(data_words[story_list],data_durs[story_list])):
        data_w = data_w + list(words)
        if ii>0:
            data_d[-1][1] += 10.
        data_d = data_d + list(durs)
    return (np.array(data_w,dtype='int32')[:,0], np.array(data_d,dtype='float32'))

def data_shuffle_combined(datafile, is_shuffle=True):
    data = []
    variables = ['data_words_last','data_words_curr','data_words_next',
                 'data_phones','data_durs']
    storynames = datafile['talknames']
    story_list = np.arange(len(storynames))
    if is_shuffle:
        np.random.shuffle(story_list)
    for variable in variables:
        data.append([])
        for talk_data in datafile[variable][story_list]:
            if variable=='data_durs':
                talk_data[-1][1] += 10
            data[-1] = data[-1]+list(talk_data)
        data[-1] = np.array(data[-1])
    return tuple(data)

def data_shuffle_mfcc(datafile, is_shuffle=True):
    data = []
    variables = ['data_words','data_phones','data_mfcc']
    storynames = datafile['talknames']
    story_list = np.arange(len(storynames))
    if is_shuffle:
        np.random.shuffle(story_list)
    for variable in variables:
        data.append([])
        for talk_data in datafile[variable][story_list]:
            for t_d in talk_data:
                data[-1].append(t_d)
        data[-1] = np.concatenate(data[-1],0)
    return tuple(data)

def data_producer(data, batch_size, seq_length, qu_steps=20, cuda=False, use_durs='input', evaluation=False):
    datanp = np.array(data[0]).astype('int32')
    datanp_d = np.array(data[1]).astype('float32')
    if use_durs=='output':
        datanp = datanp[1:]
        datanp_d = datanp_d[:-1,:]
    batch_len = int(np.floor(datanp.shape[0]/batch_size))
    # prepare word data
    datanp = np.reshape(datanp[0 : batch_size * batch_len],
                        [batch_size, batch_len])
    datanp = torch.from_numpy(datanp.T).long()
    # prepare duration data
    datanp_d = np.reshape(datanp_d.flatten()[0 : batch_size * batch_len * 2],
                      [batch_size, batch_len, 2])
    bins = np.logspace(np.log10(.05), np.log10(10.),qu_steps-1)
    datanp_d = np.digitize(datanp_d, bins)
    datanp_d_s = np.digitize(datanp_d.sum(2), bins)
    datanp_d = np.array(datanp_d, dtype='int32')
    datanp_d_s = np.array(datanp_d_s, dtype='int32')
    datanp_d = torch.from_numpy(datanp_d).long().transpose(0,1).contiguous()
    datanp_d_s = torch.from_numpy(datanp_d_s).long().transpose(0,1).contiguous()
    if cuda:
        datanp = datanp.cuda()
        datanp_d = datanp_d.cuda()
        datanp_d_s = datanp_d_s.cuda()
    epoch_size = (batch_len - 1) // seq_length
    pointers = np.arange(0,batch_len-seq_length,seq_length)
    for (ii, pointer) in enumerate(pointers):
        x = datanp[pointer:pointer+seq_length,:]
        x_d = datanp_d[pointer:pointer+seq_length,:,:]
        y = datanp[pointer+1:pointer+seq_length+1,:]
        y_d = datanp_d_s[pointer+1:pointer+seq_length+1,:]
        yield ((Variable(x, volatile=evaluation), Variable(x_d, volatile=evaluation)),
               (Variable(y.view(-1), volatile=evaluation),
                Variable(y_d.view(-1), volatile=evaluation)), ii)

def data_producer_combined(data, batch_size, seq_length, qu_steps=20, cuda=False, use_durs='input', evaluation=False):

    batch_len = int(np.floor(data[0].shape[0]/batch_size))
    epoch_size = (batch_len - 1) // seq_length
    pointers = np.arange(0,batch_len-seq_length,seq_length)
    bins = np.logspace(np.log10(.05), np.log10(10.),qu_steps-1)

    # token data
    datanp = []
    for data_int in data[:-1]:
        datanp.append(np.array(data_int).astype('int32'))
        datanp[-1] = np.reshape(datanp[-1][0 : batch_size * batch_len],
                                [batch_size, batch_len])
        datanp[-1] = torch.from_numpy(datanp[-1].T).long()
        if cuda:
            datanp[-1] = datanp[-1].cuda()
    # duration data
    datanp_d = np.array(data[-1]).astype('float32')
    datanp_d = np.reshape(datanp_d.flatten()[0 : batch_size * batch_len * 2],
                      [batch_size, batch_len, 2])
    datanp_d = np.digitize(datanp_d, bins)
    datanp_d = np.array(datanp_d, dtype='int32')
    datanp_d = torch.from_numpy(datanp_d).long().transpose(0,1).contiguous()
    if cuda:
        datanp_d = datanp_d.cuda()

    for (ii, pointer) in enumerate(pointers):
        x = [Variable(d[pointer:pointer+seq_length,:], volatile=evaluation) for d in datanp]
        x_d = Variable(datanp_d[pointer:pointer+seq_length,:,:], volatile=evaluation)
        y = [Variable(d[pointer+1:pointer+seq_length+1,:], volatile=evaluation) for d in datanp]
        # Input: previous phone, previous word, duration of previous phone and pause
        # Output: current phone, current word, next word
        yield ((x[3],x[0],x_d),(y[3].view(-1),y[1].view(-1),y[2].view(-1)),ii)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
