import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from rnn_modules import *

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, is_lnorm=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        elif rnn_type in ['LSTM2', 'HMLSTM']:
            exec("self.rnn = "+rnn_type+"(ninp, nhid, nlayers, dropout=dropout, is_lnorm=is_lnorm)")
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = ntoken

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input[0]))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type in ['LSTM', 'LSTM2']:
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        elif self.rnn_type == 'HMLSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()+1),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()+1),
                    Variable(weight.new(self.nlayers, bsz).zero_()+1))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def criterion(self, output, targets):
        loss_fcn = nn.CrossEntropyLoss()
        loss = loss_fcn(output.view(-1, self.nout), targets[0])
        return loss


class TedliumModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, qu_steps=20, dropout=0.5, tie_weights=False, is_lnorm=False):
        super(TedliumModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder_dursw = nn.Embedding(qu_steps, qu_steps)
        self.encoder_dursp = nn.Embedding(qu_steps, qu_steps)
        ninp_rnn = ninp + 2*qu_steps
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp_rnn, nhid, nlayers, dropout=dropout)
        elif rnn_type in ['LSTM2', 'HMLSTM']:
            exec("self.rnn = "+rnn_type+"(ninp_rnn, nhid, nlayers, dropout=dropout, is_lnorm=is_lnorm)")
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = ntoken

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.encoder_dursw.weight.data.uniform_(-initrange, initrange)
        self.encoder_dursp.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb_word = self.drop(self.encoder(input[0]))
        emb_durw = self.drop(self.encoder_dursw(input[1][:,:,0]))
        emb_durp = self.drop(self.encoder_dursp(input[1][:,:,1]))
        emb = torch.cat([emb_word, emb_durw, emb_durp], 2)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type in ['LSTM', 'LSTM2']:
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        elif self.rnn_type == 'HMLSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()+1),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()+1),
                    Variable(weight.new(self.nlayers, bsz).zero_()+1))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def criterion(self, output, targets):
        loss_fcn = nn.CrossEntropyLoss()
        loss = loss_fcn(output.view(-1, self.nout), targets[0])
        return loss

class TedliumModelPredictDurs(TedliumModel):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, qu_steps=20, dropout=0.5, tie_weights=False):
        super(TedliumModelPredictDurs, self).__init__(rnn_type, ntoken, ninp, nhid, nlayers, qu_steps, dropout, tie_weights)

        self.decoder = nn.Linear(nhid, qu_steps)

        if tie_weights:
            raise ValueError('No tied weights for duration prediction')
            # self.decoder.weight = self.encoder_dursp.weight

        self.init_weights()

        self.nout = qu_steps

    def criterion(self, output, targets):
        loss_fcn = nn.CrossEntropyLoss()
        loss = loss_fcn(output.view(-1, self.nout), targets[1])
        return loss

class TedliumModelCombined(TedliumModel):

    def __init__(self, rnn_type, ntoken_word, ntoken_phone, ninp, nhid, nlayers, qu_steps=20, dropout=0.5, tie_weights=False):

        super(TedliumModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder_word = nn.Embedding(ntoken_word, ninp)
        # ninp_phone = max((int(ninp/5), ntoken_phone))
        self.encoder_phone = nn.Embedding(ntoken_phone, ninp)
        self.encoder_dursw = nn.Embedding(qu_steps, qu_steps)
        self.encoder_dursp = nn.Embedding(qu_steps, qu_steps)
        ninp_rnn = 2*ninp + 2*qu_steps
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp_rnn, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp_rnn, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder_phone = nn.Linear(nhid, ntoken_phone)
        self.decoder_word = nn.Linear(nhid, ntoken_word)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_phone.weight = self.encoder_phone.weight
            self.decoder_word.weight = self.encoder_word.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout_phone = ntoken_phone
        self.nout_word = ntoken_word

    def init_weights(self):
        initrange = 0.1
        self.encoder_phone.weight.data.uniform_(-initrange, initrange)
        self.encoder_word.weight.data.uniform_(-initrange, initrange)
        self.decoder_phone.bias.data.fill_(0)
        self.decoder_phone.weight.data.uniform_(-initrange, initrange)
        self.decoder_word.bias.data.fill_(0)
        self.decoder_word.weight.data.uniform_(-initrange, initrange)
        self.encoder_dursw.weight.data.uniform_(-initrange, initrange)
        self.encoder_dursp.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb_phone = self.drop(self.encoder_phone(input[0]))
        emb_word = self.drop(self.encoder_word(input[1]))
        emb_durw = self.drop(self.encoder_dursw(input[2][:,:,0]))
        emb_durp = self.drop(self.encoder_dursp(input[2][:,:,1]))
        emb = torch.cat([emb_phone, emb_word, emb_durw, emb_durp], 2)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded_phone = self.decoder_phone(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded_word = self.decoder_word(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded_phone = decoded_phone.view(output.size(0), output.size(1), decoded_phone.size(1))
        decoded_word = decoded_word.view(output.size(0), output.size(1), decoded_word.size(1))
        return (decoded_phone, decoded_word), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def criterion(self, output, targets):
        loss_fcn = nn.CrossEntropyLoss()
        loss_phone = loss_fcn(output[0].view(-1, self.nout_phone), targets[0])
        loss_word = loss_fcn(output[1].view(-1, self.nout_word), targets[1])
        loss = (loss_phone+loss_word)/2
        return (loss,loss_phone,loss_word)

class TedliumModelCombined2(TedliumModel):

    def __init__(self, rnn_type, ntoken_word, ntoken_phone, ninp, nhid, nlayers, qu_steps=20, dropout=0.5, tie_weights=False):

        super(TedliumModel, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder_word = nn.Embedding(ntoken_word, ninp)
        # ninp_phone = max((int(ninp/5), ntoken_phone))
        self.encoder_phone = nn.Embedding(ntoken_phone, ninp)
        self.encoder_dursw = nn.Embedding(qu_steps, qu_steps)
        self.encoder_dursp = nn.Embedding(qu_steps, qu_steps)
        ninp_rnn = 2*ninp + 2*qu_steps
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp_rnn, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp_rnn, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder_phone = nn.Linear(nhid, ntoken_phone)
        self.decoder_word = nn.Linear(nhid, ntoken_word)
        self.decoder_word2 = nn.Linear(nhid, ntoken_word)

        # if tie_weights:
        #     if nhid != ninp:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder_phone.weight = self.encoder_phone.weight
        #     self.decoder_word.weight = self.encoder_word.weight
        #     self.decoder_word2.weight = self.encoder_word.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nout_phone = ntoken_phone
        self.nout_word = ntoken_word

    def init_weights(self):
        initrange = 0.1
        self.encoder_phone.weight.data.uniform_(-initrange, initrange)
        self.encoder_word.weight.data.uniform_(-initrange, initrange)
        self.decoder_phone.bias.data.fill_(0)
        self.decoder_phone.weight.data.uniform_(-initrange, initrange)
        self.decoder_word.bias.data.fill_(0)
        self.decoder_word.weight.data.uniform_(-initrange, initrange)
        self.decoder_word2.bias.data.fill_(0)
        self.decoder_word2.weight.data.uniform_(-initrange, initrange)
        self.encoder_dursw.weight.data.uniform_(-initrange, initrange)
        self.encoder_dursp.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb_phone = self.drop(self.encoder_phone(input[0]))
        emb_word = self.drop(self.encoder_word(input[1]))
        emb_durw = self.drop(self.encoder_dursw(input[2][:,:,0]))
        emb_durp = self.drop(self.encoder_dursp(input[2][:,:,1]))
        emb = torch.cat([emb_phone, emb_word, emb_durw, emb_durp], 2)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded_phone = self.decoder_phone(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded_word = self.decoder_word(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded_word2 = self.decoder_word2(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded_phone = decoded_phone.view(output.size(0), output.size(1), decoded_phone.size(1))
        decoded_word = decoded_word.view(output.size(0), output.size(1), decoded_word.size(1))
        decoded_word2 = decoded_word2.view(output.size(0), output.size(1), decoded_word2.size(1))
        return (decoded_phone, decoded_word, decoded_word2), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def criterion(self, output, targets):
        loss_fcn = nn.CrossEntropyLoss()
        loss_phone = loss_fcn(output[0].view(-1, self.nout_phone), targets[0])
        loss_word = loss_fcn(output[1].view(-1, self.nout_word), targets[1])
        loss_word2 = loss_fcn(output[2].view(-1, self.nout_word), targets[2])
        loss = (loss_phone+loss_word+loss_word2)/3
        return (loss,loss_phone,loss_word,loss_word2)
