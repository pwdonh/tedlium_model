import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import data as mdata
import model

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--lnorm', action='store_true',
                    help='use layer normalization')
parser.add_argument('--adam', action='store_true',
                    help='use the ADAM optimizer instead of pure SGD')
parser.add_argument('--slope_anneal', type=float, default=.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--use_durs', type=str, default='no',
                    help='use durations in training')
parser.add_argument('--tier', type=str, default='words',
                    help='use durations in training')
parser.add_argument('--qu_steps', type=int, default=20,
                    help='how many steps to use for time binning')
parser.add_argument('--device', type=int, default=0,
                    help='which cuda device to use (index)')
args = parser.parse_args()

print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.device)

step_slope = 1.

###############################################################################
# Load data
###############################################################################

eval_batch_size = 10

path_train = './data/data_'+args.tier+'_train.npz'
path_dev = './data/data_'+args.tier+'_dev.npz'
path_test = './data/data_'+args.tier+'_test.npz'

datafile_train = np.load(path_train)
datafile_dev = np.load(path_dev)
datafile_test = np.load(path_test)

if args.tier=='combined':
    ntokens_word = len(datafile_train['vocab_words'])
    ntokens_phone = len(datafile_train['vocab_phones'])
    data_shuffle = mdata.data_shuffle_combined
    data_producer = mdata.data_producer_combined
else:
    ntokens = len(datafile_train['vocab'])
    data_shuffle = mdata.data_shuffle
    data_producer = mdata.data_producer


val_data = data_shuffle(datafile_dev, is_shuffle=False)
test_data = data_shuffle(datafile_test, is_shuffle=False)
import pdb; pdb.set_trace()
###############################################################################
# Build the model
###############################################################################

if args.tier=='combined':
    model = model.TedliumModelCombined(args.model, ntokens_word, ntokens_phone, args.emsize, args.nhid, args.nlayers, qu_steps=args.qu_steps, dropout=args.dropout, tie_weights=args.tied)
else:
    if args.use_durs=='no':
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, dropout=args.dropout, tie_weights=args.tied)
    elif args.use_durs=='input':
        model = model.TedliumModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, qu_steps=args.qu_steps, dropout=args.dropout, tie_weights=args.tied, is_lnorm=args.lnorm)
    elif args.use_durs=='output':
        model = model.TedliumModelPredictDurs(args.model, ntokens, args.emsize, args.nhid, args.nlayers, qu_steps=args.qu_steps, dropout=args.dropout, tie_weights=args.tied)
if args.cuda:
    model.cuda()

# criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    phone_loss = 0; word_loss = 0
    hidden = model.init_hidden(eval_batch_size)
    for (data,targets,batch) in data_producer(data_source, eval_batch_size, args.bptt, cuda=args.cuda, use_durs=args.use_durs, evaluation=True):
        output, hidden = model(data, hidden)
        if args.tier=='combined':
            loss, loss_phone, loss_word = model.criterion(output, targets)
            total_loss += loss.data
            phone_loss += loss_phone.data
            word_loss += loss_word.data
        else:
            total_loss += model.criterion(output, targets).data
        hidden = repackage_hidden(hidden)
    if args.tier=='combined':
        return (total_loss[0]/(batch+1), phone_loss[0]/(batch+1), word_loss[0]/(batch+1))
    else:
        return total_loss[0]/(batch+1)

def train():
    # Turn on training mode which enables dropout.
    print('Load training data')
    model.train()
    if hasattr(model.rnn, 'step_slope'):
        model.rnn.step_slope = step_slope
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    # Shuffle order of talks
    train_data = data_shuffle(datafile_train)
    print('Start training')
    for (data,targets,batch) in data_producer(train_data, args.batch_size, args.bptt, cuda=args.cuda, use_durs=args.use_durs):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        if args.tier=='combined':
            loss, loss_phone, loss_word = model.criterion(output, targets)
        else:
            loss = model.criterion(output, targets)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data[0]) // args.batch_size // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

if args.adam:
    optimizer = optim.Adam(model.parameters(), lr)
else:
    optimizer = optim.SGD(model.parameters(), lr)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        print('-' * 89)
        if args.tier=='combined':
            val_loss, val_loss_phone, val_loss_word = evaluate(val_data)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss phones {:5.2f} | '
                    'valid ppl phones {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss_phone, math.exp(val_loss_phone)))
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss words {:5.2f} | '
                    'valid ppl words {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss_word, math.exp(val_loss_word)))
        else:
            val_loss = evaluate(val_data)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))

        print('-' * 89)
        step_slope = np.min([5.,1+args.slope_anneal*epoch])
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load(f))

# Run on test data.
print('=' * 89)
if args.tier=='combined':
    test_loss, test_loss_phone, test_loss_word = evaluate(test_data)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss_phone, math.exp(test_loss_phone)))
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss_word, math.exp(test_loss_word)))
else:
    test_loss = evaluate(test_data)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))

print('=' * 89)
