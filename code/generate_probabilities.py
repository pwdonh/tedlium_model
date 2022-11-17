# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from params import (
    OUTPUT_FOLDER, model_checkpoint_path, model_data_path, args
)

from func_model import (
    TedliumModelCombined, repackage_hidden
)
from func_model_data import (
    data_shuffle_combined, data_producer_combined
)

from tqdm import tqdm

# %%

eval_batch_size = 1

# %% Load state dict to CPU

state_dict = torch.load(
    model_checkpoint_path,
    map_location=lambda storage, loc: storage
)

# %%

def data_extract(datafile, i_talk=0):
    data = []
    variables = ['data_words_last','data_words_curr','data_words_next',
                 'data_phones','data_durs']
    for variable in variables:
        data.append([])
        for talk_data in [datafile[variable][i_talk]]:
            if variable=='data_durs':
                talk_data[-1][1] += 10
            data[-1] = data[-1]+list(talk_data)
        if np.ndim(talk_data)==1:
            data[-1] = data[-1]+[0]*args.bptt
        else:
            data[-1] = data[-1]+[[0.,0.]]*args.bptt
        data[-1] = np.array(data[-1])
    return tuple(data)

# %%

datafile = np.load(model_data_path, allow_pickle=True)

# %%

words = datafile['vocab_words'].tolist()
phones = datafile['vocab_phones'].tolist()
ntokens_word = len(words)
ntokens_phone = len(phones)

# %%

model = TedliumModelCombined(
    args.model, ntokens_word, ntokens_phone, args.emsize, 
    args.nhid, args.nlayers, qu_steps=args.qu_steps, 
    dropout=args.dropout, tie_weights=args.tied
)
model.load_state_dict(state_dict)

# %%

model.eval()
total_loss = 0
phone_loss = 0; word_loss = 0

for i_talk in tqdm(range(11)):

    data_source = data_extract(datafile, i_talk=i_talk)
    n_t = len(datafile['data_phones'][i_talk])
    n_t_pad = len(data_source[0])

    df = pd.DataFrame(
        index = np.arange(n_t_pad, dtype=int),
        columns = [
            'phone', 'word', 'target_phone', 'last_word', 
            'target_word', 'surprise', 'entropy'
        ]+phones
    )

    p_words = np.zeros((n_t_pad, ntokens_word))

    hidden = model.init_hidden(eval_batch_size)

    for (data,targets,batch) in data_producer_combined(data_source, eval_batch_size, args.bptt, cuda=args.cuda, use_durs=args.use_durs, evaluation=True):
        
        output, hidden = model(data, hidden)
        loss, loss_phone, loss_word = model.criterion(output, targets)
        total_loss += loss.data
        phone_loss += loss_phone.data
        word_loss += loss_word.data
        hidden = repackage_hidden(hidden)

        # Fill in data frame
        indices = range(batch*args.bptt, batch*args.bptt+args.bptt)
        df.loc[indices, 'phone'] = datafile['vocab_phones'][data[0][:,0].data.numpy()]
        df.loc[indices, 'target_phone'] = datafile['vocab_phones'][targets[0].data.numpy()]
        df.loc[indices, 'last_word'] = datafile['vocab_words'][data[1][:,0].data.numpy()]
        df.loc[indices, 'target_word'] = datafile['vocab_words'][targets[1].data.numpy()]

        # Phones
        p_phones = torch.nn.functional.softmax(output[0].view(-1,ntokens_phone)).data.numpy()
        df.loc[indices,phones] = p_phones
        p = np.array(df.loc[indices,phones], dtype=float)
        df.loc[indices, 'entropy'] = -np.sum(p*np.log(p),1)
        for index in indices:
            df.loc[index, 'surprise'] = -np.log(df.loc[index, df.loc[index,'target_phone']])

        # Words
        p_words[indices,:] = torch.nn.functional.softmax(output[1].view(-1,ntokens_word)).data.numpy()
        # df.loc[indices,words] = p_words

    results = (total_loss[0]/(batch+1), phone_loss[0]/(batch+1), word_loss[0]/(batch+1))

    # Post-process data frame before saving

    df = df.iloc[:n_t]
    df.iloc[-1]['target_phone'] = ''
    df.iloc[-1]['target_word'] = ''
    df.iloc[-1]['surprise'] = np.nan

    df.loc[df.index[1:],'word'] = df.iloc[:-1]['target_word'].values

    # Check if first word has two phonemes
    ws = df.iloc[:2]['last_word'].values.tolist()
    if ws[0]==ws[1]:
        df.loc[df.index[0],'word'] = df.loc[df.index[1],'word']
    else:
        print('dfdf')

    talkname = datafile['talknames'][i_talk]
    outpath = os.path.join(OUTPUT_FOLDER, talkname+'_phone_outputs.csv')

    df.to_csv(outpath)

# %%

# df.loc[:,['p_'+word.lower() for word in words]] = p_words[:n_t]

# outpath = os.path.join('..', 'rnn_output', talkname+'_all_outputs.csv')

# df.to_csv(outpath)