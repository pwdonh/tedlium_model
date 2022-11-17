# %%

import numpy as np
import os.path as op
from argparse import Namespace

# import os.path as op
# from glob import glob
# talks = [op.splitext(op.basename(filename))[0] for filename in glob('alignments/*.TextGrid')]

MODEL_FOLDER = '../models'
DATA_FOLDER = '../data'
OUTPUT_FOLDER = '../output'

model_name = 'tedlium_yesdurs_combined_out_phones_setting_6'
model_checkpoint_path = op.join(MODEL_FOLDER, 'tedlium_yesdurs_combined_setting_6.pt')
model_data_path = op.join(DATA_FOLDER, 'data_combined_test.npz')

# Hardcode the settings we used for the paper (corresponding to checkpoint "..settings_6.pt")
args = Namespace()
args.model = 'LSTM'
args.emsize = 1500
args.nhid = 1500
args.nlayers = 2
args.qu_steps = 20
args.dropout = .65
args.tied = False
args.bptt = 35
args.cuda = False
args.use_durs = 'input'

talks = [
    'EricMead_2009P',
    'MichaelSpecter_2010',
    'JaneMcGonigal_2010',
    'TomWujec_2010U',
    'GaryFlake_2010',
    'RobertGupta_2010U',
    'DanBarber_2010',
    'JamesCameron_2010',
    'BillGates_2010',
    'DanielKahneman_2010',
    'AimeeMullins_2009P'
]

datafile = np.load(model_data_path, allow_pickle=True)
talknames = datafile['talknames']
vocab_phones = datafile['vocab_phones']