"""
Now using python to launch tests, using settings stored in args/*
"""
import os
import glob
import argparse
import tensorflow as tf
from argparse import Namespace

from test_svs import main as test_svs
from test_npy import main as test_npy

""" Get the base timestamps """
def list_runs(argdir):
    argfiles = sorted(glob.glob(os.path.join(argdir, '*.txt')))
    print('Got {} runs'.format(len(argfiles)))
    runs = []
    for a in argfiles:
        r = os.path.basename(a).replace('.txt', '')
        print('\t', r)
        runs.append(r)

    return runs

""" Grab a dictionary from the tab-delineated text """
def parse_args(argfile):
    d = {}
    with open(argfile, 'r') as f:
        for l in f:
            l = l.replace('\n', '')
            k, v = l.split('\t')
            # Check if it's a boolean:
            if v in ['True', 'False']:
                d[k] = bool(v)
                continue
            
            # Check if it's a None:
            if v == 'None':
                d[k] = None
                continue

            try:
                d[k] = float(v)
            except:
                d[k] = v
    n = Namespace(**d)
    return n

defaults = {
    'timestamp': None,
    'testdir': 'test_lists',
    'savedir': 'save',
    'n_classes': 2,
    'input_dim': 96,
    'mag': 5,
    'batch_size': 64,
    'ramdisk': '/dev/shm',
    'odir': 'result',
    'fgdir': '../usable_area/inference',
    'mcdropout': False,
    'mcdropout_t': 25,
    'mcdropout_sample': 0.25,
    'mil': 'attention',
    'gated_attention': True
}
def translate_args(timestamp, run_args, args):
    ns = Namespace(**defaults)
    ns.__dict__['timestamp'] = timestamp
    ns.__dict__['mcdropout'] = args.mcdropout
    ns.__dict__['mil'] = run_args.mil
    ns.__dict__['gated_attention'] = run_args.gated_attention
    return ns

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def main(args):
    runs = list_runs(args.argdir)
    for timestamp in runs:
        print('\n', timestamp)
        argfile = os.path.join(args.argdir, '{}.txt'.format(timestamp))
        run_args = parse_args(argfile)
        run_args = translate_args(timestamp, run_args, args)

        # print(run_args)
        for v in vars(run_args):
            print('\t', v, '\t', getattr(run_args, v))

        # translate into the expected namespace
        # Spin up a new session and run it
        sess = tf.Session(config=config)
        if args.test == 'svs':
            test_svs(run_args, sess)
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--argdir', default='args', type=str)
    parser.add_argument('--test', default='svs', type=str)
    parser.add_argument('--mcdropout', default=False, type=bool)
    args = parser.parse_args()
    main(args)