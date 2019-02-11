"""
Now using python to launch tests, using settings stored in args/*
"""
import re
import os
import glob
import argparse
import tensorflow as tf
from argparse import Namespace

#from test_svs import main as test_svs
from test_npy import main as test_npy

""" Get the base timestamps """
def list_runs(argdir, run_list=None):
    runs = []
    if run_list is not None:
        with open(run_list, 'r') as f:
            for l in f:
                l = l.replace('\n', '')
                if '/' in l:
                    l = os.path.splitext(os.path.basename(l))[0]
                runs.append(l)
        print('Using run list with {} experiments'.format(len(runs)))
        return runs

    argfiles = sorted(glob.glob(os.path.join(argdir, '*.txt')))
    print('Got {} runs'.format(len(argfiles)))
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

# TODO: unify svs and npy test modes
defaults = {
    'timestamp': None,
    'testdir': '../experiment/test_lists',
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
    'gated_attention': True,
    'deep_classifier': False,
    # Arguments for npy tests
    'x_size': 128,
    'y_size': 128,
    'crop_size': 96,
    'scale': 1.0,
    'batch_size': 64,
    'n_repeat': 1,
}
def translate_args(timestamp, run_args, args):
    """
    `run_args` are arguments picked up from a file
    `args` are arguments passed into this script
    """
    ns = Namespace(**defaults)
    ns.__dict__['timestamp'] = timestamp
    ns.__dict__['mcdropout'] = args.mcdropout
    ns.__dict__['mil'] = run_args.mil
    ns.__dict__['gated_attention'] = run_args.gated_attention
    ns.__dict__['odir'] = args.odir

    if 'deep_classifier' in run_args.__dict__.keys():
        ns.__dict__['deep_classifier'] = run_args.deep_classifier

    if args.val:
        ns.__dict__['testdir'] = '../experiment/val_lists'
    return ns

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def main(args):
    runs = list_runs(args.argdir, args.run_list)

    if args.test == 'svs':
        # Spin up a new session and run it
        sess = tf.Session(config=config)

    for timestamp in runs:
        print('\n', timestamp)
        argfile = os.path.join(args.argdir, '{}.txt'.format(timestamp))
        run_args = parse_args(argfile)
        run_args = translate_args(timestamp, run_args, args)

        print('Running test with arguments:')
        for v in vars(run_args):
            print('\t', v, '\t', getattr(run_args, v))

        # translate into the expected namespace
        if args.test == 'svs':
            # Spin up a new session and run it
            # sess = tf.Session(config=config)
            test_svs(run_args, sess)
        if args.test == 'npy':
            test_npy(run_args)

    if args.test == 'svs':
        sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', default=False, action='store_true')
    parser.add_argument('--odir', default='result', type=str)
    parser.add_argument('--test', default='npy', type=str)
    parser.add_argument('--argdir', default='args', type=str)
    parser.add_argument('--mcdropout', default=False, action='store_true')

    ## should point to a text file
    parser.add_argument('--run_list', default=None, type=str)

    args = parser.parse_args()
    main(args)
