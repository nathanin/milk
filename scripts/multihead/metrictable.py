#!/usr/bin/env python

import numpy as np
import pandas as pd

import os 
import sys
import argparse

def score2decision(score, threshold=0.5):
    return score > threshold

def TP_fn(ytrue, ypred):
    total_pos = ytrue.sum()
    tp = (ytrue & ypred).sum()
    tp_rate = tp / float(total_pos)
    return tp_rate

def TN_fn(ytrue, ypred):
    ytrue_inv = np.logical_not(ytrue) # known negatives
    ypred_inv = np.logical_not(ypred) # called negatives
    return TP_fn(ytrue_inv, ypred_inv)

def FP_fn(ytrue, ypred):
    ytrue_inv = np.logical_not(ytrue) # known negatives
    false_pos = (ytrue_inv & ypred).sum()
    true_neg = (ytrue == 0).sum()
    return false_pos / float(true_neg)

""" portion of called negatives that should be positive """
def FN_fn(ytrue, ypred):
    ypred_inv = np.logical_not(ypred) # called negatives
    false_neg = (ypred_inv & ytrue).sum()
    true_pos = ytrue.sum()
    return false_neg / float(true_pos)

def accuracy(ytrue, ypred):
    return (ytrue == ypred).mean()

def main(args, fout):

    df = pd.read_csv(args.fin)

    ytrue = df.loc[:, 'ytrue']
    ytrue = score2decision(ytrue)
    df.drop('ytrue', axis=1, inplace=True)
    print(df.head())

    ## Take individual heads
    for c in df.columns:
        ypred = score2decision(df.loc[:, c])
        acc = accuracy(ytrue, ypred)
        tp = TP_fn(ytrue, ypred)
        tn = TN_fn(ytrue, ypred)
        fp = FP_fn(ytrue, ypred)
        fn = FN_fn(ytrue, ypred)

        fout.write('{},{},{},{},{},{}\n'.format(
            c, acc, tp, tn, fp, fn))

    ## Average the heads
    avgpred = np.mean(df.values, axis=1)
    ypred = score2decision(avgpred)
    acc = accuracy(ytrue, ypred)
    tp = TP_fn(ytrue, ypred)
    tn = TN_fn(ytrue, ypred)
    fp = FP_fn(ytrue, ypred)
    fn = FN_fn(ytrue, ypred)
    fout.write('{},{},{},{},{},{}\n'.format(
        'all', acc, tp, tn, fp, fn))

    ## Confidence interval with multiple predictors
    sample_ci = []
    for s in range(df.shape[0]):
        ci = np.quantile(df.loc[s, :], [0.1, 0.9])
        # Contains 0.5; reject
        if ci[0] < 0.5 and ci[1] > 0.5:
            sample_ci.append(-1)
        # Confidently 0
        elif ci[0] < 0.5 and ci[1] < 0.5:
            sample_ci.append(0)
        # Confidently 1
        elif ci[0] > 0.5 and ci[1] > 0.5:
            sample_ci.append(1)
    sample_ci = np.array(sample_ci)
    no_decision = sample_ci == -1
    decision = sample_ci > -1
    ytrue_ci = ytrue[decision]
    ypred_ci = sample_ci[decision]

    if no_decision.sum() == len(sample_ci):
        fout.write('{}/{},{},{},{},{},{}\n'.format(
            no_decision.sum(), df.shape[0], 0, 0, 0, 0, 0))
        return

    ypred_ci = score2decision(ypred_ci)
    acc = accuracy(ytrue_ci, ypred_ci)
    tp = TP_fn(ytrue_ci, ypred_ci)
    tn = TN_fn(ytrue_ci, ypred_ci)
    fp = FP_fn(ytrue_ci, ypred_ci)
    fn = FN_fn(ytrue_ci, ypred_ci)
    fout.write('{}/{},{},{},{},{},{}\n'.format(
        no_decision.sum(), df.shape[0], acc, tp, tn, fp, fn))
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('fin')

    args = p.parse_args()
    assert os.path.exists(args.fin)
    outp = args.fin + 'summary.txt'

    with open(outp, 'w+') as fout:
        fout.write('head,accuracy,TP,TN,FP,FN\n')
        main(args, fout)
    
    print('Summary --> {}'.format(outp))