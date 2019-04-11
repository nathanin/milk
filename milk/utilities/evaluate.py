import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, 
  classification_report, confusion_matrix, f1_score,
  precision_score, recall_score)

def score2pred(y_score):
  y_pred = np.argmax(y_score, axis=-1)
  return y_pred

def acc_fn(y_true, y_score):
  y_pred = score2pred(y_score)
  acc = (y_true == y_pred).mean()
  return acc

# https://en.wikipedia.org/wiki/Precision_and_recall
def specificity(y_true, y_score):
  y_pred = score2pred(y_score)
  true_negative = np.sum((y_true == 0) * (y_pred == 0))
  condition_negative = np.sum(y_true == 0)
  return float(true_negative) / condition_negative

def precision_score_fn(y_true, y_score):
  y_pred = score2pred(y_score)
  return precision_score(y_true, y_pred)

def recall_score_fn(y_true, y_score):
  y_pred = score2pred(y_score)
  return recall_score(y_true, y_pred)

def f1_score_fn(y_true, y_score):
  y_pred = score2pred(y_score)
  return f1_score(y_true, y_pred)
  
metrics = {
  'accuracy': acc_fn,
  'precision': precision_score_fn,
  'recall': recall_score_fn,
  'specificity': specificity,
  'auc': roc_auc_score,
  'F1': f1_score_fn
}

def eval_suite(y_true, y_score):

  metric_values = {k: fn(y_true=y_true, y_score=y_score) for k, fn in metrics.items()}

  return metric_values