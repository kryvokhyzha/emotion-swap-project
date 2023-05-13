import random
import numpy as np
import torch
import warnings
import torch.nn.functional as functional
from sklearn.metrics import f1_score


def calculate_f1(preds, labels, average='micro'):
    return f1_score(labels, preds, average=average)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return (
            0.5 * functional.kl_div(p, m, reduction='batchmean', log_target=False)
            +
            0.5 * functional.kl_div(q, m, reduction='batchmean', log_target=False)
    )


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_warnings():
    warnings.filterwarnings("ignore")
