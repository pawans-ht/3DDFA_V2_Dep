# coding: utf-8

__author__ = 'cleardusk'

import os
import numpy as np
import torch
import pickle


def mkdir(d):
    os.makedirs(d, exist_ok=True)


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))


def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()


def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()


def _load_gpu(fp):
    return torch.from_numpy(_load(fp)).cuda()


_load_cpu = _load
def _numpy_to_tensor(x):
    return torch.from_numpy(x)
def _tensor_to_numpy(x):
    return x.numpy()
def _numpy_to_cuda(x):
    return _tensor_to_cuda(torch.from_numpy(x))
def _cuda_to_tensor(x):
    return x.cpu()
def _cuda_to_numpy(x):
    return x.cpu().numpy()
