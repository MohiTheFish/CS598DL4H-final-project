import torch
import numpy as np

def padMatrixWithoutTime(data, max_codes):
    lengths = np.array([len(d) for d in data]).astype('int32')
    n_samples = len(data)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, max_codes))
    for idx, d in enumerate(data):
        for xvec, subd in zip(x[:, idx, :], d):
            xvec[subd] = 1.

    return x

def yToTensor(y, params):
    y = torch.from_numpy(np.array(y)).long().to(params["device"])
    return y

def xToTensor(x, embedding_dim, params):
    x = padMatrixWithoutTime(x, embedding_dim)
    x = torch.from_numpy(x).float().to(params["device"])
    return x

