import random
import numpy
import torch
import collections
import os

def seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
