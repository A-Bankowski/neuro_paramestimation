"""
Parameter Estimation of the neurotransmission model
"""
import numpy as np
from utils import *

ts = np.load('/data/numerik/people/abankowski/neuro_param_estimation/codes/ts.npy')
times = ts
np.save('/data/numerik/people/abankowski/neuro_param_estimation/codes/times.npy',times)

