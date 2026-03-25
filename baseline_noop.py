import numpy as np

def revise_state(s):
    s = np.asarray(s, dtype=float)
    return s

def intrinsic_reward(updated_s):
    return 0.0
