import numpy as np
from numba import jit


@jit(nopython=True)
def gauss(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / sig)

def cs_formatter(cs, spin_pol):
    """
    :param cs list(np.ndarray): List of dos/pdos/pcohp arrays of length N (parallel to Erange of length N)
    :param spin_pol: Return as (up, down) or (tot)
    :return cs np.ndarray: Numpy array of shape (2,N) if spin_pol, or shape (,N) if not spin_pol
    """
    if len(cs) > 2:
        raise ValueError(f"Unexpected numbers of spin ({len(cs)} found, only 1-2 supported)")
    if spin_pol:
        if len(cs) == 1:
            raise ValueError("Spin-polarized output not supported for spin-paired output")
        else:
            return np.array([cs[0], cs[1]])
    else:
        if len(cs) == 1:
            return cs[0]
        else:
            return cs[0] + cs[1]


def fidcs(idcs):
    if type(idcs) is int:
        return [idcs]
    elif type(idcs) is list:
        return idcs
    else:
        raise ValueError("atom indices must be int or list of int")


def get_orb_bool_func(orbs):
    orb_bool_func = None
    if not orbs is None:
        if type(orbs) is list:
            orb_bool_func = lambda s: True in [o in s for o in orbs]
        else:
            orb_bool_func = lambda s: orbs in s
    return orb_bool_func



