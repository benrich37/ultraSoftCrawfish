import sys
sys.path.append("..")

from helpers.data_parsers import parse_data
from helpers.pcohp_helpers import get_just_ipcohp_helper, get_cheap_pcohp_helper, get_pcohp_pieces
import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti


def get_cheap_pcohp(idcs1, idcs2, path, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None):
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, path, data=data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    if len(cs) == 2:
        return Erange, cs[0] + cs[1]
    elif len(cs) == 1:
        return Erange, cs[0]
    else:
        raise ValueError("Unexpected shape of output from get_cheap_pcohp_helper")


def get_tetr_pcohp(idcs1, idcs2, path, data=None, res=0.01, orbs1=None, orbs2=None, Erange=None):
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, path, data=data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    tetr_pcohp = lti(atoms.cell, E_sabcj[0], Erange, weights=weights_sabcj[0])
    if np.shape(E_sabcj)[0] > 1:
        tetr_pcohp += lti(atoms.cell, E_sabcj[1], Erange, weights=weights_sabcj[1])
    return Erange, tetr_pcohp


def get_ipcohp(idcs1, idcs2, path, data=None, orbs1=None, orbs2=None):
    if data is None:
        data = parse_data(root=path)
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, path, data=data, orbs1=orbs1, orbs2=orbs2)
    icohp = get_just_ipcohp_helper(occ_sabcj, weights_sabcj, wk)
    return icohp


