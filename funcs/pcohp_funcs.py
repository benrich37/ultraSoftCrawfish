import sys
sys.path.append("..")

from helpers.data_parsers import parse_data
from helpers.pcohp_helpers import get_just_ipcohp_helper, get_cheap_pcohp_helper, get_pcohp_pieces
from helpers.misc_helpers import cs_formatter
import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti


def get_cheap_pcohp(idcs1, idcs2, path, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None, spin_pol = False):
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, path, data=data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    cs = cs_formatter(cs, spin_pol)
    return Erange, cs


def get_tetr_pcohp(idcs1, idcs2, path, data=None, res=0.01, orbs1=None, orbs2=None, Erange=None, spin_pol=False):
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, path, data=data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cs = []
    nSpin = np.shape(E_sabcj)[0]
    for s in range(nSpin):
        cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    tetr_pcohp = cs_formatter(cs, spin_pol)
    return Erange, tetr_pcohp


def get_ipcohp(idcs1, idcs2, path, data=None, orbs1=None, orbs2=None):
    if data is None:
        data = parse_data(root=path)
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, path, data=data, orbs1=orbs1, orbs2=orbs2)
    icohp = get_just_ipcohp_helper(occ_sabcj, weights_sabcj, wk)
    return icohp


