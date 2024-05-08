import sys
sys.path.append("..")
import numpy as np

from helpers.pdos_helpers import get_pdos_pieces
from helpers.pcohp_helpers import get_cheap_pcohp_helper
from helpers.misc_helpers import cs_formatter
from helpers.ase_helpers import get_atoms
from ase.dft.dos import linear_tetrahedron_integration as lti


def get_cheap_pdos(idcs, path, data=None, res=0.01, sig=0.00001, orbs=None, Erange=None, spin_pol=False):
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, path, data, res, orbs, Erange)
    cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    pdos = cs_formatter(cs, spin_pol)
    return Erange, pdos


def get_tetr_pdos(idcs, path, data=None, res=0.01, orbs=None, Erange=None, spin_pol=False):
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, path, data, res, orbs, Erange)
    atoms = get_atoms(path)
    cs = []
    nSpin = np.shape(E_sabcj)[0]
    for s in range(nSpin):
        cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    tetr_pdos = cs_formatter(cs, spin_pol)
    return Erange, tetr_pdos