import sys
sys.path.append("..")
import numpy as np

from helpers.pdos_helpers import get_pdos_pieces
from helpers.pcohp_helpers import get_cheap_pcohp_helper
from helpers.misc_helpers import cs_formatter
from helpers.ase_helpers import get_atoms
from ase.dft.dos import linear_tetrahedron_integration as lti


def get_cheap_pdos(idcs, path, data=None, res=0.01, sig=0.00001, orbs=None, Erange=None, spin_pol=False):
    """
    :param idcs: list[int]
        List of atom indices to plot pDOS of (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: tuple
        Output dictionary from 'parse_data' function found in helpers.data_parsers
    :param res: float
        dE for evenly spaced energy array (in Hartree)
    :param sig: float
        Controls smearing amplitude of gaussian smearings
    :param orbs: list[str]
        List orbitals to include in pDOS (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :param Erange: np.ndarray[float] of shape (,N)
        Array of energy values to evaluate pDOS for (Hartree)
    :param spin_pol: bool
        If true, pDOS return array returned as (2,N) array, where first index belongs to spin
    :return Erange, pdos:
        :Erange: np.ndarray[float] of shape (,N)
        :pdos:
            if spin_pol:
                np.ndarray[float] of shape (2,N)
            else:
                np.ndarray[float] of shape (,N)
    """
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, path, data, res, orbs, Erange)
    cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    pdos = cs_formatter(cs, spin_pol)
    return Erange, pdos


def get_tetr_pdos(idcs, path, data=None, res=0.01, orbs=None, Erange=None, spin_pol=False):
    """
    :param idcs: list[int]
        List of atom indices to plot pDOS of (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: tuple
        Output dictionary from 'parse_data' function found in helpers.data_parsers
    :param res: float
        dE for evenly spaced energy array (in Hartree)
    :param orbs: list[str]
        List orbitals to include in pDOS (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :param Erange: np.ndarray[float] of shape (,N)
        Array of energy values to evaluate pDOS for (Hartree)
    :param spin_pol: bool
        If true, pDOS return array returned as (2,N) array, where first index belongs to spin
    :return Erange, pdos:
        :Erange: np.ndarray[float] of shape (,N)
        :pdos:
            if spin_pol:
                np.ndarray[float] of shape (2,N)
            else:
                np.ndarray[float] of shape (,N)
    """
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, path, data, res, orbs, Erange)
    atoms = get_atoms(path)
    cs = []
    nSpin = np.shape(E_sabcj)[0]
    for s in range(nSpin):
        cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    tetr_pdos = cs_formatter(cs, spin_pol)
    return Erange, tetr_pdos