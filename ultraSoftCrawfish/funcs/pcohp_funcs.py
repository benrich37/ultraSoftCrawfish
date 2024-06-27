import sys
sys.path.append("../..")

from ultraSoftCrawfish.helpers.ElecData import get_data_and_path
from ultraSoftCrawfish.helpers.pcohp_helpers import get_just_ipcohp_helper, get_cheap_pcohp_helper, get_pcohp_pieces
from ultraSoftCrawfish.helpers.misc_helpers import cs_formatter
import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti


def get_cheap_pcohp(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None, spin_pol = False):
    """
    :param idcs1: list[int]
        List of atom indices to belong to first group of the pCOHP pair (0-based indices)
    :param idcs2: list[int]
        List of atom indices to belong to second group of the pCOHP pair (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: ElecData
        ElecData class object for calculation
    :param res: float
        dE for evenly spaced energy array (in Hartree)
    :param sig: float
        Controls smearing amplitude of gaussian smearings
    :param orbs1/2: list[str]
        List orbitals to include in pCOHP evaluation (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :param Erange: np.ndarray[float] of shape (,N)
        Array of energy values to evaluate pDOS for (Hartree)
    :param spin_pol: bool
        If true, pCOHP return array returned as (2,N) array, where first index belongs to spin
    :return Erange, pcohp:
        :Erange: np.ndarray[float] of shape (,N)
        :pcohp:
            if spin_pol:
                np.ndarray[float] of shape (2,N)
            else:
                np.ndarray[float] of shape (,N)
    """
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    pcohp = cs_formatter(cs, spin_pol)
    return Erange, pcohp


def get_tetr_pcohp(idcs1, idcs2, path=None, data=None, res=0.01, orbs1=None, orbs2=None, Erange=None, spin_pol=False):
    """
    :param idcs1: list[int]
        List of atom indices to belong to first group of the pCOHP pair (0-based indices)
    :param idcs2: list[int]
        List of atom indices to belong to second group of the pCOHP pair (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: ElecData
        ElecData class object for calculation
    :param res: float
        dE for evenly spaced energy array (in Hartree)
    :param orbs1/2: list[str]
        List orbitals to include in pCOHP evaluation (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :param Erange: np.ndarray[float] of shape (,N)
        Array of energy values to evaluate pDOS for (Hartree)
    :param spin_pol: bool
        If true, pCOHP return array returned as (2,N) array, where first index belongs to spin
    :return Erange, pcohp:
        :Erange: np.ndarray[float] of shape (,N)
        :pcohp:
            if spin_pol:
                np.ndarray[float] of shape (2,N)
            else:
                np.ndarray[float] of shape (,N)
    """
    data, path = get_data_and_path(data, path)
    if not data.lti_allowed:
        raise ValueError("Inconsistency encountered in number of expected kpts, spins, and found states. " + \
                         "Due to uncertainty of kfolding, linear tetrahedral integration cannot be used.")
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
                         "to your JDFTx in file.")

    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cs = []
    nSpin = np.shape(E_sabcj)[0]
    for s in range(nSpin):
        cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    tetr_pcohp = cs_formatter(cs, spin_pol)
    return Erange, tetr_pcohp


def get_ipcohp(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None):
    """
    :param idcs1: list[int]
        List of atom indices to belong to first group of the pCOHP pair (0-based indices)
    :param idcs2: list[int]
        List of atom indices to belong to second group of the pCOHP pair (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: ElecData
        ElecData class object for calculation
    :param orbs1/2: list[str]
        List orbitals to include in pCOHP evaluation (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :return ipcohp: float
        Integrated pCOHP up to fermi level
    """
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2)
    ipcohp = get_just_ipcohp_helper(occ_sabcj, weights_sabcj, wk)
    return ipcohp


def get_ipcohp_array(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, use_occs=True):
    """ Return
    :param idcs1: list[int]
        List of atom indices to belong to first group of the pCOHP pair (0-based indices)
    :param idcs2: list[int]
        List of atom indices to belong to second group of the pCOHP pair (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: ElecData
        ElecData class object for calculation
    :param orbs1/2: list[str]
        List orbitals to include in pCOHP evaluation (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :param use_occs: bool
        If True, pCOHP will be integrated with occupation values
    :return E: np.ndarray
        Ordered non-uniform energy array running parallel to ipcohp array
    :return ipcohp: np.ndarray
        Array of integrated pCOHP
    """
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2)
    wk_sabcj = np.array([wk_sabc]*data.get_nProj())
    E_flat = E_sabcj.flatten()
    idcs = np.argsort(E_flat)
    E = E_flat[idcs]
    pW = weights_sabcj.flatten()[idcs]
    kW = wk_sabcj.flatten()[idcs]
    occ = occ_sabcj.flatten()[idcs]
    if use_occs:
        pcohp = pW*kW*occ
    else:
        pcohp = pW*kW
    ipcohp = np.zeros(len(pcohp)+1)
    for i in range(len(E)):
        ipcohp[i+1] = pcohp[i]+ipcohp[i]
    return E, ipcohp[1:]


