import sys
sys.path.append("../..")
import numpy as np

from ultraSoftCrawfish.helpers.pdos_helpers import get_pdos_pieces
from ultraSoftCrawfish.helpers.ElecData import  get_data_and_path
from ultraSoftCrawfish.helpers.pcohp_helpers import get_cheap_pcohp_helper, get_cheap_dos_helper
from ultraSoftCrawfish.helpers.misc_helpers import cs_formatter
from ultraSoftCrawfish.helpers.rs_helpers import get_target_kjs_dict, get_rs_wfn, write_cube_writer
from ase.dft.dos import linear_tetrahedron_integration as lti
from os.path import join as opj


def get_cheap_pdos(idcs, path=None, data=None, res=0.01, sig=0.00001, orbs=None, Erange=None, spin_pol=False):
    """
    :param idcs: list[int]
        List of atom indices to plot pDOS of (0-based indices)
    :param path: str | path
        Full path for directory containing output files from calculation
    :param data: ElecData
        ElecData class object for calculation
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
    data, path = get_data_and_path(data, path)
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, data, res, orbs, Erange)
    cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    pdos = cs_formatter(cs, spin_pol)
    return Erange, pdos


def get_tetr_pdos(idcs, path=None, data=None, res=0.01, orbs=None, Erange=None, spin_pol=False):
    """
    :param idcs: list[int]
        List of atom indices to plot pDOS of (0-based indices)
    :param path: str or path
        Full path for directory containing output files from calculation
    :param data: ElecData
        ElecData class object for calculation
    :param res: float
        dE for evenly spaced energy array (in Hartree)
    :param orbs: list[str]
        List orbitals to include in pDOS (includes all if None)
            - ie orbs = ["s"] would include only s orbitals,
                orbs = ["d"] would include only d orbitals,
                orbs = ["px"] would include only px orbitals
    :param Erange: np.ndarray[float] of shape (,N)
        Array of energy values to evaluate pDOS for (Hartree)
    :param sp
        If true, pDOS return array returned as (2,N) array, where first index belongs to spin
    :return Erange, pdos:
        :Erange: np.ndarray[float] of shape (,N)
        :pdos:
            if spin_pol:
                np.ndarray[float] of shape (2,N)
            else:
                np.ndarray[float] of shape (,N)
    """
    data, path = get_data_and_path(data, path)
    if not data.lti_allowed:
        raise ValueError("Inconsistency encountered in number of expected kpts, spins, and found states. " + \
                         "Due to uncertainty of kfolding, linear tetrahedral integration cannot be used.")
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, data, res, orbs, Erange)
    atoms = data.get_atoms()
    cs = []
    nSpin = np.shape(E_sabcj)[0]
    for s in range(nSpin):
        cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    tetr_pdos = cs_formatter(cs, spin_pol)
    return Erange, tetr_pdos

def get_cheap_dos(path=None, data=None, res=0.01, sig=0.00001, Erange=None, spin_pol=False):
    data, path = get_data_and_path(data, path)
    E_sabcj = data.get_E_sabcj()
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj) - (10 * res), np.max(E_sabcj) + (10 * res), res)
    cs = get_cheap_dos_helper(Erange, E_sabcj, sig=sig)
    dos = cs_formatter(cs, spin_pol)
    return Erange, dos


def write_dos_cub(path=None, data=None, Ebounds=None, cubename=None):
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" + \
                         "to your JDFTx in file.")
    nStates = data.get_nStates()
    nBands = data.get_nBands()
    wk = data.get_wk_sabc()
    weights_sabcj = np.array([wk]*nBands)
    weights_kj = weights_sabcj.reshape([nStates, nBands])
    E_kj = data.get_E_sabcj().reshape([nStates, nBands])
    target_kjs_dict = get_target_kjs_dict(E_kj, Ebounds=Ebounds, weights_kj=weights_kj)
    rs_wfn = get_rs_wfn(path, weights=weights_kj, target_kjs_dict=target_kjs_dict)
    if cubename is None:
        cubename = f"DOS"
        if not Ebounds is None:
            cubename += f"-({'_'.join([str(b) for b in Ebounds])})"
    if ".cub" in cubename:
        cubename = cubename.split(".")[0]
    fname = opj(path, f"{cubename}.cub")
    write_cube_writer(data.get_atoms(), fname, rs_wfn, f"DOS {cubename}")


def write_pdos_cub(idcs, path=None, data=None, orbs=None, cubename=None, Erange=None, Ebounds=None, res=0.01, ):
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" + \
                         "to your JDFTx in file.")
    nStates = data.get_nStates()
    nBands = data.get_nBands()
    wk = data.get_wk_sabc()
    Erange, weights_sabcj, E_sabcj = get_pdos_pieces(idcs, data, res, orbs, Erange)
    weights_kj = weights_sabcj.reshape([nStates, nBands])
    E_kj = data.get_E_sabcj().reshape([nStates, nBands])
    target_kjs_dict = get_target_kjs_dict(E_kj, Ebounds=Ebounds, weights_kj=weights_kj)
    rs_wfn = get_rs_wfn(path, weights=weights_kj, target_kjs_dict=target_kjs_dict)
    if cubename is None:
        cubename = f"pDOS-({'_'.join([str(idx) for idx in idcs])})"
        if not orbs is None:
            if type(orbs) is str:
                cubename += f"-({orbs})"
            else:
                cubename += f"-({'_'.join([str(o) for o in orbs])})"
        if not Ebounds is None:
            cubename += f"-({'_'.join([str(b) for b in Ebounds])})"
    if ".cub" in cubename:
        cubename = cubename.split(".")[0]
    fname = opj(path, f"{cubename}.cub")
    write_cube_writer(data.get_atoms(), fname, rs_wfn, f"pDOS {cubename}")