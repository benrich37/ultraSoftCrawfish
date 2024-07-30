import sys
sys.path.append("../..")

from ultraSoftCrawfish.helpers.ElecData import get_data_and_path
from ultraSoftCrawfish.helpers.pcohp_helpers import get_just_ipcohp_helper, get_cheap_pcohp_helper, get_pcohp_pieces, get_pcoop_pieces
from ultraSoftCrawfish.helpers.misc_helpers import cs_formatter, fidcs
import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti
from ultraSoftCrawfish.helpers.rs_helpers import get_rs_wfn, write_cube_writer, get_target_kjs_dict
from os.path import join as opj
from copy import deepcopy


def get_pcohp(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None,
              spin_pol = False, tetr = False,
              directional=False, iso_acceptance=False, iso_donation=False):
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
        :param tetr: bool
            If true, pCOHP will be evaluated on Erange through Linear Tetrahedral Integration.
            If false, pCOHP will be evaluated on Erange as sum of gaussians with width controlled by sig
        :param directional: bool
            If true, contributions from the first index set will be isolated
        :param iso_acceptance: bool
            If true, contributions with polarity pointed towards first index set will be isolated
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
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" + \
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange,
                                                                            directional=directional,
                                                                            a=iso_acceptance, d=iso_donation)
    if not tetr:
        cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    else:
        cs = []
        nSpin = np.shape(E_sabcj)[0]
        for s in range(nSpin):
            cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    pcohp = cs_formatter(cs, spin_pol)
    return Erange, pcohp


def get_cheap_pcohp(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None, spin_pol = False):
    print("get_cheap_pcohp deprecated - please use get_pcohp with tetr=False instead")
    return get_pcohp(idcs1, idcs2, path=path, data=data, res=res, sig=sig, orbs1=orbs1, orbs2=orbs2, Erange=Erange, spin_pol = spin_pol, tetr=False)


def get_tetr_pcohp(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None, spin_pol = False):
    print("get_tetr_pcohp deprecated - please use get_pcohp with tetr=True instead")
    return get_pcohp(idcs1, idcs2, path=path, data=data, res=res, sig=sig, orbs1=orbs1, orbs2=orbs2, Erange=Erange, spin_pol = spin_pol, tetr=True)






def get_ipcohp_array(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, ebounds=None,
                     weights_sabcj=None, E_sabcj=None, wk=None, occ_sabcj=None,
                     use_occs=True):
    print("get_ipcohp_array deprecated - please use get_ipcohp with as_array=True")
    return get_ipcohp(idcs1, idcs2, path=path, data=data, orbs1=orbs1, orbs2=orbs2, ebounds=ebounds,
               weights_sabcj=weights_sabcj, E_sabcj=E_sabcj, wk=wk, occ_sabcj=occ_sabcj, as_array=True, use_occs=use_occs)


def write_pcohp_cub(idcs1, idcs2, path=None, data=None, res=0.01, orbs1=None, orbs2=None, Ebounds=None, cubename=None):
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" + \
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, res=res, orbs1=orbs1, orbs2=orbs2)
    nStates = np.prod(np.shape(wk))
    nBands = np.shape(weights_sabcj)[-1]
    weights_kj = weights_sabcj.reshape([nStates, nBands])
    E_kj = E_sabcj.reshape([nStates, nBands])
    target_kjs_dict = get_target_kjs_dict(E_kj, Ebounds=Ebounds, weights_kj=weights_kj)
    rs_wfn = get_rs_wfn(path, weights=weights_kj, target_kjs_dict=target_kjs_dict)
    if cubename is None:
        cubename = f"{'_'.join([str(i) for i in idcs1])}"
        if not orbs1 is None:
            cubename += f"({'_'.join(orbs1)})"
        cubename += f"-{'_'.join([str(i) for i in idcs2])}"
        if not orbs2 is None:
            cubename += f"({'_'.join(orbs2)})"
        if not Ebounds is None:
            cubename += f"-({'_'.join([str(b) for b in Ebounds])})"
    if ".cub" in cubename:
        cubename = cubename.split(".")[0]
    fname = opj(path, f"{cubename}.cub")
    write_cube_writer(data.get_atoms(), fname, rs_wfn, f"pCOHP {cubename}")


def get_pcoop(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None,
              spin_pol = False, tetr = False,
              directional=False, iso_acceptance=False, iso_donation=False):
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
        :param Erange: np.ndarray[floao] of shape (,N)
            Array of energy values to evaluate pDOS for (Hartree)
        :param spin_pol: bool
            If true, pCOOP return array returned as (2,N) array, where first index belongs to spin
        :param tetr: bool
            If true, pCOOP will be evaluated on Erange through Linear Tetrahedral Integration.
            If false, pCOOP will be evaluated on Erange as sum of gaussians with width controlled by sig
        :param directional: bool
            If true, contributions from the first index set will be isolated
        :param iso_acceptance: bool
            If true, contributions with polarity pointed towards first index set will be isolated
        :return Erange, pcoop:
            :Erange: np.ndarray[float] of shape (,N)
            :pcoop:
                if spin_pol:
                    np.ndarray[float] of shape (2,N)
                else:
                    np.ndarray[float] of shape (,N)
        """
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" + \
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcoop_pieces(idcs1, idcs2, data, res=res,
                                                                            orbs1=orbs1, orbs2=orbs2, Erange=Erange,
                                                                            directional=directional,
                                                                            a=iso_acceptance, d=iso_donation)
    if not tetr:
        cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    else:
        cs = []
        nSpin = np.shape(E_sabcj)[0]
        for s in range(nSpin):
            cs.append(lti(atoms.cell, E_sabcj[s], Erange, weights=weights_sabcj[s]))
    pcohp = cs_formatter(cs, spin_pol)
    return Erange, pcohp

def get_ipcohp(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, ebounds=None,
               weights_sabcj=None, E_sabcj=None, wk=None, occ_sabcj=None,
               directional=False, iso_acceptance=False, as_array=False, use_occs=True, iso_donation=False):
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
    :param ebounds: list[num]
        List of energies such that ipCOHP will only be evaluated within ebounds[2*i], ebounds[1+2*i]
        (must be of len 2N)
    :param directional: bool
        If true, contributions from the first index set will be isolated
    :param iso_acceptance: bool
        If true, contributions with polarity pointed towards first index set will be isolated
    :param as_array: bool:
        If true,
            :return Es, ipcohps: array(float), array(float)
                Es: Uneven energy array parallel to ipcohps
                ipcohp: pCOHP integrated up to E[i] at ipcohp[i]
        If false,
            :return ipcohp: float
                Integrated pCOHP up to fermi level
    """
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
                         "to your JDFTx in file.")
    reval = False
    for arr in [occ_sabcj, weights_sabcj, wk, E_sabcj]:
        if arr is None:
            reval = True
            break
    if reval:
        Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2, directional=directional, a=iso_acceptance, d=iso_donation)
    if not as_array:
        ipcohp = get_just_ipcohp_helper(occ_sabcj, weights_sabcj, wk, ebounds=ebounds, E_sabcj=E_sabcj)
        return ipcohp
    else:
        wk_sabcj = np.array([wk] * data.get_nBands())
        wk_sabcj = wk_sabcj.reshape(np.shape(E_sabcj))
        E_flat = E_sabcj.flatten()
        idcs = np.argsort(E_flat)
        E = E_flat[idcs]
        pW = weights_sabcj.flatten()[idcs]
        kW = wk_sabcj.flatten()[idcs]
        if use_occs:
            occ = occ_sabcj.flatten()[idcs]
            pcohp = pW * kW * occ
        else:
            pcohp = pW * kW
        ipcohp = np.zeros(len(pcohp) + 1)
        ipcohp[1:] = np.cumsum(pcohp)
        return E, ipcohp[1:]


def get_ipcoop(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, use_occs=True, as_array=False,
               ebounds=None, directional=False, iso_acceptance=False, iso_donation=False):
    data, path = get_data_and_path(data, path)
    if not data.complex_bandprojs:
        raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
                         "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
                         "to your JDFTx in file.")
    Erange, weights_sabcj, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcoop_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2, directional=directional, a=iso_acceptance, d=iso_donation)
    if not as_array:
        return get_just_ipcohp_helper(occ_sabcj, weights_sabcj, wk_sabc, ebounds=ebounds, E_sabcj=E_sabcj)
    else:
        wk_sabcj = np.array([wk_sabc]*data.get_nBands())
        wk_sabcj = wk_sabcj.reshape(np.shape(E_sabcj))
        E_flat = E_sabcj.flatten()
        idcs = np.argsort(E_flat)
        E = E_flat[idcs]
        pW = weights_sabcj.flatten()[idcs]
        kW = wk_sabcj.flatten()[idcs]
        if use_occs:
            occ = occ_sabcj.flatten()[idcs]
            pcoop = pW*kW*occ
        else:
            pcoop = pW*kW
        ipcoop = np.zeros(len(pcoop)+1)
        ipcoop[1:] = np.cumsum(pcoop)
        return E, ipcoop[1:]



# def get_directional_pcohp(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None, spin_pol = False):
#     """
#     Same functionality as get_cheap_pcohp, but pCOHP_uv = P_uv*H_uv will instead be evaluated as pCOHP_uv = P_uu*H_uv with
#     a projection tensor P renormalized such that P_uv = P_uu/(P_uu+P_vv)
#     """
#     data, path = get_data_and_path(data, path)
#     if not data.complex_bandprojs:
#         raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
#                          "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
#                          "to your JDFTx in file.")
#     Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, res=res,
#                                                                             orbs1=orbs1, orbs2=orbs2, Erange=Erange,
#                                                                             directional=True)
#     cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
#     pcohp = cs_formatter(cs, spin_pol)
#     return Erange, pcohp
#
#
# def get_dipcohp_array(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, use_occs=True):
#     """ Same as get_ipcohp_array but only evaluates the bonding contributions of dpcohp(i1, i2) and antibonding of dpcohp(i2,i1)
#     """
#     data, path = get_data_and_path(data, path)
#     if not data.complex_bandprojs:
#         raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
#                          "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
#                          "to your JDFTx in file.")
#     Erange, weights_sabcj_1, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcohp_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2, directional=True)
#     Erange, weights_sabcj_2, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcohp_pieces(idcs2, idcs1, data, orbs1=orbs2,
#                                                                                    orbs2=orbs1, directional=True)
#     wk_sabcj = np.array([wk_sabc]*data.get_nBands())
#     wk_sabcj = wk_sabcj.reshape(np.shape(E_sabcj))
#     E_flat = E_sabcj.flatten()
#     idcs = np.argsort(E_flat)
#     E = E_flat[idcs]
#     pW_1 = np.minimum(weights_sabcj_1.flatten()[idcs], np.zeros(len(E)))
#     pW_2 = np.maximum(weights_sabcj_2.flatten()[idcs], np.zeros(len(E)))
#     pW = pW_1 + pW_2
#     kW = wk_sabcj.flatten()[idcs]
#     occ = occ_sabcj.flatten()[idcs]
#     if use_occs:
#         pcohp = pW*kW*occ
#     else:
#         pcohp = pW*kW
#     ipcohp = np.zeros(len(pcohp)+1)
#     ipcohp[1:] = np.cumsum(pcohp)
#     return E, ipcohp[1:]
#
#
# def get_ipcoop_array(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, use_occs=True):
#     data, path = get_data_and_path(data, path)
#     if not data.complex_bandprojs:
#         raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
#                          "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
#                          "to your JDFTx in file.")
#     Erange, weights_sabcj_1, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcoop_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2, directional=False)
#     wk_sabcj = np.array([wk_sabc]*data.get_nBands())
#     wk_sabcj = wk_sabcj.reshape(np.shape(E_sabcj))
#     E_flat = E_sabcj.flatten()
#     idcs = np.argsort(E_flat)
#     E = E_flat[idcs]
#     pW = weights_sabcj_1.flatten()[idcs]
#     kW = wk_sabcj.flatten()[idcs]
#     occ = occ_sabcj.flatten()[idcs]
#     if use_occs:
#         pcoop = pW*kW*occ
#     else:
#         pcoop = pW*kW
#     ipcohp = np.zeros(len(pcoop)+1)
#     ipcohp[1:] = np.cumsum(pcoop)
#     return E, ipcohp[1:]
#
#
# def get_dipcoop_array(idcs1, idcs2, path=None, data=None, orbs1=None, orbs2=None, use_occs=True):
#     """ Same as get_ipcoop_array but only evaluates the bonding contributions of dpcoop(i1, i2) and antibonding of dpcoop(i2,i1)
#     """
#     data, path = get_data_and_path(data, path)
#     if not data.complex_bandprojs:
#         raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
#                          "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
#                          "to your JDFTx in file.")
#     Erange, weights_sabcj_1, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcoop_pieces(idcs1, idcs2, data, orbs1=orbs1, orbs2=orbs2, directional=True)
#     Erange, weights_sabcj_2, E_sabcj, atoms, wk_sabc, occ_sabcj = get_pcoop_pieces(idcs2, idcs1, data, orbs1=orbs2,
#                                                                                    orbs2=orbs1, directional=True)
#     wk_sabcj = np.array([wk_sabc]*data.get_nBands())
#     wk_sabcj = wk_sabcj.reshape(np.shape(E_sabcj))
#     E_flat = E_sabcj.flatten()
#     idcs = np.argsort(E_flat)
#     E = E_flat[idcs]
#     pW_1 = np.maximum(weights_sabcj_1.flatten()[idcs], np.zeros(len(E)))
#     pW_2 = np.minimum(weights_sabcj_2.flatten()[idcs], np.zeros(len(E)))
#     pW = pW_1 + pW_2
#     kW = wk_sabcj.flatten()[idcs]
#     occ = occ_sabcj.flatten()[idcs]
#     if use_occs:
#         pcoop = pW*kW*occ
#     else:
#         pcoop = pW*kW
#     ipcohp = np.zeros(len(pcoop)+1)
#     ipcohp[1:] = np.cumsum(pcoop)
#     return E, ipcohp[1:]
#
#
# def get_directional_pcoop(idcs1, idcs2, path=None, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None, spin_pol = False):
#     """
#     Same functionality as get_cheap_pcohp, but pCOHP_uv = P_uv*H_uv will instead be evaluated as pCOHP_uv = P_uu*H_uv with
#     a projection tensor P renormalized such that P_uv = P_uu/(P_uu+P_vv)
#     """
#     data, path = get_data_and_path(data, path)
#     if not data.complex_bandprojs:
#         raise ValueError("Data was not provided bandProjections in complex form - pCOHP analysis not available.\n" + \
#                          "To generate data suitable for pCOHP analysis, pleased add 'band-projection-params yes no' \n" +\
#                          "to your JDFTx in file.")
#     Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj = get_pcoop_pieces(idcs1, idcs2, data, res=res,
#                                                                             orbs1=orbs1, orbs2=orbs2, Erange=Erange,
#                                                                             directional=True)
#     cs = get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig)
#     pcohp = cs_formatter(cs, spin_pol)
#     return Erange, pcohp
