from ultraSoftCrawfish.helpers.data_parsing_helpers import get_el_orb_u_dict
from ultraSoftCrawfish.helpers.ElecData import parse_data
from ultraSoftCrawfish.helpers.misc_helpers import get_orb_bool_func, fidcs
import numpy as np


def get_pdos_weights_sabcj(idcs, data, orb_bool_func):
    if idcs is None:
        idcs = list(range(len(data.get_atoms())))
    idcs = fidcs(idcs)
    orbs_idx_dict = data.get_orbs_idx_dict()
    atoms = data.get_atoms()
    kmap = data.get_kmap()
    E_sabcj = data.get_E_sabcj()
    proj_sabcju = data.get_proj_sabcju()
    orbs = []
    if not orb_bool_func is None:
        el_orb_u_dict = get_el_orb_u_dict(data.root, atoms, orbs_idx_dict, idcs)
        for el in el_orb_u_dict:
            for orb in el_orb_u_dict[el]:
                if orb_bool_func(orb):
                    orbs += el_orb_u_dict[el][orb]
    else:
        for idx in idcs:
            orbs += orbs_idx_dict[kmap[idx]]
    weights_sabcj = np.zeros(np.shape(E_sabcj))
    for orb in orbs:
        weights_sabcj += np.abs(proj_sabcju[:,:,:,:,:,orb])**2
    return weights_sabcj


def get_pdos_pieces(idcs, data, res, orbs, Erange):
    idcs = fidcs(idcs)
    orb_bool_func = get_orb_bool_func(orbs)
    E_sabcj = data.get_E_sabcj()
    weights_sabcj = get_pdos_weights_sabcj(idcs, data, orb_bool_func)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj) - (10 * res), np.max(E_sabcj) + (10 * res), res)
    return Erange, weights_sabcj, E_sabcj


def _get_orb_idcs(idcs1, idcs2, orbs1, orbs2, data):
    orbs_idx_dict = data.get_orbs_idx_dict()
    kmap = data.get_kmap()
    orb_idcs = [[], []]
    orbs_pulls = [orbs1, orbs2]
    for i, set in enumerate([idcs1, idcs2]):
        orbs_pull = orbs_pulls[i]
        if not orbs_pull is None:
            el_orb_u_dict = get_el_orb_u_dict(data.root, data.get_atoms(), orbs_idx_dict, set)
            orb_bool_func = get_orb_bool_func(orbs_pull)
            for el in el_orb_u_dict:
                for orb in el_orb_u_dict[el]:
                    if orb_bool_func(orb):
                        orb_idcs[i] += el_orb_u_dict[el][orb]
        else:
            for idx in set:
                orb_idcs[i] += orbs_idx_dict[kmap[idx]]
    return orb_idcs
