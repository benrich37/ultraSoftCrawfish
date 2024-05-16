from ultraSoftCrawfish.helpers.data_parsing_helpers import get_el_orb_u_dict
from ultraSoftCrawfish.helpers.ElecData import parse_data
from ultraSoftCrawfish.helpers.misc_helpers import get_orb_bool_func
import numpy as np


def get_pdos_weights_sabcj(idcs, path, data, orb_bool_func):
    if data is None:
        data = parse_data(root=path)
    if idcs is None:
        idcs = list(range(len(data.get_atoms())))
    orbs_idx_dict = data.get_orbs_idx_dict()
    atoms = data.get_atoms()
    kmap = data.get_kmap()
    E_sabcj = data.get_E_sabcj()
    proj_sabcju = data.get_proj_sabcju()
    orbs = []
    if not orb_bool_func is None:
        el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_idx_dict, idcs)
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


def get_pdos_pieces(idcs, path, data, res, orbs, Erange):
    orb_bool_func = get_orb_bool_func(orbs)
    if data is None:
        data = parse_data(root=path)
    E_sabcj = data.get_E_sabcj()
    weights_sabcj = get_pdos_weights_sabcj(idcs, path, data, orb_bool_func)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj) - (10 * res), np.max(E_sabcj) + (10 * res), res)
    return Erange, weights_sabcj, E_sabcj
