from helpers.ase_helpers import get_atoms
from helpers.data_parsers import get_kmap, get_el_orb_u_dict, parse_data
import numpy as np


def get_pdos_weights_sabcj(idcs, path, data, orb_bool_func):
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    atoms = get_atoms(path)
    kmap = get_kmap(atoms)
    orbs = []
    if not orb_bool_func is None:
        el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, idcs)
        for el in el_orb_u_dict:
            for orb in el_orb_u_dict[el]:
                if orb_bool_func(orb):
                    orbs += el_orb_u_dict[el][orb]
    else:
        for idx in idcs:
            orbs += orbs_dict[kmap[idx]]
    weights_sabcj = np.zeros(np.shape(E_sabcj))
    for orb in orbs:
        weights_sabcj += np.abs(proj_sabcju[:,:,:,:,:,orb])
    return weights_sabcj


def get_pdos_pieces(idcs, path, data, res, orbs, Erange):
    orb_bool_func = None
    if not orbs is None:
        if type(orbs) is list:
            orb_bool_func = lambda s: True in [o in s for o in orbs]
        else:
            orb_bool_func = lambda s: orbs in s
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    weights_sabcj = get_pdos_weights_sabcj(idcs, path, data, orb_bool_func)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj) - (10 * res), np.max(E_sabcj) + (10 * res), res)
    return Erange, weights_sabcj, E_sabcj
