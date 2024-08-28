from ase.data import atomic_numbers
from ultraSoftCrawfish.helpers.data_parsing_helpers import get_el_orb_u_dict, get_start_line, orb_ref_list
import numpy as np
from os.path import join as opj

def get_Z_val(el, outfile):
    start_line = get_start_line(outfile)
    reading_key = "Reading pseudopotential file"
    valence_key = " valence electrons in orbitals"
    Z_val = 0
    with open(outfile, "r") as f:
        reading = False
        for i, line in enumerate(f):
            if i > start_line:
                if reading:
                    if valence_key in line:
                        v = line.split(valence_key)[0].split(" ")[-1]
                        Z_val = int(v)
                        break
                    else:
                        continue
                else:
                    if reading_key in line:
                        fpath = line.split("'")[1]
                        fname = fpath.split("/")[-1]
                        ftitle = fname.split(".")[0].lower()
                        if el.lower() in ftitle.split("_"):
                            reading = True
                        else:
                            reading = False
    return Z_val


def get_n_elec_dicts(data, aidcs, charges=None):
    norm = data.norm
    data.norm_projs_t2()
    _occ_sabcj = data.get_occ_sabcj()
    orbs_idx_dict = data.get_orbs_idx_dict()
    _proj_sabcju = np.abs(data.get_proj_sabcju()) ** 2
    # _proj_sabcju = np.abs(data.get_proj_sabcju()) ** 2 / np.prod(np.shape(_occ_sabcj)[1:4])
    n_elec_dicts = []
    for i, aidx in enumerate(aidcs):
        el_orb_u_dict = get_el_orb_u_dict(data.root, data.get_atoms(), orbs_idx_dict, [aidx])
        el = list(el_orb_u_dict.keys())[0]
        n_elec_dict = {}
        for utype in el_orb_u_dict[el]:
            n_elec_dict[utype] = 0
            for u in el_orb_u_dict[el][utype]:
                # print(np.sum(_proj_sabcju[:,:,:,:,:,u]))
                n_elec_dict[utype] += np.sum(_occ_sabcj * _proj_sabcju[:, :, :, :, :, u])
        if not charges is None:
            _ned = {}
            tocc = 0
            for utype in el_orb_u_dict[el]:
                tocc += n_elec_dict[utype]
            tocc_target = get_Z_val(el, opj(data.root, "out")) - charges[i]
            coef = tocc_target / tocc
            for utype in n_elec_dict:
                _ned[utype] = coef*n_elec_dict[utype]
            if _ned_is_good(_ned):
                n_elec_dict = _ned
            else:
                _ned = _ned_opt(_ned)
                n_elec_dict = _ned
        n_elec_dicts.append(n_elec_dict)
    if not norm:
        data.unnorm_projs()
    else:
        if norm == 1:
            data.norm_projs_t1()
    return n_elec_dicts


def _ned_is_good(_ned):
    good = np.all([_ned[ut]<2 for ut in list(_ned.keys())])
    return good


def get_ned_subtr(_ned):
    subtr = 0
    for ut in _ned.keys():
        if _ned[ut] > 2:
            subtr += _ned[ut] - 2
            _ned[ut] = 2
    return subtr, _ned

def redist_subtr(subtr, _ned):
    can_add = []
    for ut in _ned.keys():
        if _ned[ut] < 2:
            can_add.append(ut)
    for ut in can_add:
        _ned[ut] += subtr/len(can_add)
    return _ned

def bad_subtr_ned_combo(subtr, _ned, tol):
    room = 0
    for ut in _ned:
        room += 2 - _ned[ut]
    bad = subtr > room+tol
    return bad

def _ned_opt(_ned, tol=1e-5):
    subtr, _ned = get_ned_subtr(_ned)
    if not bad_subtr_ned_combo(subtr, _ned, tol):
        while subtr - tol > 0:
            _ned = redist_subtr(subtr, _ned)
            subtr, _ned = get_ned_subtr(_ned)
    else:
        raise ValueError("Requested charge requires electrons exceeding capacity of associated projections")
    return _ned




def get_pol_dicts(data, aidcs, charges=None):
    n_elec_dicts = get_n_elec_dicts(data, aidcs, charges=charges)
    _occ_sabcj = data.get_occ_sabcj()
    _proj_sabcju = np.abs(data.get_proj_sabcju()) ** 2 / np.prod(np.shape(_occ_sabcj)[1:4])
    orbs_idx_dict = data.get_orbs_idx_dict()
    pol_dicts = []
    for i, aidx in enumerate(aidcs):
        el_orb_u_dict = get_el_orb_u_dict(data.root, data.get_atoms(), orbs_idx_dict, [aidx])
        el = list(el_orb_u_dict.keys())[0]
        pol_dict = {}
        for utype in el_orb_u_dict[el]:
            up = 0
            down = 0
            for u in el_orb_u_dict[el][utype]:
                # print(np.sum(_proj_sabcju[:,:,:,:,:,u]))
                up += np.sum(_occ_sabcj[0] * _proj_sabcju[0, :, :, :, :, u])
                down += np.sum(_occ_sabcj[1] * _proj_sabcju[1, :, :, :, :, u])
            coef = n_elec_dicts[i][utype] / (up + down)
            up *= coef
            down *= coef
            pol_dict[utype] = abs(up - down)
        pol_dicts.append(pol_dict)
    return pol_dicts


def get_elec_config_dicts(data, aidcs, charges=None, ml=False, rm_empty=True):
    elec_config_dicts = []
    n_elec_dicts = get_n_elec_dicts(data, aidcs, charges=charges)
    uts = ["s", "p", "d", "f"]
    if ml:
        uts = []
        for _ in orb_ref_list:
            uts += _
    for i, aidx in enumerate(aidcs):
        elec_config_dict = {}
        for ut in uts:
            elec_config_dict[ut] = 0
        for orb in elec_config_dict:
            for utype in n_elec_dicts[i]:
                if orb in utype:
                    elec_config_dict[orb] += n_elec_dicts[i][utype]
        elec_config_dicts.append(elec_config_dict)
    if rm_empty:
        to_pop = []
        for ut in elec_config_dicts[0]:
            if np.all([elec_config_dicts[i][ut] == 0 for i in range(len(aidcs))]):
                to_pop.append(ut)
        _elec_config_dicts = []
        for i in range(len(aidcs)):
            _elec_config_dict = {}
            elec_config_dict = elec_config_dicts[i]
            for ut in elec_config_dict:
                if not ut in to_pop:
                    _elec_config_dict[ut] = elec_config_dict[ut]
            _elec_config_dicts.append(_elec_config_dict)
        return _elec_config_dicts
    else:
        return elec_config_dicts

