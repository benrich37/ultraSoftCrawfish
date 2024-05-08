import numpy as np
from numba import jit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

from helpers.ase_helpers import get_atoms
from helpers.data_parsers import get_kmap, get_el_orb_u_dict, parse_data

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@jit(nopython=True)
def get_P_uvjsabc_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin):
    for u in range(nProj):
        for v in range(nProj):
            for j in range(nBands):
                for a in range(nKa):
                    for b in range(nKb):
                        for c in range(nKc):
                            for s in range(nSpin):
                                t1 = proj_sabcju[s, a, b, c, j, u]
                                t2 = proj_sabcju[s, a, b, c, j, v]
                                P_uvjsabc[u, v, j, s, a, b, c] = np.conj(t1) * t2
    return P_uvjsabc


def get_P_uvjsabc(proj_sabcju):
    shape = np.shape(proj_sabcju)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nProj = shape[5]
    P_uvjsabc = np.zeros([nProj, nProj, nBands, nSpin, nKa, nKb, nKc], dtype=complex)
    P_uvjsabc = get_P_uvjsabc_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin)
    return np.real(P_uvjsabc)


@jit(nopython=True)
def get_P_uvjsabc_bare_min_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nBands):
                for a in range(nKa):
                    for b in range(nKb):
                        for c in range(nKc):
                            for s in range(nSpin):
                                t1 = proj_sabcju[s, a, b, c, j, u]
                                t2 = proj_sabcju[s, a, b, c, j, v]
                                P_uvjsabc[u, v, j, s, a, b, c] += np.real(np.conj(t1) * t2)
    return P_uvjsabc


def get_P_uvjsabc_bare_min(proj_sabcju, orbs_u, orbs_v):
    shape = np.shape(proj_sabcju)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nProj = shape[5]
    P_uvjsabc = np.zeros([nProj, nProj, nBands, nSpin, nKa, nKb, nKc], dtype=np.float32)
    P_uvjsabc = get_P_uvjsabc_bare_min_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v)
    return np.real(P_uvjsabc)


@jit(nopython=True)
def get_H_uvsabc_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin):
    for u in range(nProj):
        for v in range(nProj):
            for j in range(nBands):
                for s in range(nSpin):
                    for a in range(nKa):
                        for b in range(nKb):
                            for c in range(nKc):
                                H_uvsabc[u, v, s, a, b, c] += P_uvjsabc[u, v, j, s, a, b, c] * E_sabcj[s, a, b, c, j]
    return H_uvsabc


def get_H_uvsabc(P_uvjsabc, E_sabcj):
    shape = np.shape(P_uvjsabc)
    nProj = shape[0]
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    H_uvsabc = np.zeros([nProj, nProj, nSpin, nKa, nKb, nKc], dtype=complex)
    return get_H_uvsabc_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin)


@jit(nopython=True)
def get_H_uvsabc_bare_min_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nBands):
                for s in range(nSpin):
                    for a in range(nKa):
                        for b in range(nKb):
                            for c in range(nKc):
                                H_uvsabc[u, v, s, a, b, c] += P_uvjsabc[u, v, j, s, a, b, c] * E_sabcj[s, a, b, c, j]
    return H_uvsabc


def get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orbs_u, orbs_v):
    shape = np.shape(P_uvjsabc)
    nProj = shape[0]
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    H_uvsabc = np.zeros([nProj, nProj, nSpin, nKa, nKb, nKc], dtype=float)
    return get_H_uvsabc_bare_min_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v)


@jit(nopython=True)
def get_pCOHP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                p1 = P_uvjsabc[u, v, j, s, a, b, c]
                                p2 = H_uvsabc[u, v, s, a, b, c]
                                p3 = wk_sabc[s, a, b, c]
                                uv_sum += np.real(p1 * p2) * p3
                        pCOHP_sabcj[s, a, b, c, j] += uv_sum
    return pCOHP_sabcj


def get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, orbs_u, orbs_v, wk_sabc=None):
    shape = np.shape(P_uvjsabc)
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    pCOHP_sabcj = np.zeros([nSpin, nKa, nKb, nKc, nBands])
    if wk_sabc is None:
        wk_sabc = np.ones([nSpin, nKa, nKb, nKc])
    return get_pCOHP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj)


def get_just_icohp_helper(occ_sabcj, weights_sabcj, wk):
    shape = np.shape(occ_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    icohp = 0
    icohp = get_just_icohp_helper_jit(occ_sabcj, weights_sabcj, wk, nSpin, nKa, nKb, nKc, nBands, icohp)
    return icohp


@jit(nopython=True)
def get_just_icohp_helper_jit(occ_sabcj, weights_sabcj, wk_sabc, nSpin, nKa, nKb, nKc, nBands, icohp):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        icohp += occ_sabcj[s,a,b,c,j]*weights_sabcj[s,a,b,c,j]*wk_sabc[s,a,b,c]
    return icohp


def get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=None, orbs2=None):
    orb_idcs = [[],[]]
    orbs_pulls = [orbs1, orbs2]
    kmap = get_kmap(atoms)
    for i, set in enumerate([idcs1, idcs2]):
        orbs_pull = orbs_pulls[i]
        if not orbs_pull is None:
            el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, set)
            for el in el_orb_u_dict:
                for orb in el_orb_u_dict[el]:
                    if type(orbs_pull) is list:
                        for orbi in orbs_pull:
                            if orbi in orb:
                                orb_idcs[i] += el_orb_u_dict[el][orb]
                                break
                    else:
                        if orbs_pull in orb:
                            orb_idcs[i] += el_orb_u_dict[el][orb]
        else:
            for idx in set:
                orb_idcs[i] += orbs_dict[kmap[idx]]
    return orb_idcs


def get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig):
    nSpin = np.shape(E_sabcj)[0]
    ws = []
    es = []
    cs = []
    for s in range(nSpin):
        ws.append(weights_sabcj[s].flatten())
        es.append(E_sabcj[s].flatten())
        cs.append(np.zeros(np.shape(Erange), dtype=float))
        cs[s] = get_cheap_pcohp_jit(Erange, es[s], es[s], cs[s], sig)
    return cs


@jit(nopython=True)
def get_cheap_pcohp_jit(Erange, eflat, wflat, cflat, sig):
    for i in range(len(eflat)):
        cflat += gauss(Erange, eflat[i], sig)*wflat[i]
    return cflat

@jit(nopython=True)
def gauss(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / sig)


def get_pcohp_pieces(idcs1, idcs2, path, data=None, res=0.01, orbs1=None, orbs2=None, Erange=None):
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    atoms = get_atoms(path)
    kmap = get_kmap(atoms)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
    orb_idcs = [[],[]]
    orbs_pulls = [orbs1, orbs2]
    for i, set in enumerate([idcs1, idcs2]):
        orbs_pull = orbs_pulls[i]
        if not orbs_pull is None:
            el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, set)
            for el in el_orb_u_dict:
                for orb in el_orb_u_dict[el]:
                    if type(orbs_pull) is list:
                        for orbi in orbs_pull:
                            if orbi in orb:
                                orb_idcs[i] += el_orb_u_dict[el][orb]
                                break
                    else:
                        if orbs_pull in orb:
                            orb_idcs[i] += el_orb_u_dict[el][orb]
        else:
            for idx in set:
                orb_idcs[i] += orbs_dict[kmap[idx]]
    P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0], orb_idcs[1])
    H_uvsabc = get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orb_idcs[0], orb_idcs[1])
    weights_sabcj = get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, orb_idcs[0], orb_idcs[1])
    return Erange, weights_sabcj, E_sabcj, atoms, wk, occ_sabcj
