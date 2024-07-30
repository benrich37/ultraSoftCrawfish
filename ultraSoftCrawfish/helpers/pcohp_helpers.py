import numpy as np
from numba import jit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from ultraSoftCrawfish.helpers.data_parsing_helpers import get_kmap_from_atoms, get_el_orb_u_dict
from ultraSoftCrawfish.helpers.misc_helpers import gauss, get_orb_bool_func
from copy import deepcopy
from ultraSoftCrawfish.helpers.rs_helpers import get_ebound_arr
from ultraSoftCrawfish.helpers.misc_helpers import fidcs
from ultraSoftCrawfish.helpers.pdos_helpers import _get_orb_idcs




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

# # @jit(nopython=True)
# def get_P_uvjsabc_bare_min_vec(proj_jsabcu, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v):
#     # for u in orbs_u:
#     #     for v in orbs_v:
#     #         t1 = proj_jsabcu[:, :, :, :, :, u]
#     #         t2 = proj_jsabcu[:, :, :, :, :, v]
#     #         P_uvjsabc[u, v] += np.real(np.conj(t1) * t2)
#     return P_uvjsabc
#
#
# import time


def get_P_uvjsabc_bare_min(proj_sabcju, orbs_u, orbs_v):
    shape = np.shape(proj_sabcju)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nProj = shape[5]
    P_uvjsabc = np.zeros([nProj, nProj, nBands, nSpin, nKa, nKb, nKc], dtype=np.float32)
    orbs_u = np.asarray(orbs_u)
    orbs_v = np.asarray(orbs_v)
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


@jit(nopython=True)
def get_pCOOP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, wk_sabc, pCOOP_sabcj):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                p1 = P_uvjsabc[u, v, j, s, a, b, c]
                                p3 = wk_sabc[s, a, b, c]
                                uv_sum += np.real(p1 * p3)
                        pCOOP_sabcj[s, a, b, c, j] += uv_sum
    return pCOOP_sabcj


def get_pCOOP_sabcj(P_uvjsabc, orbs_u, orbs_v, wk_sabc=None):
    shape = np.shape(P_uvjsabc)
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    pCOOP_sabcj = np.zeros([nSpin, nKa, nKb, nKc, nBands])
    if wk_sabc is None:
        wk_sabc = np.ones([nSpin, nKa, nKb, nKc])
    return get_pCOOP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, wk_sabc, pCOOP_sabcj)





def mod_weights_for_ebounds(weights_sabcj, E_sabcj, ebounds):
    shape = np.shape(weights_sabcj)
    bool_arr = get_ebound_arr(ebounds, E_sabcj)
    weights_sabcj = mod_weights_for_ebounds_jit(weights_sabcj, shape, bool_arr)
    return weights_sabcj

@jit(nopython=True)
def mod_weights_for_ebounds_jit(_weights_sabcj, shape, bool_arr):
    for s in range(shape[0]):
        for a in range(shape[1]):
            for b in range(shape[2]):
                for c in range(shape[3]):
                    for j in range(shape[4]):
                        if not bool_arr[s,a,b,c,j]:
                            _weights_sabcj[s,a,b,c,j] *= 0
    return _weights_sabcj




def get_just_ipcohp_helper(occ_sabcj, weights_sabcj, wk, ebounds=None, E_sabcj=None):
    shape = np.shape(occ_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    icohp = 0
    if not ebounds is None:
        weights_sabcj = mod_weights_for_ebounds(weights_sabcj, E_sabcj, ebounds)
    icohp = get_just_ipcohp_helper_jit(occ_sabcj, weights_sabcj, wk, nSpin, nKa, nKb, nKc, nBands, icohp)
    return icohp



@jit(nopython=True)
def get_just_ipcohp_helper_jit(occ_sabcj, weights_sabcj, wk_sabc, nSpin, nKa, nKb, nKc, nBands, icohp):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        icohp += occ_sabcj[s,a,b,c,j]*weights_sabcj[s,a,b,c,j]*wk_sabc[s,a,b,c]
    return icohp





def get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=None, orbs2=None):
    idcs1 = fidcs(idcs1)
    idcs2 = fidcs(idcs2)
    orb_idcs = [[],[]]
    orbs_pulls = [orbs1, orbs2]
    kmap = get_kmap_from_atoms(atoms)
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


def get_cheap_dos_helper(Erange, E_sabcj, sig):
    nSpin = np.shape(E_sabcj)[0]
    es = []
    cs = []
    weights = np.ones(len(E_sabcj[0].flatten()))
    for s in range(nSpin):
        es.append(E_sabcj[s].flatten())
        cs.append(np.zeros(np.shape(Erange), dtype=float))
        cs[s] = get_cheap_pcohp_jit(Erange, es[s], weights, cs[s], sig)
    return cs


def get_cheap_pcohp_helper(Erange, E_sabcj, weights_sabcj, sig):
    nSpin = np.shape(E_sabcj)[0]
    ws = []
    es = []
    cs = []
    for s in range(nSpin):
        ws.append(weights_sabcj[s].flatten())
        es.append(E_sabcj[s].flatten())
        cs.append(np.zeros(np.shape(Erange), dtype=float))
        cs[s] = get_cheap_pcohp_jit(Erange, es[s], ws[s], cs[s], sig)
    return cs


@jit(nopython=True)
def get_cheap_pcohp_jit(Erange, eflat, wflat, cflat, sig):
    for i in range(len(eflat)):
        cflat += gauss(Erange, eflat[i], sig) * wflat[i]
    return cflat





def get_pcohp_pieces(idcs1, idcs2, data, res=0.01, orbs1=None, orbs2=None, Erange=None, directional=False,
                      a=False, d=False):
    if True in [a, d]:
        if not directional:
            raise ValueError("pCOHP must be directional to allow accepting/donation interactions")
    idcs1 = fidcs(idcs1)
    idcs2 = fidcs(idcs2)
    atoms = data.get_atoms()
    E_sabcj = data.get_E_sabcj()
    proj_sabcju = data.get_proj_sabcju()
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
    orb_idcs = _get_orb_idcs(idcs1, idcs2, orbs1, orbs2, data)
    if directional:
        P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0]+orb_idcs[1], orb_idcs[0]+orb_idcs[1])
    else:
        P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0], orb_idcs[1])
    H_uvsabc = get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orb_idcs[0], orb_idcs[1])
    if directional:
        P_uvjsabc = assign_directionality_to_P_uvjsabc(P_uvjsabc, a, d, orb_idcs[0], orb_idcs[1])
    weights_sabcj = get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, orb_idcs[0], orb_idcs[1])
    return Erange, weights_sabcj, E_sabcj, atoms, data.get_wk_sabc(), data.get_occ_sabcj()

import time


def make_P_uvjsabc_directional(P_uvjsabc, us, vs, f=False):
    _P_uvjsabc = deepcopy(P_uvjsabc)
    for u in us:
        for v in vs:
            v1 = np.abs(P_uvjsabc[u,u])
            v2 = np.abs(P_uvjsabc[v,v])
            # v1 = P_uvjsabc[u,u]
            # v2 = P_uvjsabc[v,v]
            vsum = v1+v2
            if not f:
                fac = np.divide(v1, vsum, out=np.ones_like(v1), where = vsum != 0)
            else:
                fac = np.divide(v2, vsum, out=np.ones_like(v1), where=vsum != 0)
            _P_uvjsabc[u, v] *= fac
    return _P_uvjsabc

# @jit(nopython=True)
# def make_P_uvjsabc_directional_jit(P_uvjsabc, us, vs, jsabc):
#     for u in us:
#         for v in vs:
#             v1 = np.abs(P_uvjsabc[u,u])
#             v2 = np.abs(P_uvjsabc[v,v])
#             vsum = v1+v2
#             for j in range(jsabc[0]):
#                 for s in range(jsabc[1]):
#                     for a in range(jsabc[2]):
#                         for b in range(jsabc[3]):
#                             for c in range(jsabc[4]):
#                                 if not np.isclose(vsum[j,s,a,b,c], 0):
#                                     P_uvjsabc[u, v, j, s, a, b,c ] *= v1[j,s,a,b,c]/vsum[j,s,a,b,c]
#             #P_uvjsabc[u, v] *= np.divide(v2, vsum, out=np.ones_like(v1), where = vsum != 0)
#     return P_uvjsabc
#
#
# @jit(nopython=True)
# def make_P_uvjsabc_directional_jit_f(P_uvjsabc, us, vs, jsabc):
#     for u in us:
#         for v in vs:
#             v1 = np.abs(P_uvjsabc[u,u])
#             v2 = np.abs(P_uvjsabc[v,v])
#             vsum = v1+v2
#             for j in range(jsabc[0]):
#                 for s in range(jsabc[1]):
#                     for a in range(jsabc[2]):
#                         for b in range(jsabc[3]):
#                             for c in range(jsabc[4]):
#                                 if not np.isclose(vsum[j,s,a,b,c], 0):
#                                     P_uvjsabc[u, v, j, s, a, b,c ] *= v2[j,s,a,b,c]/vsum[j,s,a,b,c]
#             #P_uvjsabc[u, v] *= np.divide(v2, vsum, out=np.ones_like(v1), where = vsum != 0)
#     return P_uvjsabc


def get_pcoop_pieces(idcs1, idcs2, data, res=0.01, orbs1=None, orbs2=None, Erange=None, directional=False, a=False, d=False):
    idcs1 = fidcs(idcs1)
    idcs2 = fidcs(idcs2)
    atoms = data.get_atoms()
    E_sabcj = data.get_E_sabcj()
    proj_sabcju = data.get_proj_sabcju()
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
    orb_idcs = _get_orb_idcs(idcs1, idcs2, orbs1, orbs2, data)
    if directional:
        P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0]+orb_idcs[1], orb_idcs[0]+orb_idcs[1])
        P_uvjsabc = assign_directionality_to_P_uvjsabc(P_uvjsabc, a, d, orb_idcs[0], orb_idcs[1])
    else:
        P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0], orb_idcs[1])
    weights_sabcj = get_pCOOP_sabcj(P_uvjsabc, orb_idcs[0], orb_idcs[1])
    return Erange, weights_sabcj, E_sabcj, atoms, data.get_wk_sabc(), data.get_occ_sabcj()

def assign_directionality_to_P_uvjsabc(P_uvjsabc, a, d, o0s, o1s):
    if not True in [a, d]:
        P_uvjsabc = make_P_uvjsabc_directional(P_uvjsabc, o0s, o1s)
    else:
        # s = time.time()
        # aP_uvjsabc = make_P_uvjsabc_directional(P_uvjsabc, o0s, o1s)
        # dP_uvjsabc = make_P_uvjsabc_directional(P_uvjsabc, o0s, o1s, f=True)
        # P_uvjsabc = np.zeros_like(P_uvjsabc)
        # if a:
        #     padd = np.maximum(aP_uvjsabc, 0) + np.minimum(dP_uvjsabc, 0)
        #     P_uvjsabc = add_padd(P_uvjsabc, padd, o0s, o1s)
        # if d:
        #     padd = np.maximum(dP_uvjsabc, 0) + np.minimum(aP_uvjsabc, 0)
        #     P_uvjsabc = add_padd(P_uvjsabc, padd, o0s, o1s)
        # print(f"old: {time.time()-s}")
        # s = time.time()
        if a:
            P_uvjsabc = make_aP_uvjsabc(P_uvjsabc, o0s, o1s)
        if d:
            P_uvjsabc = make_dP_uvjsabc(P_uvjsabc, o0s, o1s)
    return P_uvjsabc

def make_aP_uvjsabc(P_uvjsabc, o0s, o1s):
    aP_uvjsabc = np.zeros_like(P_uvjsabc)
    for u in o0s:
        for v in o1s:
            v1 = np.abs(P_uvjsabc[u, u])
            v2 = np.abs(P_uvjsabc[v, v])
            vsum = v1 + v2
            aP_uvjsabc[u, v] += np.maximum(P_uvjsabc[u, v] * np.divide(v1, vsum, out=np.ones_like(v1), where=vsum != 0), 0) +\
                                np.minimum(P_uvjsabc[u, v] * np.divide(v2, vsum, out=np.ones_like(v1), where=vsum != 0), 0)
            # aP_uvjsabc[u,v] += P_uvjsabc[u, v]*(np.maximum(np.divide(v1, vsum, out=np.ones_like(v1), where = vsum != 0), 0) +
            #                                     np.minimum(np.divide(v2, vsum, out=np.ones_like(v1), where=vsum != 0),
            #                                                0))
    return aP_uvjsabc


def make_dP_uvjsabc(P_uvjsabc, o0s, o1s):
    dP_uvjsabc = np.zeros_like(P_uvjsabc)
    for u in o0s:
        for v in o1s:
            v1 = np.abs(P_uvjsabc[u, u])
            v2 = np.abs(P_uvjsabc[v, v])
            vsum = v1 + v2
            dP_uvjsabc[u, v] += np.maximum(P_uvjsabc[u, v] * np.divide(v2, vsum, out=np.ones_like(v1), where=vsum != 0), 0) +\
                                np.minimum(P_uvjsabc[u, v] * np.divide(v1, vsum, out=np.ones_like(v1), where=vsum != 0), 0)
            # aP_uvjsabc[u,v] += P_uvjsabc[u, v]*(np.maximum(np.divide(v1, vsum, out=np.ones_like(v1), where = vsum != 0), 0) +
            #                                     np.minimum(np.divide(v2, vsum, out=np.ones_like(v1), where=vsum != 0),
            #                                                0))
    return dP_uvjsabc
    

@jit(nopython=True)
def add_padd(P, padd, orb1, orb2):
    for u in orb1:
        for v in orb2:
            P[u, v] += padd[u, v]
    return P
