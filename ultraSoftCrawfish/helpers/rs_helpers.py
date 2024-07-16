import numpy as np
import sys
sys.path.append("../shared_funcs")
sys.path.append("C:\\Users\\User\\PycharmProjects\\Musgrave_scripts\\BenRich\\common_funcs\\common_funcs")
# from spath_funcs import *
from ase.io import read
from pyfftw.interfaces.numpy_fft import ifftn
from os.path import join as opj, exists as ope
from os import mkdir


def pcat(root, dirnames, force=False):
    path = root
    for dn in dirnames:
        path = opj(path, dn)
        if force:
            if not ope(path):
                mkdir(path)
    return path

def get_start_lines(outfname):
    start_lines = []
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line:
            start_lines.append(i)
    return start_lines

def get_start_line(outfname):
    start_lines = get_start_lines(outfname)
    return start_lines[-1]

def get_S_R(outfile):
    start = get_start_line(outfile)
    R = np.zeros((3, 3))
    iLine = 0
    refLine = -10
    initDone = False
    Rdone = False
    Sdone = False
    for i, line in enumerate(open(outfile)):
        if i > start:
            if line.startswith('Initialization completed'):
                initDone = True
            if line.find('Initializing the Grid') >= 0:
                refLine = iLine
            if not Rdone:
                rowNum = iLine - (refLine + 2)
                if rowNum >= 0 and rowNum < 3:
                    R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
                if rowNum == 3:
                    Rdone = True
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
            iLine += 1
    return S, R

def dump_cub_inner(inner_array, S_2):
    dump_str = ""
    nLoops = int(np.floor(S_2/6.))
    nSpill = int(S_2 - (6.*nLoops))
    for i in range(nLoops):
        for j in range(6):
            dump_str += f"{float_to_str(inner_array[(6*i) + j])} "
    for i in list(range(nSpill))[::-1]:
        dump_str += f"{float_to_str(inner_array[i])} "
    dump_str += "\n"
    return dump_str

def float_to_str(num):
    return f"{num:.{6}e}"


def write_cube_writer(atoms, cube_file_path, d, title_card):
    nAtoms = len(atoms)
    S = np.shape(d)
    nNums = atoms.get_atomic_numbers()
    R = np.array(atoms.cell) * (1/0.529177) # A
    posns = atoms.positions * (1/0.529177)
    dump_str = f"Title card\n{title_card}\n"
    dump_str += f"{nAtoms} 0.0 0.0 0.0 \n"
    for i in range(3): # writing the voxel size
        dump_str += f"{S[i]} "
        v = R[i]/float(S[i])
        for j in range(3):
            dump_str += f"{v[j]} "
        dump_str += "\n"
    for i in range(nAtoms):
        dump_str += f"{int(nNums[i])} {float(nNums[i])} "
        posn = posns[i]
        for j in range(3):
            dump_str += f"{posn[j]} "
        dump_str += "\n"
    for i in range(S[0]):
        for j in range(S[1]):
            dump_str += dump_cub_inner(d[i,j,:], S[2])
    with open(cube_file_path, "w") as f:
        f.write(dump_str)

def write_cube_helper(outfile, contcar_path, d, cube_file_prefix=None, title_card="Electron density from Total SCF Density", atoms=None):
    S, R = get_S_R(outfile)
    if atoms is None:
        atoms = read(contcar_path, format="vasp")
    dshape = np.shape(d)
    if not len(dshape) == len(S):
        d = d.reshape(S)
    if cube_file_prefix is None:
        cube_file_prefix = contcar_path
    add_cube_suffix = False
    if not "." in cube_file_prefix:
        add_cube_suffix = True
    elif cube_file_prefix.split(".")[-1] != ".cub":
        add_cube_suffix = True
    if add_cube_suffix:
        cube_file_prefix = opj(cube_file_prefix, ".cub")
    cube_file_path = cube_file_prefix
    write_cube_writer(atoms, cube_file_path, d, title_card)

# def write_cube(calc_dir, fname_prefix, infs=None, outdir=None, atoms=None):
#     outfile = opj(calc_dir, "out")
#     if atoms is None:
#         atoms = get_atoms_from_out(outfile)
#     if not ope(infs[0]):
#         infs = [opj(calc_dir, inf) for inf in infs]
#     n_arr = np.fromfile(infs[0])
#     for i in range(len(infs) - 1):
#         n_arr += np.fromfile(infs[i+1])
#     if outdir is None:
#         outdir = calc_dir
#     write_cube_helper(outfile, opj(outdir, fname_prefix), n_arr, cube_file_prefix=None,
#                       title_card="Electron density from Total SCF Density", atoms=atoms)

def get_n_arr(calc_dir, S):
    infs = [opj(calc_dir, "n")]
    if not ope(infs[0]):
        infs = [opj(calc_dir, x) for x in ["n_up", "n_dn"]]
    n_arr = np.fromfile(infs[0])
    for i in range(len(infs) - 1):
        n_arr += np.fromfile(infs[i+1])
    n_arr = n_arr.reshape(S)
    return n_arr

# def get_fd_fukui_arrs(sys_dir):
#     outfile = pcat(sys_dir, ["No_bias", "ion_opt", "out"])
#     S, R = get_S_R(outfile)
#     n_neut = get_n_arr(pcat(sys_dir, ["No_bias", "ion_opt"]), S)
#     n_p1 = get_n_arr(pcat(sys_dir, ["No_bias_p1", "ion_opt"]), S)
#     n_n1 = get_n_arr(pcat(sys_dir, ["No_bias_n1", "ion_opt"]), S)
#     return n_neut, n_n1, n_p1
#
# def get_fd_fukui_cubes(sys_dir, title_card="Electron density from Total SCF Density"):
#     """
#     :param sys_dir: Path for system to analyze - subdirectories must include [No_bias,No_bias_p1,No_bias_b1]
#                     and each must have density files
#     :return:
#     """
#     outfile = pcat(sys_dir, ["No_bias", "ion_opt", "out"])
#     atoms = get_atoms_from_out(outfile)
#     n_neut, n_n1, n_p1 = get_fd_fukui_arrs(sys_dir)
#     f_nphil = n_neut - n_p1
#     f_ephil = n_n1 - n_neut
#     f_rphil = (f_ephil + f_nphil)/2
#     write_cube_writer(atoms, opj(sys_dir, "f_ephil.cub"), f_ephil, title_card)
#     write_cube_writer(atoms, opj(sys_dir, "f_nphil.cub"), f_nphil, title_card)
#     write_cube_writer(atoms, opj(sys_dir, "f_rphil.cub"), f_rphil, title_card)


###########


def get_S_R_mu(outfile):
    start = get_start_line(outfile)
    R = np.zeros((3, 3))
    iLine = 0
    refLine = -10
    initDone = False
    Rdone = False
    Sdone = False
    mu = None
    for i, line in enumerate(open(outfile)):
        if i > start:
            if line.startswith('Initialization completed'):
                initDone = True
            if initDone and line.find('FillingsUpdate:') >= 0:
                mu = float(line.split()[2])
            if line.find('Initializing the Grid') >= 0:
                refLine = iLine
            if not Rdone:
                rowNum = iLine - (refLine + 2)
                if rowNum >= 0 and rowNum < 3:
                    R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
                if rowNum == 3:
                    Rdone = True
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
            iLine += 1
    return S, R, mu



def get_iGarr_wk(Gvectors_path):
    iGarr = []
    wk = []
    for line in open(Gvectors_path):
        if line.startswith('#'):
            # wk should collect the k point weights, not spin
            # the if/else below should correct this
            if line.split()[-1] == 'spin':
                wk.append(float(line.split()[-3]))
            else:
                wk.append(float(line.split()[-1]))
            iGcur = []
        elif len(line)>1:
            iGcur.append([int(tok) for tok in line.split()])
        else:
            iGarr.append(np.array(iGcur))
    return iGarr, wk

def get_nStates(wk):
    return len(wk)

def get_E(eigenvalspath, nStates):
    E = np.fromfile(eigenvalspath)
    E = E.reshape(nStates,-1)
    return E

def get_nBands(E):
    nBands = E.shape[1]
    return nBands



def get_rs_wfns_runner(wfnspath, iGarr, S, nBands, target_states, target_bands):
    rs_wfns_shape = [len(target_states), len(target_bands)]
    rs_wfns_shape += list(S)
    print(rs_wfns_shape)
    rs_wfns = np.zeros(rs_wfns_shape, dtype=float)
    with open(wfnspath) as fp:
        for iState, iGcur in enumerate(iGarr):
            #  wkCur = wk[iState]
            nG = iGcur.shape[0]
            iGcur += np.where(iGcur<0, S[None,:], 0)
            stride = np.array([S[2]*S[1], S[2], 1], dtype=int)
            index = np.dot(iGcur, stride)
            C = np.reshape(np.fromfile(fp, dtype=np.complex128, count=nBands*nG), (nBands, nG))
            if iState in target_states:
                _iState = target_states.index(iState)
                psiTilde = np.zeros(np.prod(S), dtype=np.complex128)
                for iBand in range(nBands):
                    if iBand in target_bands:
                        _iBand = target_bands.index(iBand)
                        psiTilde[index] = C[iBand]
                        psi = ifftn(np.reshape(psiTilde, S)) * np.prod(S)
                        rs_wfns[_iState, _iBand] += abs(psi)
    return rs_wfns

def get_rs_wfns(calc_dir, target_states=None, target_bands=None):
    outfile = opj(calc_dir, "out")
    S, R, mu = get_S_R_mu(outfile)
    Gvectors_path = opj(calc_dir, "Gvectors")
    iGarr, wk = get_iGarr_wk(Gvectors_path)
    nStates = get_nStates(wk)
    eigenvalspath = opj(calc_dir, "eigenvals")
    E = get_E(eigenvalspath, nStates)
    nBands = get_nBands(E)
    wfnspath = opj(calc_dir, "wfns")
    if target_states is None:
        target_states = list(range(nStates))
    if target_bands is None:
        target_bands = [nBands-1]
    rs_wfns = get_rs_wfns_runner(wfnspath, iGarr, S, nBands, target_states, target_bands)
    return rs_wfns


def get_rs_wfn_target_runner(wfnspath, iGarr, S, nBands, weights, target_kjs_dict):
    rs_wfn = np.zeros(S, dtype=float)
    with open(wfnspath) as fp:
        for k, iGcur in enumerate(iGarr):
            # wkCur = wk[iState]
            nG = iGcur.shape[0]
            iGcur += np.where(iGcur<0, S[None,:], 0)
            stride = np.array([S[2]*S[1], S[2], 1], dtype=int)
            index = np.dot(iGcur, stride)
            C = np.reshape(np.fromfile(fp, dtype=np.complex128, count=nBands*nG), (nBands, nG))
            if str(k) in target_kjs_dict:
                psiTilde = np.zeros(np.prod(S), dtype=np.complex128)
                for j in range(nBands):
                    if j in target_kjs_dict[str(k)]:
                        psiTilde[index] = C[j]
                        psi = ifftn(np.reshape(psiTilde, S)) * np.prod(S)
                        rs_wfn += abs(psi)*weights[k, j]
    return rs_wfn

def get_rs_wfn_runner(wfnspath, iGarr, S, nBands, weights):
    rs_wfn = np.zeros(S, dtype=float)
    with open(wfnspath) as fp:
        for k, iGcur in enumerate(iGarr):
            # wkCur = wk[iState]
            nG = iGcur.shape[0]
            iGcur += np.where(iGcur<0, S[None,:], 0)
            stride = np.array([S[2]*S[1], S[2], 1], dtype=int)
            index = np.dot(iGcur, stride)
            C = np.reshape(np.fromfile(fp, dtype=np.complex128, count=nBands*nG), (nBands, nG))
            psiTilde = np.zeros(np.prod(S), dtype=np.complex128)
            for j in range(nBands):
                psiTilde[index] = C[j]
                psi = ifftn(np.reshape(psiTilde, S)) * np.prod(S)
                rs_wfn += abs(psi)*weights[k, j]
    return rs_wfn

def get_rs_wfn(calc_dir, weights=None, target_kjs_dict = None):
    outfile = opj(calc_dir, "out")
    S, R, mu = get_S_R_mu(outfile)
    Gvectors_path = opj(calc_dir, "Gvectors")
    iGarr, wk = get_iGarr_wk(Gvectors_path)
    nStates = get_nStates(wk)
    eigenvalspath = opj(calc_dir, "eigenvals")
    E = get_E(eigenvalspath, nStates)
    nBands = get_nBands(E)
    wfnspath = opj(calc_dir, "wfns")
    if weights is None:
        weights=np.ones([nStates, nBands])
    else:
        wshape = np.shape(weights)
        if not np.prod(wshape) == nStates*nBands:
            raise ValueError("Bad weights shape")
        else:
            if len(wshape) == 5:
                weights = weights.reshape([nStates, nBands])
            elif not len(wshape) == 2:
                raise ValueError("Unexpected weights dimensionality")
    if target_kjs_dict is None:
        rs_wfn = get_rs_wfn_runner(wfnspath, iGarr, S, nBands, weights)
    else:
        rs_wfn = get_rs_wfn_target_runner(wfnspath, iGarr, S, nBands, weights, target_kjs_dict)
    return rs_wfn

# def get_homo_lumo_idcs(E, mu):
#     n = np.shape(E)
#     nBands = n[1]
#     for nBand in range(nBands):
#         if E[0, nBand] > mu:
#             homo = nBand - 1
#             lumo = nBand
#             return homo, lumo
#     return None, None
#
#
# def get_rs_homo_lumo(calc_dir):
#     outfile = opj(calc_dir, "out")
#     S, R, mu = get_S_R_mu(outfile)
#     Gvectors_path = opj(calc_dir, "Gvectors")
#     iGarr, wk = get_iGarr_wk(Gvectors_path)
#     nStates = get_nStates(wk)
#     eigenvalspath = opj(calc_dir, "eigenvals")
#     E = get_E(eigenvalspath, nStates)
#     nBands = get_nBands(E)
#     wfnspath = opj(calc_dir, "wfns")
#     target_states = list(range(nStates))
#     homo_idx, lumo_idx = get_homo_lumo_idcs(E, mu)
#     _homo = get_rs_wfns_runner(wfnspath, iGarr, S, nBands, target_states, [homo_idx])
#     _lumo = get_rs_wfns_runner(wfnspath, iGarr, S, nBands, target_states, [lumo_idx])
#     homo = np.zeros(S)
#     lumo = np.zeros(S)
#     for k in range(len(target_states)):
#         homo += _homo[k, 0]
#         lumo += _lumo[k, 0]
#     return homo, lumo
#
#
# def get_homo_lumo_cubes(calc_dir, out_dir=None):
#     if out_dir is None:
#         out_dir = calc_dir
#     homo, lumo = get_rs_homo_lumo(calc_dir)
#     atoms = get_atoms_from_out(opj(calc_dir, "out"))
#     write_cube_writer(atoms, opj(out_dir, "HOMO.cub"), homo, "title card")
#     write_cube_writer(atoms, opj(out_dir, "LUMO.cub"), lumo, "title card")