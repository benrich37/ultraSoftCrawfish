import numpy as np
from os.path import join as opj, exists as ope


def parse_data(root=None, bandfile="bandProjections", kPtsfile="kPts", eigfile="eigenvals", fillingsfile="fillings",
               outfile="out"):
    """
    :param bandfile: Path to BandProjections file (str)
    :param gvecfile: Path to Gvectors file (str)
    :param eigfile: Path to eigenvalues file (str)
    :param guts: Whether to data not directly needed by main functions (Boolean)
    :return:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of band functions (integer)
        - nProj: the number of band projections (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure (list(int))
        - wk: A list of weight factors for each k-point (list(float))
        - k_points: A list of k-points (given as 3 floats) for each k-point. (list(list(float))
        - E: nStates by nBands array of KS eigenvalues (np.ndarray(float))
        *- iGarr: A list of numpy arrays for the miller indices of each G-vector used in the
                  expansion of each state (list(np.ndarray(int)))
    :rtype: tuple
    """
    if not root is None:
        bandfile = opj(root, bandfile)
        kPtsfile = opj(root, kPtsfile)
        eigfile = opj(root, eigfile)
        fillingsfile = opj(root, fillingsfile)
        outfile = opj(root, outfile)
    proj_kju, nStates, nBands, nProj, nSpecies, nOrbsPerAtom = jfunc.parse_complex_bandfile(bandfile)
    orbs_dict = jfunc.orbs_idx_dict(outfile, nOrbsPerAtom)
    kfolding = jfunc.get_kfolding(outfile)
    nK = int(np.prod(kfolding))
    nSpin = int(nStates / nK)
    if ope(kPtsfile):
        wk, ks, nStates = jfunc.parse_kptsfile(kPtsfile)
        wk = np.array(wk)
        ks = np.array(ks)
    else:
        ks = np.zeros([nK*nSpin, 3])
        wk = np.ones(nK*nSpin)
        wk *= (1/nK)
    wk = wk.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2]])
    ks = ks.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2], 3])
    E = np.fromfile(eigfile)
    Eshape = [nSpin, kfolding[0], kfolding[1], kfolding[2], nBands]
    E_sabcj = E.reshape(Eshape)
    fillings = np.fromfile(fillingsfile)
    occ_sabcj = fillings.reshape(Eshape)
    proj_shape = Eshape
    proj_shape.append(nProj)
    proj_flat = proj_kju.flatten()
    proj_sabcju = proj_flat.reshape(proj_shape)
    mu = jfunc.get_mu(outfile)
    return proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu