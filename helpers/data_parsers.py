import numpy as np
from os.path import join as opj, exists as ope


def parse_data(root=None, bandfile="bandProjections", kPtsfile="kPts", eigfile="eigenvals", fillingsfile="fillings",
               outfile="out"):
    """
    :param bandfile: Path to BandProjections file (str)
    :param kPtsfile: Path to kPts file (str)
    :param eigfile: Path to eigenvalues file (str)
    :param fillingsfile: Path to fillings files (str)
    :param outfile: Path to out file (str)
    :return:
        - proj_sabcju: a numpy array containing the complex band projection, data (<φ_μ|ψ_j> = T_μj) shaped to
                       (nSpin, nKa, nKb, nKc, nBand, nProj) where
                - nSpin (s) = alpha/beta
                - nKa/b/c (a/b/c) = index along kpt folding for 1st, 2nd, and 3rd reciprocal lattice vector, respectively
                - nBand (j) = index of band
                - nProj (u) = index of projected atomic orbital
        - E_sabcj: Numpy array of kohn-sham eigenvalues (float)
        - occ_sabcj: Numpy array of electronic fillings for respective state/band (float)
        - wk_sabc: Numpy array of kpt weights (float)
        - ks_sabc: Numpy array of each kpt in reciprocal space (float)
        - orbs_dict: Reference dictionary mapping each atom (using key of format 'el #n' (str), where el is atom id, and n is
                     number of specific atom as it appears in JDFTx out file using 1-based indexing) to indices (int) of all
                     atomic orbital projections (in 0-based indexing) belonging to said atom.
        - mu: Fermi level in Hartree (float)
    :rtype: tuple
    """
    if not root is None:
        bandfile = opj(root, bandfile)
        kPtsfile = opj(root, kPtsfile)
        eigfile = opj(root, eigfile)
        fillingsfile = opj(root, fillingsfile)
        outfile = opj(root, outfile)
    proj_kju, nStates, nBands, nProj, nSpecies, nOrbsPerAtom = parse_complex_bandfile(bandfile)
    orbs_dict = orbs_idx_dict(outfile, nOrbsPerAtom)
    kfolding = get_kfolding(outfile)
    nK = int(np.prod(kfolding))
    nSpin = int(nStates / nK)
    if ope(kPtsfile):
        wk, ks, nStates = parse_kptsfile(kPtsfile)
        wk = np.array(wk)
        ks = np.array(ks)
    else:
        ks = np.zeros([nK*nSpin, 3])
        wk = np.ones(nK*nSpin)
        wk *= (1/nK)
    wk_sabc = wk.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2]])
    ks_sabc = ks.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2], 3])
    E = np.fromfile(eigfile)
    Eshape = [nSpin, kfolding[0], kfolding[1], kfolding[2], nBands]
    E_sabcj = E.reshape(Eshape)
    fillings = np.fromfile(fillingsfile)
    occ_sabcj = fillings.reshape(Eshape)
    proj_shape = Eshape
    proj_shape.append(nProj)
    proj_flat = proj_kju.flatten()
    proj_sabcju = proj_flat.reshape(proj_shape)
    mu = get_mu(outfile)
    return proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks_sabc, orbs_dict, mu


def is_complex_bandfile(bandfile):
    # Returns True if specified path leads to a JDFTx bandProjections file with complex ("unnormalized") projections
    hash_lines = 0
    with open(bandfile, 'r') as f:
        for i, line in enumerate(f):
            if "#" in line:
                hash_lines += 1
                if hash_lines == 2:
                    if "|projection|^2" in line:
                        return False
                    else:
                        return True

def parse_complex_bandprojection(tokens):
    """ Should only be called for data generated by modified JDFTx
    :param tokens: Parsed data from bandProjections file
    :return out: data in the normal numpy complex data format (list(complex))
    """
    out = []
    for i in range(int(len(tokens)/2)):
        repart = tokens[2*i]
        impart = tokens[(2*i) + 1]
        num = complex(float(repart), float(impart))
        out.append(num)
    return out


def parse_complex_bandfile(bandfile):
    """ Parser function for the 'bandProjections' file produced by JDFTx
    :param bandfile: the path to the bandProjections file to parse
    :type bandfile: str
    :return: a tuple containing six elements:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of energy bands (integer)
        - nProj: the number of band projections (integer)
        - nSpecies: the number of atomic species (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure
    :rtype: tuple
    """
    if not is_complex_bandfile(bandfile):
        raise ValueError(
            "Bandprojections file contains |proj|^2, not proj - invalid data for COHP analysis \n (next time add 'band-projection-params yes no' to inputs file)")
    with open(bandfile, 'r') as f:
        for iLine, line in enumerate(f):
            tokens = line.split()
            if iLine==0:
                nStates = int(tokens[0])
                nBands = int(tokens[2])
                nProj = int(tokens[4])
                nSpecies = int(tokens[6])
                proj = np.zeros((nStates, nBands, nProj), dtype=complex)
                nOrbsPerAtom = []
            elif iLine>=2:
                if iLine<nSpecies+2:
                    nAtoms = int(tokens[1])
                    nOrbsPerAtom.extend( [int(tokens[2]),] * nAtoms)
                else:
                    iState = (iLine-(nSpecies+2)) // (nBands+1)
                    iBand = (iLine-(nSpecies+2)) - iState*(nBands+1) - 1
                    if iBand>=0 and iState<nStates:
                        proj[iState,iBand]=np.array(parse_complex_bandprojection(tokens))
    f.close()
    return proj, nStates, nBands, nProj, nSpecies, nOrbsPerAtom

def get_mu(outfile):
    # Returns fermi level in Hartree
    mu = 0
    lookkey = "FillingsUpdate:  mu:"
    with open(outfile, "r") as f:
        for line in f:
            if lookkey in line:
                mu = float(line.split(lookkey)[1].strip().split()[0])
    return mu

def count_ions(ionNames):
    # Counts number of times each atom type appears
    ion_names = []
    ion_counts = []
    for name in ionNames:
        if name not in ion_names:
            ion_names.append(name)
            ion_counts.append(ionNames.count(name))
    return ion_names, ion_counts

def orbs_idx_dict_helper(ion_names, ion_counts, nOrbsPerAtom):
    # Helper function for orbs_idx_dict function
    orbs_dict_out = {}
    iOrb = 0
    atom = 0
    for i, count in enumerate(ion_counts):
        for atom_num in range(count):
            atom_label = ion_names[i] + ' #' + str(atom_num + 1)
            norbs = nOrbsPerAtom[atom]
            orbs_dict_out[atom_label] = list(range(iOrb, iOrb + norbs))
            iOrb += norbs
            atom += 1
    return orbs_dict_out

def orbs_idx_dict(outfile, nOrbsPerAtom):
    # Returns a reference dictionary mapping each atom (using key of format 'el #n' (str), where el is atom id, and n is
    # number of specific atom as it appears in JDFTx out file using 1-based indexing) to indices (int) of all
    # atomic orbital projections (in 0-based indexing) belonging to said atom.
    ionPos, ionNames, R = get_coords_vars(outfile)
    ion_names, ion_counts = count_ions(ionNames)
    orbs_dict = orbs_idx_dict_helper(ion_names, ion_counts, nOrbsPerAtom)
    return orbs_dict

def get_coords_vars(outfile):
    """ get ionPos, ionNames, and R from outfile
    :param outfile: Path to output file (str)
    :return:
        - ionPos: ion positions in lattice coordinates (np.ndarray(float))
        - ionNames: atom names (list(str))
        - R: lattice vectors (np.ndarray(float))
    :rtype: tuple
    """
    start = get_start_line(outfile)
    iLine = 0
    refLine = -10
    R = np.zeros((3, 3))
    Rdone = False
    ionPosStarted = False
    ionNames = []
    ionPos = []
    for i, line in enumerate(open(outfile)):
        if i > start:
            # Lattice vectors:
            if line.find('Initializing the Grid') >= 0 and (not Rdone):
                refLine = iLine
            rowNum = iLine - (refLine + 2)
            if rowNum >= 0 and rowNum < 3:
                R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
            if rowNum == 3:
                refLine = -10
                Rdone = True
            # Coordinate system and ionic positions:
            if ionPosStarted:
                tokens = line.split()
                if len(tokens) and tokens[0] == 'ion':
                    ionNames.append(tokens[1])
                    ionPos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
                else:
                    break
            if line.find('# Ionic positions in') >= 0:
                coords = line.split()[4]
                ionPosStarted = True
            # Line counter:
            iLine += 1
    ionPos = np.array(ionPos)
    if coords != 'lattice':
        ionPos = np.dot(ionPos, np.linalg.inv(R.T))  # convert to lattice
    return ionPos, ionNames, R

def get_start_line(outfile):
    """ Retrieves beginning line number of most recent JDFTx calculation contained in the specified out file.
    :param outfile: Full path of JDFTx out file
    :return: integer line number of start of most recent JDFTx calculation,
    """
    start = None
    for i, line in enumerate(open(outfile)):
        if ('JDFTx' in line) and ('***' in line):
            start = i
    if start is None:
        for i, line in enumerate(open(outfile)):
            if ("Input parsed successfully" in line):
                start = i
    return start

def get_kfolding(outfile):
    """ Returns kpt foldings
    :param outfile: Path to out file (str)
    :return: Numpy array of shape (3,) of ints containing kpt folding
    """
    key = "kpoint-folding "
    with open(outfile, "r") as f:
        for i, line in enumerate(f):
            if key in line:
                val = np.array(line.split(key)[1].strip().split(), dtype=int)
                return val

def parse_kptsfile(kptsfile):
    """ Parser for kPts output file
    :param kptsfile: path to kPts file
    :return:
        - wk_list: List of kpt weights
        - k_points_list: List of each kpt in reciprocal space
        - nStates: Number of states (nSpin x nKa x nKb x nKc)
    :rtype: tuple
    """
    wk_list = []
    k_points_list = []
    with open(kptsfile, "r") as f:
        for line in f:
            k_points = line.split("[")[1].split("]")[0].strip().split()
            k_points = [float(v) for v in k_points]
            k_points_list.append(k_points)
            wk = float(line.split("]")[1].strip().split()[0])
            wk_list.append(wk)
    nStates = len(wk_list)
    return wk_list, k_points_list, nStates