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
    E = np.fromfile(eigfile)
    nSpin = get_nSpin(outfile)
    wk_sabc, ks_sabc, kfolding = get_kpts_info_handler(nSpin, kfolding, kPtsfile, nStates)
    Eshape = [nSpin, kfolding[0], kfolding[1], kfolding[2], nBands]
    E_sabcj = E.reshape(Eshape)
    if ope(fillingsfile):
        fillings = np.fromfile(fillingsfile)
        occ_sabcj = fillings.reshape(Eshape)
    else:
        occ_sabcj = np.ones(Eshape)*np.nan
    proj_shape = Eshape
    proj_shape.append(nProj)
    proj_flat = proj_kju.flatten()
    proj_sabcju = proj_flat.reshape(proj_shape)
    mu = get_mu(outfile)
    return proj_sabcju, E_sabcj, occ_sabcj, wk_sabc, ks_sabc, orbs_dict, mu


def get_kpts_info_handler(nSpin, kfolding, kPtsfile, nStates):
    _nK = int(np.prod(kfolding))
    nK = int(np.prod(kfolding))
    if nSpin != int(nStates / _nK):
        print("WARNING: Internal inconsistency found (nSpin * nK-pts != nStates).")
        print("No safety net for this which allows for tetrahedral integration currently implemented.")
        print("k-folding will be changed to arbitrary length 3 array to satisfy shaping criteria.")
        nK = int(nStates / nSpin)
    if ope(kPtsfile):
        wk, ks, nStates = parse_kptsfile(kPtsfile)
        wk = np.array(wk)
        ks = np.array(ks)
        if (nK != _nK):
            if len(ks) == nK: # length of kpt data matches interpolated nK value
                kfolding = get_kfolding_from_kpts(kPtsfile, nK)
            else:
                kfolding = get_arbitrary_kfolding(nK)
                ks = np.ones([nK * nSpin, 3]) * np.nan
                wk = np.ones(nK * nSpin)
                wk *= (1 / nK)

    else:
        if nK != _nK:
            kfolding = get_arbitrary_kfolding(nK)
        ks = np.ones([nK * nSpin, 3]) * np.nan
        wk = np.ones(nK * nSpin)
        wk *= (1 / nK)
    wk_sabc = wk.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2]])
    ks_sabc = ks.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2], 3])
    return wk_sabc, ks_sabc, kfolding


def get_kfolding_from_kpts_reader(kPtsfile):
    return None

def get_arbitrary_kfolding(nK):
    kfolding = [1, 1, nK]
    assert np.prod(kfolding) == nK
    return kfolding

def get_kfolding_from_kpts(kPtsfile, nK):
    kfolding = get_kfolding_from_kpts_reader(kPtsfile)
    if kfolding is None:
        kfolding = get_arbitrary_kfolding(nK)
    return kfolding





spintype_nSpin = {
    "no-spin": 1,
    "spin-orbit": 2,
    "vector-spin": 2,
    "z-spin": 2
}
def get_nSpin(outfile):
    start = get_start_line(outfile)
    key = "spintype"
    with open(outfile, "r") as f:
        for i, line in enumerate(f):
            if i > start:
                if key in line:
                    tokens = line.strip().split()
                    val = tokens[1]
                    if val in spintype_nSpin:
                        return spintype_nSpin[val]




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

def get_start_lines(outfname, add_end=False):
    start_lines = []
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line or "Input parsed successfully" in line:
            start_lines.append(i)
        end_line = i
    if add_end:
        start_lines.append(end_line)
    return start_lines

def get_start_line(outfile):
    """ Retrieves beginning line number of most recent JDFTx calculation contained in the specified out file.
    :param outfile: Full path of JDFTx out file
    :return: integer line number of start of most recent JDFTx calculation,
    """
    start_lines = get_start_lines(outfile, add_end=False)
    return start_lines[-1]

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


def get_input_coord_vars_from_outfile(outfname):
    start_line = get_start_line(outfname)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    with open(outfname) as f:
        for i, line in enumerate(f):
            if i > start_line:
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    elif tokens[0] == "lattice":
                        active_lattice = True
                    elif active_lattice:
                        if lat_row < 3:
                            R[lat_row, :] = [float(x) for x in tokens[:3]]
                            lat_row += 1
                        else:
                            active_lattice = False
                    elif "Initializing the Grid" in line:
                        break
    return names, posns, R


def get_kmap(atoms):
    el_counter_dict = {}
    idx_to_key_map = []
    els = atoms.get_chemical_symbols()
    for i, el in enumerate(els):
        if not el in el_counter_dict:
            el_counter_dict[el] = 0
        el_counter_dict[el] += 1
        idx_to_key_map.append(f"{el} #{el_counter_dict[el]}")
    return idx_to_key_map

def get_atom_orb_labels_dict(root):
    """
    :param root: Path of directory containing JDFTx calculation data
    :return: Reference dictionary containing string representations of each atomic orbital projection
                ( NOTE - out of laziness, atoms with multiple shells of a particular angular momentum are indexed
                  starting at 0, and are not representative of their true principal quantum number)
    """
    fname = opj(root, "bandProjections")
    labels_dict = {}
    ref_lists = [
        ["s"],
        ["px", "py", "pz"],
        ["dxy", "dxz", "dyz", "dx2y2", "dz2"],
        ["fx3-3xy2", "fyx2-yz2", "fxz2", "fz3", "fyz2", "fxyz", "f3yx2-y3"]
    ]
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            if i > 1:
                if "#" in line:
                    return labels_dict
                else:
                    lsplit = line.strip().split()
                    sym = lsplit[0]
                    labels_dict[sym] = []
                    lmax = int(lsplit[3])
                    for j in range(lmax+1):
                        refs = ref_lists[j]
                        nShells = int(lsplit[4+j])
                        for k in range(nShells):
                            if nShells > 1:
                                for r in refs:
                                    labels_dict[sym].append(f"{k}{r}")
                            else:
                                labels_dict[sym] += refs


def get_el_orb_u_dict(path, atoms, orbs_dict, aidcs):
    """
    :param path: Path of directory containing JDFTx calculation data
    :param atoms: Atoms object of calculated system
    :param orbs_dict: Reference orbs dict
    :param aidcs: Indices for atoms of interest
    :return: Dictionary mapping atom symbol and atomic orbital string to all relevant projection indices
    """
    els = [atoms.get_chemical_symbols()[i] for i in aidcs]
    kmap = get_kmap(atoms)
    labels_dict = get_atom_orb_labels_dict(path)
    el_orbs_dict = {}
    for i, el in enumerate(els):
        if not el in el_orbs_dict:
            el_orbs_dict[el] = {}
        for ui, u in enumerate(orbs_dict[kmap[aidcs[i]]]):
            orb = labels_dict[el][ui]
            if not orb in el_orbs_dict[el]:
                el_orbs_dict[el][orb] = []
            el_orbs_dict[el][orb].append(u)
    return el_orbs_dict