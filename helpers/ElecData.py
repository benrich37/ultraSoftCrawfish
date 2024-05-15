import numpy as np
from helpers.data_parsing_helpers import get_bandprojections_from_bandfile, is_complex_bandfile, get_nSpin_helper, get_kfolding_from_outfile, get_kpts_info_handler
from helpers.data_parsing_helpers import get_E_sabcj_helper, get_mu_from_outfile, get_kmap_from_atoms
from helpers.data_parsing_helpers import get_nProj_from_bandfile, get_nBands_from_bandfile, get_nStates_from_bandfile, get_nSpecies_from_bandfile, get_nOrbsPerAtom_from_bandfile
from helpers.ase_helpers import get_atoms_from_out
from os.path import join as opj, exists as ope
from copy import deepcopy


def parse_data(root=None, bandfile="bandProjections", kPtsfile="kPts", eigfile="eigenvals", fillingsfile="fillings",
               outfile="out"):
    """
    :param bandfile: Name of BandProjections file in root (str)
    :param kPtsfile: Name of kPts file in root (str)
    :param eigfile: Name of eigenvalues file in root (str)
    :param fillingsfile: Name of fillings files in root (str)
    :param outfile: Name of out file in root (str)
    :return:
        data: ElecData
    """
    data = ElecData(root=root, bandfile=bandfile, kPtsfile=kPtsfile, eigfile=eigfile, fillingsfile=fillingsfile, outfile=outfile)
    return data

class ElecData:

    def __init__(self, root=None, bandfile="bandProjections", kPtsfile="kPts",
                 eigfile="eigenvals", fillingsfile="fillings", outfile="out"):
        self.kmap = None
        self.nProj = None # Total number of atomic projections
        self.nBands = None
        self.nOrbsPerAtom = None
        self.orbs_idx_dict = None
        self.mu = None
        self.occ_sabcj = None
        self.proj_sabcju = None
        self.E_sabcj = None
        self.nStates = None
        self.atoms = None
        self.nSpin = None
        self.alloc_file_paths(root, bandfile, kPtsfile, eigfile, fillingsfile,
                              outfile)
        self.alloc_kpt_data()
        self.alloc_elec_data()


    def get_atoms(self):
        if self.atoms is None:
            self.atoms = get_atoms_from_out(self.outfile)
        return self.atoms

    def get_ion_names(self):
        atoms = self.get_atoms()
        return atoms.get_chemical_symbols()

    def get_nSpin(self):
        if self.nSpin is None:
            self.nSpin = get_nSpin_helper(self.outfile)
        return self.nSpin

    def get_nStates(self):
        if self.nStates is None:
            self.nStates = get_nStates_from_bandfile(self.bandfile)
        return self.nStates

    def get_nBands(self):
        if self.nBands is None:
            self.nBands = get_nBands_from_bandfile(self.bandfile)
        return self.nBands

    def get_nProj(self):
        if self.nProj is None:
            self.nProj = get_nProj_from_bandfile(self.bandfile)
        return self.nProj


    def get_E_sabcj(self):
        if self.E_sabcj is None:
            eigfile = self.eigfile
            nSpin = self.get_nSpin()
            nBands = self.get_nBands()
            kfolding = self.get_kfolding()
            self.E_sabcj = get_E_sabcj_helper(eigfile, nSpin, nBands, kfolding)
        return self.E_sabcj


    def get_proj_sabcju(self):
        if self.proj_sabcju is None:
            nSpin = self.get_nSpin()
            kfolding = self.get_kfolding()
            nBands = self.get_nBands()
            nProj = self.get_nProj()
            proj_shape = [nSpin] + list(kfolding) + [nBands, nProj]
            proj_tju = self.get_proj_tju()
            self.proj_sabcju = proj_tju.reshape(proj_shape)
        return self.proj_sabcju

    def get_occ_sabcj(self):
        if self.occ_sabcj is None:
            nSpin = self.get_nSpin()
            kfolding = self.get_kfolding()
            nBands = self.get_nBands()
            occ_shape = [nSpin] + list(kfolding) + [nBands]
            if ope(self.fillingsfile):
                fillings = np.fromfile(self.fillingsfile)
            else:
                print("WARNING - no fillings file found. Setting occupations to arbitrary array")
                fillings = np.zeros(np.product(occ_shape))
            self.occ_sabcj = fillings.reshape(occ_shape)
        return self.occ_sabcj

    def get_mu(self):
        if self.mu is None:
            self.mu = get_mu_from_outfile(self.outfile)
        return self.mu


    def get_wk_sabc(self):
        if self.wk_sabc is None:
            self.alloc_kpt_data()
        return self.wk_sabc

    def get_ks_sabc(self):
        if self.ks_sabc is None:
            self.alloc_kpt_data()
        if not self.lti_allowed:
            print("WARNING - these k-points are arbitrary and don't reflect the k-points used in the DFT calculation.")
        return self.ks_sabc

    def get_kfolding(self):
        if self.kfolding is None:
            self.alloc_kpt_data()
        if not self.lti_allowed:
            print("WARNING - this k-point folding array was assigned arbitrary and doesn't reflect the k-point mesh used in the DFT calculation")
        return self.kfolding

    def get_proj_tju(self, allow_normalized=False):
        if self.proj_sabcju is None:
            self.complex_bandprojs = is_complex_bandfile(self.bandfile)
            if not self.complex_bandprojs:
                msg = "Bandprojections file contains |proj|^2, not proj - invalid data for COHP analysis \n (next time add 'band-projection-params yes no' to inputs file)"
                if allow_normalized:
                    print(msg)
                else:
                    raise ValueError(msg)
            return get_bandprojections_from_bandfile(self.bandfile, self.complex_bandprojs)
        else:
            proj_tju = deepcopy(self.proj_sabcju)
            nStates = self.get_nStates()
            nBands = self.get_nBands()
            nProj = self.get_nProj()
            proj_shape = [nStates, nBands, nProj]
            proj_tju.reshape(proj_shape)
            return proj_tju

    def get_nOrbsPerAtom(self):
        if self.nOrbsPerAtom is None:
            self.nOrbsPerAtom = get_nOrbsPerAtom_from_bandfile(self.bandfile)
        return self.nOrbsPerAtom

    def get_orbs_idx_dict(self):
        # Returns a reference dictionary mapping each atom (using key of format 'el #n' (str), where el is atom id, and n is
        # number of specific atom as it appears in JDFTx out file using 1-based indexing) to indices (int) of all
        # atomic orbital projections (in 0-based indexing) belonging to said atom.
        if self.orbs_idx_dict is None:
            ion_names = self.get_ion_names()
            els, el_counts = count_ions(ion_names)
            nOrbsPerAtom = self.get_nOrbsPerAtom()
            self.orbs_idx_dict = orbs_idx_dict_helper(els, el_counts, nOrbsPerAtom)
        return self.orbs_idx_dict

    def get_kmap(self):
        if self.kmap is None:
            atoms = self.get_atoms()
            self.kmap = get_kmap_from_atoms(atoms)
        return self.kmap

    #######

    def alloc_kpt_data(self):
        kfolding = get_kfolding_from_outfile(self.outfile)
        nSpin = self.get_nSpin()
        kPtsfile = self.kPtsfile
        nStates = self.get_nStates()
        kinfo = get_kpts_info_handler(nSpin, kfolding, kPtsfile, nStates)
        self.wk_sabc = kinfo["wk_sabc"]
        self.ks_sabc = kinfo["ks_sabc"]
        self.kfolding = kinfo["kfolding"]
        self.lti_allowed = kinfo["lti"]

    def alloc_file_paths(self, root, bandfile, kPtsfile, eigfile, fillingsfile,
                         outfile):
        self.root = root
        if not self.root is None:
            bandfile = opj(self.root, bandfile)
            kPtsfile = opj(self.root, kPtsfile)
            eigfile = opj(self.root, eigfile)
            fillingsfile = opj(self.root, fillingsfile)
            outfile = opj(self.root, outfile)
        self.bandfile = bandfile
        self.kPtsfile = kPtsfile
        self.eigfile = eigfile
        self.fillingsfile = fillingsfile
        self.outfile = outfile

    def alloc_elec_data(self):
        _ = self.get_proj_sabcju()
        _ = self.get_occ_sabcj()
        _ = self.get_E_sabcj()
        _ = self.get_mu()

    #################





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



