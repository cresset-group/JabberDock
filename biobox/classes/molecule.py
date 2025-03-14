# Copyright (c) 2014-2017 Matteo Degiacomi
#
# BiobOx is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# BiobOx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with BiobOx ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Matteo Degiacomi, matteothomas.degiacomi@gmail.com

import os
from copy import deepcopy
import numpy as np
import scipy.signal
import pandas as pd
import warnings

# Definiton of constants for later calculations
epsilon0 = 8.8542 * 10**(-12) # m**-3 kg**-1 s**4 A**2, Permitivitty of free space
kB = 1.3806 * 10**(-23) # m**2 kg s**-2 K-1, Lattice Boltzmann constant
e = 1.602 * 10**(-19) # A s, electronic charge
m = 1 * 10**(-9) # number of nm in 1 m
c = 3.336 * 10**(-30) # conversion from debye to e m
Na = 6.022 * 10**(23) # Avagadros Number

from biobox.classes.structure import Structure
from biobox.lib import e_density

class Molecule(Structure):
    '''
    Subclass of :func:`Structure <structure.Structure>`, allows reading, manipulating and analyzing molecular structures.
    '''

    chain_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd',
                   'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                   'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    def __init__(self):
        '''
        At instantiation, properties associated to every individual atoms are stored in a pandas Dataframe self.data.
        The columns of the self.data have the following names:
        atom, index, name, resname, chain, resid, beta, occupancy, atomtype, radius, charge.

        self.knowledge contains a knowledge base about atoms and residues properties. Default values are:

        * 'residue_mass' property stores the average mass for most common aminoacids (values from Expasy website)
        * 'atom_vdw' vdw radius of common atoms
        * 'atom_mass' mass of common atoms

        The knowledge base can be edited. For instance, to add information about residue "TST" mass in molecule M type: M.knowledge['mass_residue']["TST"]=142.42
        '''

        super(Molecule, self).__init__(r=np.array([]))

        # knowledge base about atoms and residues properties (entry keys:
        # 'residue_mass', 'atom_vdw', 'atom_, mass' can be edited)
        self.knowledge = {}
        self.knowledge['residue_mass'] = {"ALA": 71.0788, "ARG": 156.1875, "ASN": 114.1038, "ASP": 115.0886, "CYS": 103.1388, "CYX": 103.1388, "GLU": 129.1155, "GLN": 128.1307, "GLY": 57.0519,
                                          "HIS": 137.1411, "HSE": 137.1411, "HSD": 137.1411, "HSP": 137.1411, "HIE": 137.1411, "HID": 137.1411, "HIP": 137.1411, "ILE": 113.1594, "LEU": 113.1594,
                                          "LYS": 128.1741, "MET": 131.1926, "MSE": 131.1926, "PHE": 147.1766, "PRO": 97.1167, "SER": 87.0782, "THR": 101.1051, "TRP": 186.2132, "TYR": 163.1760, "VAL": 99.1326}
        self.knowledge['atom_vdw'] = {'H': 1.20, 'N': 1.55, 'NA': 2.27, 'CU': 1.40, 'CL': 1.75, 'C': 1.70, 'O': 1.52, 'I': 1.98, 'P': 1.80, 'B': 1.85, 'BR': 1.85, 'S': 1.80, 'SE': 1.90,
                                      'F': 1.47, 'FE': 1.80, 'K': 2.75, 'MN': 1.73, 'MG': 1.73, 'ZN': 1.39, 'HG': 1.8, 'XE': 1.8, 'AU': 1.8, 'LI': 1.8, '.': 1.8}
        self.knowledge['atom_ccs'] = {'H': 1.2, 'C': 1.91, 'N': 1.91, 'O': 1.91, 'P': 1.91, 'S': 1.91, '.': 1.91}
        self.knowledge['atom_mass'] = {"H": 1.00794, "D": 2.01410178, "HE": 4.00, "LI": 6.941, "BE": 9.01, "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.998403, "NE": 20.18, "NA": 22.989769,
                                       "MG": 24.305, "AL": 26.98, "SI": 28.09, "P": 30.973762, "S": 32.065, "CL": 35.453, "AR": 39.95, "K": 39.0983, "CA": 40.078, "SC": 44.96, "TI": 47.87, "V": 50.94,
                                       "CR": 51.9961, "MN": 54.938045, "FE": 55.845, "CO": 58.93, "NI": 58.6934, "CU": 63.546, "ZN": 65.409, "GA": 69.72, "GE": 72.64, "AS": 74.9216, "SE": 78.96,
                                       "BR": 79.90, "KR": 83.80, "RB": 85.47, "SR": 87.62, "Y": 88.91, "ZR": 91.22, "NB": 92.91, "MO": 95.94, "TC": 98.0, "RU": 101.07, "RH": 102.91, "PD": 106.42,
                                       "AG": 107.8682, "CD": 112.411, "IN": 114.82, "SN": 118.71, "SB": 121.76, "TE": 127.60, "I": 126.90447, "XE": 131.29, "CS": 132.91, "BA": 137.33, "PR": 140.91,
                                       "EU": 151.96, "GD": 157.25, "TB": 158.93, "W": 183.84, "IR": 192.22, "PT": 195.084, "AU": 196.96657, "HG": 200.59, "PB": 207.2, "U": 238.03}
        self.knowledge['atomtype'] = {"C": "C", "CA": "C", "CB": "C", "CG": "C", "CG1": "C", "CG2": "C", "CZ": "C", "CD1": "C", "CD2": "C",
                                      "CD": "C", "CE": "C", "CE1": "C", "CE2": "C", "CE3": "C", "CZ2": "C", "CZ3": "C", "CH2": "C",
                                      "N": "N", "NH1": "N", "NH2": "N", "NZ": "N", "NE": "N", "NE1": "N", "NE2": "N", "ND1": "N", "ND2": "N",
                                      "O": "O", "OG": "O", "OG1": "O", "OG2": "O", "OD1": "O", "OD2": "O", "OE1": "O", "OE2": "O", "OH": "O", "OXT": "O",
                                      "SD": "S", "SG": "S", "H": "H", "HA": "H", "HB1": "H", "HB2": "H", "HE1": "H", "HE2": "H", "HD1": "H", "HD2": "H", 
                                      "H1": "H", "H2": "H", "H3": "H", "HH11": "H", "HH12": "H", "HH21": "H", "HH22": "H", "HG1": "H", "HG2": "H", "HE21": "H", 
                                      "HE22": "H", "HD11": "H", "HD12": "H", "HD13": "H", "HD21": "H", "HD22": "H", "HG11": "H", "HG12": "H", "HG13": "H", 
                                      "HG21": "H", "HG22": "H", "HG23": "H", "HZ2": "H", "HZ3": "H", "HZ": "H", "HA1": "H", "HA2": "H", "HB": "H", "HD3": "H", 
                                      "HG": "H", "HZ1": "H", "HE3": "H", "HB3": "H", "HH1": "H", "HH2": "H", "HD23": "H", "HD13": "H", "HE": "H", "HH": "H", 
                                      "OC1": "O", "OC2": "O", "OW": "O", "HW1": "H", "HW2": "H", "CH3" : "C", "HH31" : "H", "HH32" : "H", "HH33" : "H",
                                      "C00" : "C", "C01" : "C", "C02" : "C", "C04" : "C", "C06" : "C", "C08" : "C", "H03" : "H", "H05" : "H", "H07" : "H",
                                      "H09" : "H", "H0A" : "H", "H0B" : "H", "N01" : "N", "C03": "C", "C05": "C", "O06": "O", "H08": "H", "H0C": "H", "H0D": "H", 
                                      "H0E": "H", "H0F": "H", "O03": "O", "H04": "H", "H06": "H", "OD": "O", "O02" : "O", "HO" : "H", "OT" : "O", "O1" : "O", "O2" : "O"}

    def __add__(self, other):
        from biobox.classes.multimer import Multimer
        M = Multimer()
        M.load_list([self, other], ["A", "B"])
        M2 = M.make_molecule()
        return M2


    def know(self, prop):
        '''
        return information from knowledge base

        :param prop: desired property to extract from knowledge base
        :returns: value associated to requested property, or nan if failed
        '''
        if str(prop) in self.knowledge:
            return self.knowledge[str(prop)]
        else:
            raise Exception("entry %s not found in knowledge base!" % prop)

    def import_pdb(self, pdb, include_hetatm=False):
        '''
        read a pdb (possibly containing containing multiple models).

        Models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms.
        After loading, the first model (M.current_model=0) will be set as active.

        :param pdb: PDB filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        '''

        try:
            f_in = open(pdb, "r")
        except Exception as ex:
            raise Exception('ERROR: file %s not found!' % pdb)

        # store filename
        self.properties["filename"] = pdb

        data_in = []
        p = []
        r = []
        e = []
        alternative = []
        biomt = []
        symm = []
        for line in f_in:
            record = line[0:6].strip()

            # load biomatrix, if any is present
            if "REMARK 350   BIOMT" in line:
                try:
                    biomt.append(line.split()[4:8])
                except Exception as ex:
                    raise Exception("ERROR: biomatrix format seems corrupted")

            # load symmetry matrix, if any is present
            if "REMARK 290   SMTRY" in line:
                try:
                    symm.append(line.split()[4:8])
                except Exception as ex:
                    raise Exception("ERROR: symmetry matrix format seems corrupted")

            # if a complete model was parsed store all the saved data into
            # self.data entries (if needed) and temporary alternative
            # coordinates list
            if record == "ENDMDL" or record == "END":

                if len(alternative) == 0:

                    # load all the parsed data in superclass data (Dataframe)
                    # and points data structures
                    try:
                        #building dataframe
                        data = np.array(data_in).astype(str)
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)
                        # Set the index numbers to the idx values to avoid hexadecimal counts
                        self.data["index"] = idx

                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pdb)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pdb)

                    # save default charge state
                    self.data['charge'] = np.array(e)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception as ex:
                    raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' % pdb)

            if record == 'ATOM' or (include_hetatm and record == 'HETATM'):

                # extract xyz coordinates (save in list of point coordinates)
                p.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                # if no complete model has been yet parsed, load also
                # information about atoms(resid, resname, ...)
                if len(alternative) == 0:
                    w = []
                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:12].strip())  # extract atom index
                    w.append(line[12:17].strip())  # extract atomname
                    w.append(line[17:21].strip())  # extract resname
                    w.append(line[21].strip())  # extract chain name
                    w.append(line[22:26].strip())  # extract residue id

                    # extract occupancy
                    try:
                        w.append(float(line[54:60]))
                    except Exception as ex:
                        w.append(1.0)

                    # extract beta factor
                    try:
                        # w.append("{0.2f}".format(float(line[60:66])))
                        w.append(float(line[60:66]))
                    except Exception as ex:
                        w.append(0.0)

                    # extract atomtype
                    try:
                        w.append(line[76:78].strip())
                    except Exception as ex:
                        w.append("")

                    # use atomtype to extract vdw radius
                    try:
                        r.append(self.know('atom_vdw')[line[76:78].strip()])
                    except Exception as ex:
                        r.append(self.know('atom_vdw')['.'])

                    # assign default charge state of 0
                    e.append(0.0)

                    data_in.append(w)

        f_in.close()

        # if p list is not empty, that means that the PDB file does not finish with an END statement (like the ones generated by SBT, for instance).
        # In this case, dump all the remaining stuff into alternate coordinates
        # array and (if needed) into properties dictionary.
        if len(p) > 0:

            # if no model has been yet loaded, save also information in
            # properties dictionary.
            if len(alternative) == 0:

                # load all the parsed data in superclass properties['data'] and
                # points data structures
                try:
                    #building dataframe
                    data = np.array(data_in).astype(str)
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)
                    # Set the index numbers to the idx values to avoid hexadecimal counts
                    self.data["index"] = idx

                except Exception as ex:
                    raise Exception('ERROR: something went wrong when saving data in %s!\nERROR: are all the columns separated?' %pdb)

                try:
                    self.data['radius'] = np.array(r)
                except Exception as ex:
                    raise Exception('ERROR: something went wrong when saving van der Waals radii in %s!\nERROR: are all the columns separated?' % pdb)

                # save default charge state
                self.properties['charge'] = np.array(e)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception as ex:
                raise Exception('ERROR: something went wrong when saving coordinates in %s!\nERROR: are all the columns separated?' %pdb)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception as e:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                print('WARNING: found %s models, but their atom count differs' % len(alternative))
                print('WARNING: treating only the first model in file %s' % pdb)
                #raise Exception('ERROR: models appear not to have the same amount of atoms')

            self.add_xyz(alternative_xyz)
        else:
            raise Exception('ERROR: something went wrong when saving alternative coordinates in %s!\nERROR: no model was loaded... are ENDMDL statements there?' % pdb)

        # if biomatrix information is provided, creat
        if len(biomt) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(biomt), 3):
                raise Exception('ERROR: found %s BIOMT entries. A multiple of 3 is expected'%len(biomt))

            b = np.array(biomt).astype(float).reshape((len(biomt) / 3, 3, 4))
            self.properties["biomatrix"] = b

        # if symmetry information is provided, create entry in properties
        if len(symm) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(symm), 3):
                raise Exception('ERROR: found %s SMTRY entries. A multiple of 3 is expected'%len(symm))

            b = np.array(symm).astype(float).reshape((len(symm) / 3, 3, 4))
            self.properties["symmetry"] = b

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)

    def import_pqr(self, pqr, include_hetatm=False):
        '''
        Read a pqr (possibly containing containing multiple models).

        models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms.
        After loading, the first model (M.current_model=0) will be set as active.

        :param pqr: PQR filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        '''

        try:
            f_in = open(pqr, "r")
        except Exception as ex:
            raise Exception('ERROR: file %s not found!' % pqr)

        # store filename
        self.properties["filename"] = pqr

        data_in = []
        p = []  # collects coordinates for every model
        r = []  # vdW radii
        e = []  # electrostatics
        alternative = []
        for line in f_in:
            record = line[0:6].strip()
            # if a complete model was parsed store all the saved data into
            # self.properties entries (if needed) and temporary alternative
            # coordinates list
            if record == "ENDMDL" or record == "END":
                if len(alternative) == 0:
                    # load all the parsed data in superclass properties['data']
                    # and points data structures
                    try:
                        #building dataframe
                        data = np.array(data_in).astype(str)
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)
                        self.data["index"] = idx # convert to internal numbering system

                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pqr)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pqr)

                    # saving electrostatics
                    try:
                        self.data['charge'] = np.array(e)
                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' % pqr)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception as ex:
                    raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pqr)

            if record == 'ATOM' or (include_hetatm and record == 'HETATM'):

                # extract xyz coordinates (save in list of point coordinates)
                p.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                # if no complete model has been yet parsed, load also
                # information about atoms(resid, resname, ...)
                if len(alternative) == 0:

                    # extract charge
                    try:
                        # 54 is separator, 55 is plus/minus
                        e.append(float(line[54:62]))
                    except Exception as ex:
                        e.append(0.0)

                    # extract vdW radius
                    try:
                        r.append(float(line[62:69]))
                    except Exception as ex:
                        r.append(self.know('atom_vdw')['.'])

                    # initialize list
                    w = []

                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:11].strip())  # extract atom index
                    w.append(line[12:17].strip())  # extract atomname
                    w.append(line[17:20].strip())  # extract resname
                    w.append(line[21].strip())  # extract chain name
                    w.append(line[22:26].strip())  # extract residue id

                    # extract occupancy
                    w.append('1')

                    # extract beta factor
                    w.append('0')

                    # extract atomtype from atomname in BMRB notation
                    # http://www.bmrb.wisc.edu/ref_info/atom_nom.tbl
                    w.append(line[12:17].strip()[0])
                    # w.append(line[76:78].strip())

                    data_in.append(w)

        f_in.close()

        # if p list is not empty, that means that the pqr file does not finish with an END statement (like the ones generated by SBT, for instance).
        # In this case, dump all the remaining stuff into alternate coordinates
        # array and (if needed) into properties dictionary.
        if len(p) > 0:

            # if no model has been yet loaded, save also information in
            # properties dictionary.
            if len(alternative) == 0:

                # load all the parsed data in superclass properties['data'] and
                # points data structures
                try:
                    #building dataframe
                    data = np.array(data_in).astype(str)
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "occupancy", "beta", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)
                    self.data["index"] = idx # convert to internal numbering system

                except Exception as ex:
                    raise Exception('ERROR: something went wrong when saving data in %s!\nERROR: are all the columns separated?' % pqr)

                try:
                    self.data['radius'] = np.array(r)
                except Exception as ex:
                    raise Exception('ERROR: something went wrong when saving van der Waals radii in %s!\nERROR: are all the columns separated?' %pqr)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception as ex:
                raise Exception('ERROR: something went wrong when saving coordinates in %s!\nERROR: are all the columns separated?' %pqr)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception as ex:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                print('WARNING: found %s models, but their atom count differs' % len(alternative))
                print('WARNING: treating only the first model in file %s' % pqr)
                #raise Exception('ERROR: models appear not to have the same amount of atoms')

            self.add_xyz(alternative_xyz)
        else:
            raise Exception('ERROR: something went wrong when saving alternative coordinates in %s!\nERROR: no model was loaded... are ENDMDL statements there?' % pqr)

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)

    def assign_atomtype(self):
        '''
        guess atomtype from atom names
        '''
        
        a_type = []
        for i in range(0, len(self.data), 1):
            atom = self.data["name"].values[i]
            a_type.append(atom[0])

        self.data["atomtype"] = a_type

    def get_data(self, indices=[], columns=[]):
        '''
        Return information about atoms of interest (i.e., slice the data DataFrame)

        :param indices: list of indices, if not provided all atom data is returned
        :param columns: list of columns (e.g. ["resname", "resid", "chain"]), if not provided all columns are returned
        :returns: numpy array containing a slice of molecule's data
        '''

        if len(indices) == 0 and len(columns) == 0:
            return self.data.values

        elif len(indices) == 0 and len(columns) != 0:
            return self.data[columns].values

        elif len(indices) != 0 and len(columns) == 0:
            return self.data.ix[indices].values

        else:
            return self.data.ix[indices, columns].values   


    def set_data(self, value, indices=[], columns=[]):
        '''
        Return information about atoms of interest (i.e., slice the data DataFrame)

        :param indices: list of indices, if not provided all atom data is returned
        :param columns: list of columns (e.g. ["resname", "resid", "chain"]), if not provided all columns are returned
        :returns: numpy array containing a slice of molecule's data
        '''

        if len(indices) == 0 and len(columns) == 0:
            raise Exception("indices, columns or both should be provided")

        elif len(indices) == 0 and len(columns) != 0:
            self.data[columns] = value

        elif len(indices) != 0 and len(columns) == 0:
            self.data.loc[indices] = value

        else:
            self.data.loc[indices, columns] = value


    def query(self, query_text, get_index=False):
        '''
        Select specific atoms in a multimer un the basis of a text query.

        :param query_text: string selecting atoms of interest. Uses the pandas query syntax, can access all columns in the dataframe self.data.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points (in a unique array) and, if get_index is set to true, a list of their indices in subunits' self.points array.
        '''

        idx = self.data.query(query_text).index.values

        if get_index:
            return [self.points[idx], idx]
        else:
            return self.points[idx]


    def atomselect(self, chain, res, atom, get_index=False, use_resname=False):
        '''
        Select specific atoms in the protein providing chain, residue ID and atom name.

        :param chain: selection of a specific chain name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param res: residue ID of desired atoms (accepts * as wildcard). Can also be a list or numpy array of of int.
        :param atom: name of desired atom (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        # chain name boolean selector
        if isinstance(chain, str):
            if chain == '*':
                chain_query = np.array([True] * len(self.points))
            else:
                chain_query = self.data["chain"].values == chain
                
        elif isinstance(chain, list) or type(chain).__module__ == 'numpy':
            chain_query = self.data["chain"].values == chain[0]
            for c in range(1, len(chain), 1):
                chain_query = np.logical_or(chain_query, self.data["chain"].values == chain[c])
        else:
            raise Exception("ERROR: wrong type for chain selection. Should be str, list, or numpy")

        if isinstance(res, str):
            if res == '*':
                res_query = np.array([True] * len(self.points))
            elif use_resname:
                res_query = self.data["resname"].values == res
            else:
                res_query = self.data["resid"].values == res

        elif isinstance(res, int):
            if use_resname:
                res_query = self.data["resname"].values == str(res)
            else:
                res_query = self.data["resid"].values == res

        elif isinstance(res, list) or type(res).__module__ == 'numpy':
            if use_resname:
                res_query = self.data["resname"].values == str(res[0])
            else:
                res_query = self.data["resid"].values == res[0]

            for r in range(1, len(res), 1):
                if use_resname:
                    res_query = np.logical_or(res_query, self.data["resname"].values == str(res[r]))
                else:
                    res_query = np.logical_or(res_query, self.data["resid"].values == res[r])

        else:
            raise Exception("ERROR: wrong type for resid selection. Should be int, list, or numpy")

        # atom name boolean selector
        if isinstance(atom, str):
            if atom == '*':
                atom_query = np.array([True] * len(self.points))
            else:
                atom_query = self.data["name"].values == atom
        elif isinstance(atom, list) or type(atom).__module__ == 'numpy':
            atom_query = self.data["name"].values == atom[0]
            for a in range(1, len(atom), 1):
                atom_query = np.logical_or(atom_query, self.data["name"].values == atom[a])
        else:
            raise Exception("ERROR: wrong type for atom selection. Should be str, list, or numpy")

        # slice data array and return result (colums 5 to 7 contain xyz coords)
        query = np.logical_and(np.logical_and(chain_query, res_query), atom_query)


        if get_index:
            return [self.points[query], np.where(query == True)[0]]
        else:
            return self.points[query]

    def atomignore(self, chain, res, atom, get_index=False, use_resname=False):
        '''
        Select specific atoms that do not match a specific query (chain, residue ID and atom name).
        Useful to remove from a molecule atoms unwanted for further analysis, alternative conformations, etc...

        :param chain: chain name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param res: residue ID (accepts * as wildcard). Can also be a list or numpy array of of int.
        :param atom: atom name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of atoms in self.points array (and self.data)
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points not matching the query, if get_index is set to true, their indices in self.points array.
        '''

        #extract indices of atoms matching the query
        idxs = self.atomselect(chain, res, atom, get_index=True, use_resname=use_resname)[1]

        #invert the selection
        idxs2 = []
        for i in range(len(self.points)):
            if i not in idxs:
                idxs2.append(i)

        if get_index:
            return [self.points[idxs2], np.array(idxs2)]
        else:
            return self.points[idxs2]

    def same_residue(self, index, get_index=False):
        '''
        Select atoms having the same residue and chain as a given atom (or list of atoms)

        :param index indices: of atoms of choice (integer of list of integers)
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        D = self.data.values
        l = D[index]

        if len(l.shape) == 1:
            l = l.reshape(1, len(l))

        test = np.logical_and(D[:, 4] == l[:, 4], D[:, 5] == l[:, 5])

        idxs = np.where(test)[0]
        if len(idxs) > 0:
            pts = self.points[idxs]
        else:
            pts = []

        if get_index:
            return pts, idxs
        else:
            return pts

    def same_residue_unique(self, index, get_index=False):
        '''
        Select atoms having the same residue and chain as a given atom (or list of atoms)

        :param index: indices of atoms of choice (integer of list of integers)
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        try:
            test = len(index)  # this should fail if index is a number
            idlist = index
        except Exception as e:
            idlist = [index]

        D = self.data.values
        pts = []
        idxs = []
        for i in idlist:
            done = False
            j = 0  # starting from same point
            while not done:

                if i - j < 0:
                    done = True

                elif D[i, 4] == D[i - j, 4] and D[i, 5] == D[i - j, 5]:

                    if len(idxs) != 0 and i - j not in idxs:
                        pts.append(self.points[i - j])
                        idxs.append(i - j)
                    elif i - j not in idxs:
                        pts = [self.points[i - j].copy()]
                        idxs = [i - j]

                    j += 1

                else:
                    done = True

            j = 1
            done = False
            while not done:

                if i + j == len(self.points):
                    done = True

                elif D[i, 4] == D[i + j, 4] and D[i, 5] == D[i + j, 5]:

                    if len(idxs) != 0 and i + j not in idxs:
                        pts.append(self.points[i + j])
                        idxs.append(i + j)
                    elif i + j not in idxs:
                        pts = [self.points[i + j].copy()]
                        idxs = [i + j]

                    j += 1

                else:
                    done = True

        if get_index:
            return np.array(pts), np.array(idxs)
        else:
            return np.array(pts)

    def get_subset(self, idxs, conformations=[], flip = False):
        '''
        Return a :func:`Molecule <molecule.Molecule>` object containing only the selected atoms and frames

        :param ixds: atoms to extract
        :param conformations: frames to extract (by default, all)
        :param flip: If true, extract atoms that DON'T match idxs (default is False)
        :returns: :func:`Molecule <molecule.Molecule>` object
        '''

        if flip:
            self_index = set(self.data["index"])
            idxs_flip = set(idxs)
            idxs = np.asarray(list(self_index - idxs_flip) + list(idxs_flip - self_index)) # replace idxs with new keep list

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        idx = np.arange(len(idxs))

        # create molecule, and push created data information
        M = Molecule()
        postmp = self.coordinates[:, idxs]
        M.coordinates = postmp[frames]
        M.data = self.data.loc[idxs]
        M.data = M.data.reset_index(drop=True)
        M.data["index"] = idx
        M.current = 0
        M.points = M.coordinates[M.current]

        M.properties['center'] = M.get_center()

        return M

    def guess_chain_split(self, distance=3, use_backbone=True):
        '''
        reassign chain name, using distance cutoff (cannot be undone).
        If two consecutive atoms are beyond a cutoff, a new chain is assigned.

        :param distance: distance cutoff distanceR: no atomtype found!

        :param use_backbone: if True, splitting will be performed considering backbone atoms (N and C), all atoms in a sequence otherwise
        '''

        # wipe current chain assignment
        self.data["chain"] = ""

        # identify different chains
        intervals = [0]

        gaps = []
        if not use_backbone:
            for i in range(len(self.coordinates[0]) - 1):
                dist = np.sqrt(np.dot(self.points[i] - self.points[i + 1], self.points[i] - self.points[i + 1]))
                if dist > distance:
                    intervals.append(i + 1)

        else:
            #aminoacids start with N. Find where a C is too far from the next N.
            posN, idxN = self.atomselect("*", "*", "N", get_index=True)
            posC = self.atomselect("*", "*", "C")

            if len(posN) != len(posC):
                #raise Exception("mismatch in N and C count")
                warnings.warn("mismatch in N and C count")

            for i in range(len(idxN)-1):
                dist = np.sqrt(np.dot(posC[i] - posN[i+1], posC[i] - posN[i+1]))
                if dist > distance:
                    intervals.append(idxN[i+1])
                    gaps.append(dist)

        intervals.append(len(self.coordinates[0]))

        # separate chains
        for i in range(len(intervals) - 1):
            thepos = i % len(self.chain_names)
            self.data.loc[intervals[i]:intervals[i + 1], "chain"] = self.chain_names[thepos]

        return len(intervals) - 1, intervals, np.round(np.array(gaps), decimals=3)

    def get_pdb_data(self, index=[]):
        '''
        aggregate data and point coordinates, and return in a unique data structure

        Returned data is a list containing strings for points data and floats for point coordinates
        in the same order as a pdb file, i.e.
        ATOM/HETATM, index, name, resname, chain name, residue ID, x, y, z, occupancy, beta factor, atomtype.

        :returns: list aggregated data and coordinates for every point, as string.
        '''

        if len(index) == 0:
            index = range(0, len(self.points), 1)

        # create a list containing all infos contained in pdb (point
        # coordinates and properties)
        d = []
        for i in index:
            d.append([self.data["atom"].values[i],
                      self.data["index"].values[i],
                      self.data["name"].values[i],
                      self.data["resname"].values[i],
                      self.data["chain"].values[i],
                      self.data["resid"].values[i],
                      self.points[i, 0],
                      self.points[i, 1],
                      self.points[i, 2],
                      self.data["beta"].values[i],
                      self.data["occupancy"].values[i],
                      self.data["atomtype"].values[i]])

        return d

    def write_pdb(self, outname, conformations=[], index=[]):
        '''
        overload superclass method for writing (multi)pdb.

        :param outname: name of pdb file to be generated.
        :param index: indices of atoms to write to file. If empty, all atoms are returned. Index values obtaineable with a call like: index=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, a multipdb with all conformations will be produced.
        '''

        # store current frame, so it will be reestablished after file output is
        # complete
        currentbkp = self.current

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        f_out = open(outname, "w")

        for f in frames:
            # get all informations from PDB (for current conformation) in a list
            self.set_current(f)
            d = self.get_pdb_data(index)
            
            # Build our hexidecimal array if num. of atoms > 99999
            idx_val = np.arange(1, len(d) + 1, 1)
            if len(idx_val) > 99999:
                vhex = np.vectorize(hex)
                idx_val = vhex(idx_val)   # convert index values to hexidecimal
                idx_val = [num[2:] for num in idx_val]  # remove 0x at start of hexidecimal number
            
            for i in range(0, len(d), 1):
                # create and write PDB line
                if d[i][2][0].isdigit():
                    L = '%-6s%5s  %-4s%-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                elif len(d[i][2]) == 4:
                    L = '%-6s%5s  %-4s%-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                else:
                    L = '%-6s%5s  %-3s %-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                f_out.write(L)

            f_out.write("END\n")

        f_out.close()

        self.set_current(currentbkp)

        return

    def get_mass_by_atom(self, skip_resname=[]):
        '''
        compute protein mass using atoms in pdb

        sum the mass of all atoms (using a knowledge base of atom masses in Dalton)
        The knowledge base can be expanded or edited by adding or editing entries to the molecule's mass dictionary, e.g. to add the atom "PI" mass in molecule M type: M.knowledge['atom_mass']["PI"]=3.141592

        :param skip_resname: list of resnames to skip. Useful to exclude ions water or other ligands from the calculation.
        :returns: mass of molecule in Dalton
        '''

        mass = 0
        for i in range(0, len(self.data), 1):
            resname = self.data["resname"].values[i]
            atomtype = self.data["atomtype"].values[i]

            if resname not in skip_resname:
                try:
                    mass += self.know('atom_mass')[atomtype]
                except Exception as e:
                    if atomtype == "":
                        print(self.data.values[i:i+40])
                        raise Exception("ERROR: no atomtype found!")
                    else:
                        raise Exception("ERROR: mass for atom %s is unknown!\nInsert a key in protein\'s masses dictionary knowledge['atom_mass'] and retry!\nex.: protein.knowledge['atom_mass'][\"PI\"]=3.141592" %atomtype)

        return mass

    def match_residue(self, M2, sec = 5):
        '''
        Compares two bb.Molecule() peptide strands and returns the resids within both peptides when the two are homogenous
        beyond a certain secondary structure threashold. The default is 5 amino acids (given by sec) in a row must be identical

        Useful when aligning PDB structures that have been crystallised separately - so one may be missing the odd residue
        or have a few extra at the end.

        :param M2: The second bb.Molecule() to compare with
        :param sec: Number of consecutive amino acids in a row that must match before resid's are recorded
        '''
        # First run the match residue using the expected inputs
        M1_res, M2_res = self._match_residue_maths(M2, sec = sec)

        # Import a check to see if we've correctly counted the residues (e.g. in homodimer case)
        if np.shape(np.unique(M1_res)) != np.shape(np.unique(M2_res)):
            M2_res, M1_res = M2._match_residue_maths(self, sec = sec)

        return M1_res, M2_res


    def _match_residue_maths(self, M2, sec):
        '''
        Does the maths for match_residue. The reason for this additional step is that sometimes
        if the numbering is a bit off between the different proteins (i.e. a shift in the initial
        starting residues) and the protein is a homodimer, we can end up only adding the second
        monomer unit, and sometimes add it twice. Therefore we can compare two runs of this
        for an answer.

        :param M2: The second bb.Molecule() to compare with
        :param sec: Number of consecutive amino acids in a row that must match before resid's are recorded
        '''

        # Get residue names / unique IDs
        M1_reslist = self.data["resname"][self.data["name"] == 'CA'].values
        M2_reslist = M2.data["resname"][M2.data["name"] == 'CA'].values
        M1_resid = self.data["resid"][self.data["name"] == 'CA'].values
        M2_resid = M2.data["resid"][M2.data["name"] == 'CA'].values

        # Remove C or N prefixes
        for cnt, val in enumerate(M1_reslist):
            if len(val) == 4:
                M1_reslist[cnt] = val[1:]
            else:
                continue
        for cnt, val in enumerate(M2_reslist):
            if len(val) == 4:
                M2_reslist[cnt] = val[1:]
            else:
                continue

        # Rename residues temporararily so they match better
        M1_reslist[np.logical_or(np.logical_or(M1_reslist == 'HIE', M1_reslist == 'HIP'), M1_reslist == 'HID')] = 'HIS'
        M2_reslist[np.logical_or(np.logical_or(M2_reslist == 'HIE', M2_reslist == 'HIP'), M2_reslist == 'HID')] = 'HIS'
        
        M1_reskeep = []
        M2_reskeep = []
        M2_cnt = 0
        M1_cnt = 0

        while M1_cnt < len(M1_reslist):

            # Initial check to see if we have a run of good matches (more than coincidence)
            if np.all(M1_reslist[M1_cnt:(M1_cnt + sec)] == M2_reslist[M2_cnt:(M2_cnt + sec)]):
        
                while M1_reslist[M1_cnt] == M2_reslist[M2_cnt]:

                    M1_reskeep.append(M1_resid[M1_cnt])
                    M2_reskeep.append(M2_resid[M2_cnt])

                    M2_cnt += 1
                    M1_cnt += 1
                    
                    # Break if we reach the maximum array length limit
                    if M1_cnt == len(M1_reslist) or M2_cnt == len(M2_reslist):
                        break

                    # Check if we conicidently had the correct corresponding resnames
                    if len(M1_reskeep) > 2:
                        if M1_reskeep[-1] - M1_reskeep[-2] != 1 and M2_reskeep[-1] - M2_reskeep[-2] == 1:
                            M1_reskeep = M1_reskeep[:-1]
                            M2_reskeep = M2_reskeep[:-1]
                            break
                        elif M1_reskeep[-1] - M1_reskeep[-2] == 1 and M2_reskeep[-1] - M2_reskeep[-2] != 1:
                            M1_reskeep = M1_reskeep[:-1]
                            M2_reskeep = M2_reskeep[:-1]
                            break
                        else:
                            continue
        
            # Elsewise move forward in count on second structure
            else:
                M2_cnt += 1

            # Break if M1 and M2 have reached their ends, restart if only M2 has
            if M1_cnt == len(M1_reslist): #and M2_cnt == len(M2_reslist):
                break
            # break if the length of residues in M2 we are counting to now are longer than
            # the possible max number of saved residues in M2
            elif len(M2_reskeep) >= len(M2_reslist):
                break
            # Need case so we don't recount a chain in the event of a homodimer
            elif M2_cnt >= len(M2_reslist):
                M1_cnt += 1
                M2_cnt = len(M1_reskeep)
            else:
                continue

        return M1_reskeep, M2_reskeep

    def pdb2pqr(self, ff="", ligand_ff=None, amber_convert=True):
        '''
        Parses data from the pdb input into a pqr format. This uses the panda dataframe with the information
        regarding atom indexes, types etc. in the self.data files.
        It outputs a panda dataframe with the pqr equivilent information. It requires a datafile forcefield input.
        The default is the amber14sb forcefield file held within the classes/ folder.
        
        :param ff: name of forcefield text file input that needs to be read to read charges / vdw radii.
        :param ligand_ff: name of forcefield text file input that needs to be read to read charges / vdw radii for ligands (small molecules).
        :param amber_convert: If True, will assume forcefield is amber and convert resnames as necessary
        '''

        intervals = self.guess_chain_split()[1]

        if amber_convert:
            # patch naming of C-termini
            for i in intervals[1:]:
                idxs = self.same_residue(i-1, get_index=True)[1]   
                names = self.data.loc[idxs, ["name"]].values
                if np.any(names == "OC1") or np.any(names == "OXT"):
                    resname = self.data.loc[idxs[0], ["resname"]].values[0]
                    newresnames = np.array(["C"+resname]*len(idxs))
                    self.data.loc[idxs, ["resname"]] = newresnames
    
            # patch naming of N-termini
            for i in intervals[0:-1]:
                idxs = self.same_residue(i, get_index=True)[1]   
                names = self.data.loc[idxs, ["name"]].values
                if np.any(names == "H1") and np.any(names == "H2"):
                    resname = self.data.loc[idxs[0], ["resname"]].values[0]
                    newresnames = np.array(["N"+resname]*len(idxs))
                    self.data.loc[idxs, ["resname"]] = newresnames
    
            HIP = np.array(["HIP"] * 18)    # create numpy array structures to possibly reassign later
            HIE = np.array(["HIE"] * 17)    # create numpy array structures to possibly reassign later
            HID = np.array(["HID"] * 17)    # create numpy array structures to possibly reassign later
            NHIP = np.array(["NHIP"] * 20)
            NHIE = np.array(["NHIE"] * 19)
            NHID = np.array(["NHID"] * 19)
            CHIP = np.array(["CHIP"] * 20)
            CHIE = np.array(["CHIE"] * 18)
            CHID = np.array(["CHID"] * 19)

            start_chain = self.data["resid"].iloc[0]   # This is in case we get 1 or 2 as the first chain ID start
            end_chain = self.data["resid"].iloc[-1]    #  We don't know the end chain number so we find it here
            start_res = self.data["resname"].iloc[0] 
            end_res = self.data["resname"].iloc[-1] 
            
            # Need to check if first residue is actually an N-termini residue, and if so, reassign resnames if necessary
            if (self.data["name"].iloc[0:27] == 'H1').any() and (self.data["name"].iloc[0:27] == 'H2').any() and (self.data["name"].iloc[0:27] == 'H3').any() and self.data["resname"][0][0] != 'N':
                print('Found N-Termini, reassigning first resname to match the forcefield')
                start_index = self.data.index[self.data["resid"] == start_chain]
                for N in start_index:
                    self.data["resname"].iloc[N] = 'N' + start_res   # First chain needs to be prefixed with N-termini resname

             # Need to check whether it matches HIE, HID or HIP depending on what protons are present and where
            his_check = self.data["resname"] == 'HIS'  # Check if we need to do following calculation
            nhis_check = self.data["resname"] == 'NHIS' # Check for N termini HIS
            chis_check = self.data["resname"] == 'CHIS'
            if np.sum(his_check) != 0 or np.sum(nhis_check) != 0 or np.sum(chis_check) != 0:
                print("WARNING: found residue with name HIS, checking to see what protonation state it is in and reassigning to HIP, HIE or HID.\nYou should check HIS in your pdb file is right to be sure!")     
                for ix in range(len(self.data["resname"])):
                    H_length = 17 # Set this as it is more common, and also covers the basis to capture HD1 or HE2 later if necessary (as C and O tend to be last a
                    # N is always the first atom (use that as basis)                                                                                                                             
                    
                    if self.data["name"][ix] == 'N' and self.data["resname"][ix] == 'HIS':  
                                                                                           
                        if (self.data["name"][ix:(ix+H_length)] == 'HE2').any() and (self.data["name"][ix:(ix+H_length)] == 'HD1').any(): # If the residue contains HE2 and HD1, it is a HIP residue
                            H_length = 18     #   number of atoms in histdine (HIP)
                            self.data.loc[ix:(ix+H_length-1), "resname"] = HIP
    
                        elif (self.data["name"][ix:(ix+H_length)] == 'HE2').any():
                            self.data.loc[ix:(ix+H_length-1), "resname"] = HIE
    
                        elif (self.data["name"][ix:(ix+H_length)] == 'HD1').any():
                            self.data.loc[ix:(ix+H_length-1), "resname"] = HID
    
                    elif self.data["name"][ix] == 'N' and self.data["resname"][ix] == 'NHIS':
                        H_length = 19
    
                        if (self.data["name"][ix:(ix+H_length)] == 'HE2').any() and (self.data["name"][ix:(ix+H_length)] == 'HD1').any(): # If the residue contains HE2 and HD1, it is a HIP residue
                            H_length = 20     #   number of atoms in histdine (HIP)
                            self.data.loc[ix:(ix+H_length-1), "resname"] = NHIP
    
                        elif (self.data["name"][ix:(ix+H_length)] == 'HE2').any():
                            self.data.loc[ix:(ix+H_length-1), "resname"] = NHIE
    
                        elif (self.data["name"][ix:(ix+H_length)] == 'HD1').any():
                            self.data.loc[ix:(ix+H_length-1), "resname"] = NHID
    
                    elif self.data["name"][ix] == 'N' and self.data["resname"][ix] == 'CHIS':
                        H_length = 19
    
                        if (self.data["name"][ix:(ix+H_length)] == 'HE2').any() and (self.data["name"][ix:(ix+H_length)] == 'HD1').any(): # If the residue contains HE2 and HD1, it is a HIP residue
                            H_length = 20     #   number of atoms in histdine (HIP)
                            self.data.loc[ix:(ix+H_length-1), "resname"] = CHIP
    
                        elif (self.data["name"][ix:(ix+H_length)] == 'HE2').any():
                            H_length = 18
                            self.data.loc[ix:(ix+H_length-1), "resname"] = CHIE
    
                        elif (self.data["name"][ix:(ix+H_length)] == 'HD1').any():
                            self.data.loc[ix:(ix+H_length-1), "resname"] = CHID

        if len(ff) == 0:
            #"amber14sb.dat"
            folder = os.path.dirname(os.path.realpath(__file__))
            ff = "%s/amber14sb.dat" % folder
            
        if os.path.isfile(ff) != 1:
            raise Exception("ERROR: %s not found!" % ff)
        
        ff = np.loadtxt(ff, usecols=(0,1,2,3,4), dtype=str)

        if ligand_ff:
            if os.path.isfile(ligand_ff) != 1:
                raise Exception(f"ERROR: {ligand_ff} not found!")
            ligand_ff = np.loadtxt(ligand_ff, dtype=str)
            # print(ff,"\n")
            # print(ligand_ff,"\n")
            ff = np.concatenate((ff, ligand_ff), axis=0)
                        
        cols = ['resname', 'name', 'charge', 'radius', 'atomtype'] # where radius is the VdW radius in the amber file
        idx = np.arange(len(ff))
        pqr_data = pd.DataFrame(ff, index=idx, columns=cols)
    
        charges = []
        radius = []
        atomtypes = []

        # Move through each line in the pdb.data file and find the corresponding charge / vdw radius as supplied by the forcefield
        for i, resnames in enumerate(self.data["resname"]):
            values_res = pqr_data["resname"] == resnames 
            values_name = pqr_data["name"] == self.data["name"][i]
            values = np.logical_and(values_res, values_name)
            value_loc = pqr_data[values]
      
            if len(value_loc) == 0:
                print(value_loc, resnames, self.data["name"][i], self.data["resname"][i], self.data["resid"].iloc[i], self.data["index"].iloc[i])
                raise Exception("ERROR: The atom names in your PDB file do not match the PQR file")
            else:
                # For ligand atoms (which we assign a radius of 0 because Flare doesn't output the values in the topology)
                # we will want to look up the radius from the knowledge base
                if float(value_loc.iloc[0]["radius"]) == 0:

                    # the reason for looking up using [value_loc.iloc[0]["atomtype"][0], is that the atomtype is a string, 
                    # and we only want the first character (oh JD why did you implement it this way?)
                    value_loc.iloc[0]["radius"] = self.know('atom_vdw')[value_loc.iloc[0]["atomtype"][0]]

                charges.append(float(value_loc.iloc[0]["charge"]))
                radius.append(float(value_loc.iloc[0]["radius"]))
                atomtypes.append(value_loc.iloc[0]["atomtype"]) 
        
        # Drop the beta factor / occupancy data to be replaced with charge / vdw radius numbers    
        pqr = self.data.drop(['atomtype', 'radius', 'charge'], axis=1) #  remove obselete data
        pqr['atomtype'] = atomtypes  # Replace with Amber derived data for each atom
        pqr['radius'] = radius
        pqr['charge'] = charges
    
        print("Conversion Complete")
    
        return pqr
    
    def write_pqr(self, outname, conformations=[], index=[]):
        '''
        overload superclass method for writing (multi)pqr.

        :param outname: name of pqr file to be generated.
        :param index: indices of atoms to write to file. If empty, all atoms are returned. Index values obtaineable with a call like: index=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, a multipdb with all conformations will be produced.
        '''

        # store current frame, so it will be reestablished after file output is
        # complete
        currentbkp = self.current
        
        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        # Get our PQR database style
        pqr = self.pdb2pqr()

        f_out = open(outname, "w")

        for f in frames:
            # get all informations from PDB (for current conformation) in a list
            self.set_current(f)
            d = self.get_pdb_data(index)
            
            # Get our 
            
            
            # Build our hexidecimal array if num. of atoms > 99999
            idx_val = np.arange(1, len(d) + 1, 1)
            if len(idx_val) > 99999:
                vhex = np.vectorize(hex)
                idx_val = vhex(idx_val)   # convert index values to hexidecimal
                idx_val = [num[2:] for num in idx_val]  # remove 0x at start of hexidecimal number
            
            for i in range(0, len(d), 1):
                # create and write PDB line
                if d[i][2][0].isdigit():
                    L = '%-6s%5s %-5s%-4s%1s%4s    %8.3f%8.3f%8.3f%7.4f%7.4f        %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(pqr.iloc[i]["charge"]), float(pqr.iloc[i]["radius"]), d[i][11])
                else:
                    L = '%-6s%5s  %-4s%-4s%1s%4s    %8.3f%8.3f%8.3f%7.4f%7.4f        %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(pqr.iloc[i]["charge"]), float(pqr.iloc[i]["radius"]), d[i][11])
                f_out.write(L)

            f_out.write("END\n")

        f_out.close()

        self.set_current(currentbkp)

        return
    
    def clean(self, path='~/biobox/classes/remove_alt_conf.sh', remove_non_amino=True):
        '''
        clean up a PDB files from alt conformations and ligands. Requires subprocess to be installed.
        (For now) requires input to be a protein, so will remove all ligands etc.
        This removes residues with the least certainty (based on beta factor). 
        If no beta factor is present, it removes all residue conformations after the first

        :param path: Path to the removing alt conf. bash script (in current folder by default)
        :param remove_non_amino: Remove all non-standard amino acids (including water, metals etc. which are defined as ATOMS)
        :returns: Returns a new Molecule object that has been cleaned
        '''
        import subprocess

        # all amino acids (in case we want to remove non-standard residues). Also includes N and C prefixs
        amino = ['ILE','GLN', 'GLY', 'MSE', 'GLU', 'CYS', 'ASP', 'SER', 'HSD', 'HSE', 'PRO', 'CYX', 'HSP', 'HID', 'HIE', 'ASN', 
                'HIP', 'VAL', 'THR', 'HIS', 'TRP', 'LYS', 'PHE', 'ALA', 'MET', 'LEU', 'ARG', 'TYR', 'NILE', 'NGLN', 'NGLY',
                'NMSE', 'NGLU', 'NCYS', 'NASP', 'NSER', 'NHSD', 'NHSE', 'NPRO', 'NCYX', 'NHSP', 'NHID', 'NHIE', 'NASN', 'NHIP', 
                'NVAL', 'NTHR',  'NHIS','NTRP', 'NLYS', 'NPHE', 'NALA', 'NMET', 'NLEU', 'NARG', 'NTYR', 'CILE', 'CGLN', 'CGLY', 
                'CMSE', 'CGLU', 'CCYS', 'CASP', 'CSER', 'CHSD', 'CHSE', 'CPRO', 'CCYX', 'CHSP', 'CHID', 'CHIE', 'CASN', 'CHIP', 
                'CVAL', 'CTHR', 'CHIS', 'CTRP', 'CLYS', 'CPHE', 'CALA', 'CMET', 'CLEU', 'CARG', 'CTYR'] 

        self.write_pdb("tmp2.pdb")
        subprocess.call(path + " tmp2.pdb", shell=True)

        if remove_non_amino:
            B = Molecule()
            B.import_pdb("clean_tmp2.pdb")
            B_idxs = B.atomselect("*", amino, "*", get_index=True, use_resname=True)[1]
            A = B.get_subset(B_idxs)
        else:
            A = Molecule()
            A.import_pdb("clean_tmp2.pdb")
        
        # Get residues with strings in
        # Find our what first numbers are (i.e. remove strings) so we have all conformations and the non string version
        # Then check what avg beta factor is, if it's zero, chop off all string conformations
        A_idxs = A.atomselect("*", "*", "CA", get_index=True)[1]
        resid = np.asarray(A.data['resid'][A_idxs])
        repeat = []

        # get indices of repeat residues
        for i in range(1, len(resid)):
            if resid[i-1] == resid[i]:
                repeat.append(i)
            else:
                continue
        
        #get relevent chains
        chains = np.unique(A.data["chain"][A_idxs[repeat]])
        # keep a record of indices to keep and all of the ones we explore
        keep_res_idx = []
        all_repeat_idx = []
        for c in chains:
            chain_idx = A_idxs[repeat][A.data["chain"][A_idxs[repeat]] == c]

            # loop through repeat residues and calculate beta factors for removal
            for r in np.unique(A.data['resid'][chain_idx]):
                A_repeat_idx = A.atomselect(c, r, "*", get_index=True)[1]
                all_repeat_idx.extend(A_repeat_idx)
                A_subset = A.get_subset(A_repeat_idx)

                beta = []
                A_CAs = np.where(A_subset.data['name'] == "CA")[0]

                # First do a quick check in case we have any zero betas (to skip loop below)
                if np.any(np.asarray(A_subset.data['beta']) == 0.0):
                    keep_res_idx.extend(A_repeat_idx[A_CAs[0]-1 : A_CAs[1] -1])
                else:
                    # loop through each residue in the same residue set
                    for i in range(len(A_CAs)):
                        # Always an N preceding a CA
                        if i < len(A_CAs) -1:
                            beta.append(np.mean(A_subset.data['beta'][A_CAs[i] - 1 : A_CAs[i+1] - 1]))
                        else: 
                            beta.append(np.mean(A_subset.data['beta'][A_CAs[i] - 1 : 1 + np.asarray(A_subset.data['index'])[-1]]))

                    # only select residue with lowest beta
                    min_res = A_CAs[np.argmin(beta)]
                    if np.argmin(beta) + 1 == len(A_CAs):
                        keep_res_idx.extend(A_repeat_idx[min_res-1 : 1 + np.asarray(A_subset.data['index'])[-1]])
                    else:
                        keep_res_idx.extend(A_repeat_idx[min_res-1 : A_CAs[np.argmin(beta) + 1] - 1])

        # Now just pull the indices we want to remove
        all_res_set = set(all_repeat_idx)
        keep_res_set = set(keep_res_idx)
        idx_remove = np.asarray(list(all_res_set - keep_res_set) + list(keep_res_set - all_res_set))

        # clean files
        os.remove("clean_tmp2.pdb")
        os.remove("tmp2.pdb")

        return A.get_subset(idx_remove, flip=True)

    def get_dipole_map(self, orig, pqr, time_start = 0, time_end = 2,resolution = 1., vox_in_window = 3., write_dipole_map = True, fname = "dipole_map.tcl"):
        '''
        Method for generating dipole maps to be used for electron density map generation. Also prints a dipole map as a result (and if desired). It calls a cython code in lib.

        :param orig: Origin points for voxel grid
        :param pqr: PQR file for self. Can be generated by calling pdb2pqr above
        :param time_start: First frame to parse in multipdb
        :param time_end: Last frame to parse in multipdb
        :param resolution: Desired resolution of voxel
        :param vox_in_window: Amount of surrounding space to contribute to local dipole. vox_in_window * resolution gives window size (in Ang.)
        :param write_dipole_map: Write a dipole map in TCL format to be read in via VMD.
        :param fname: Name of desired dipole map to be written
        '''

        charges = pqr["charge"].values[:]

        crd = self.coordinates[time_start:time_end] # cut out coordinates we're interested in 
 
        time_end -= time_start # shift to compensate for cutting the coordinates earlier
        time_start = 0

        dipole_map = e_density.c_get_dipole_map(crd = crd, orig = orig, charges = charges, time_start = time_start, time_end = time_end,resolution = resolution, vox_in_window = vox_in_window, write_dipole_map = write_dipole_map, fname = fname)

        return dipole_map

    def get_dipole_density(self, dipole_map, orig, min_val, V, outname, vox_in_window = 3., eqn = 'gauss', T = 310.15, P = 101. * 10**3, epsilonE = 54., resolution = 1.):
        '''
        Method to generate an electron density map based on a voxel grid of dipole vectors

        :param dipole_map: The dipole map input. Can be generated with get_dipole_map above
        :param orig: Origin points for voxel grid
        :param min_val: Minimum coordinates of edge points for the voxel grid (i.e. a single x, y, z point defining the start point of the grid to match with the multipdb)
        :param V: Volume of a voxel (can be found by resolution**3, but left blank in case later version institute a sphere)
        :param outname: Name of electron density map file produced
        :param vox_in_window: Amount of surrounding space to contribute to local dipole. vox_in_window * resolution gives window size (in Ang.)
        :param eqn: Equation mode to model the electron density
        :param T: Temperature of MD
        :param P: Pressure of MD
        :param epsilonE: Continuum dielectric surrounding the protein
        :param resolution: Desired resolution of voxel
        '''

        dummy = e_density.c_get_dipole_density(dipole_map = dipole_map, orig = orig, min_val = min_val, V = V, outname = outname, vox_in_window = vox_in_window, eqn = eqn, T = T, P = P, epsilonE = epsilonE, resolution = resolution)

