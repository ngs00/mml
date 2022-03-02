import numpy
import pandas
import torch
import json
from tqdm import tqdm
from mendeleev.fetch import fetch_table
from mendeleev import element
from sklearn import preprocessing
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from ml.util import normalize
from ml.util import even_samples
from ml.util import rbf


atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
atom_syms = {v: k for k, v in atom_nums.items()}
elem_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
                   'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat']
n_elem_feats = len(elem_feat_names) + 1


def get_elem_feats():
    tb_atom_feats = fetch_table('elements')
    elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))
    ion_engs = numpy.zeros((elem_feats.shape[0], 1))

    for i in range(0, ion_engs.shape[0]):
        ion_eng = element(i + 1).ionenergies
        if 1 in ion_eng:
            ion_engs[i, 0] = element(i + 1).ionenergies[1]
        else:
            ion_engs[i, 0] = 0

    elem_feats = numpy.hstack([elem_feats, ion_engs])

    return preprocessing.scale(elem_feats)


def load_elem_embeddings(path_elem_attr):
    elem_attrs = list()
    with open(path_elem_attr) as json_file:
        elem_attr = json.load(json_file)
        for elem in atom_nums.keys():
            elem_attrs.append(numpy.array(elem_attr[elem]))

    return numpy.vstack(elem_attrs)


def load_dataset(path_structs, metadata_file, idx_struct, idx_target=None,
                 n_bond_feats=32, radius=4, norm_target=False):
    elem_feats = load_elem_embeddings('datasets/res/matscholar-embedding.json')
    list_cgs = list()
    metadata = numpy.array(pandas.read_excel(metadata_file))
    targets = None

    if idx_target is not None:
        targets = metadata[:, idx_target]
        if norm_target:
            targets = normalize(targets, f_min=numpy.min(targets), f_max=numpy.max(targets))

    for i in tqdm(range(0, metadata.shape[0])):
        mp_id = metadata[i, idx_struct]
        target = None if idx_target is None else targets[i]
        cg = read_cif(elem_feats, path_structs, mp_id, n_bond_feats, radius, idx=i, target=target)

        if cg is not None:
            list_cgs.append(cg)

    return list_cgs


def read_cif(elem_feats, path, m_id, n_bond_feats, radius, idx, target=None):
    crys = Structure.from_file(path + '/' + m_id + '.cif')
    atom_feats = get_atom_feats(crys, elem_feats)
    bonds, bond_feats = get_bonds(crys, n_bond_feats, radius)

    if bonds is None:
        return None

    atom_feats = torch.tensor(atom_feats, dtype=torch.float).cuda()
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous().cuda()
    bond_feats = torch.tensor(bond_feats, dtype=torch.float).cuda()

    if target is None:
        return Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, idx=idx, y_var=None)
    else:
        return Data(x=atom_feats, y=torch.tensor(target, dtype=torch.float).view(1, -1).cuda(), edge_index=bonds,
                    edge_attr=bond_feats, idx=idx, y_var=None)


def get_atom_feats(crys, elem_feats):
    atoms = crys.atomic_numbers
    atom_feats = list()

    for i in range(0, len(atoms)):
        atom_feats.append(elem_feats[atoms[i] - 1, :])

    return numpy.vstack(atom_feats)


def get_bonds(crys, n_bond_feats, radius):
    rbf_means = even_samples(0, radius, n_bond_feats)
    list_nbrs = crys.get_all_neighbors(radius, include_index=True)
    bonds = list()
    bond_feats = list()

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            bonds.append([i, nbrs[j][2]])
            bond_feats.append(rbf(numpy.full(n_bond_feats, nbrs[j][1]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None

    return numpy.vstack(bonds), numpy.vstack(bond_feats)
