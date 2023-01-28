import torch
import networkx as nx
from torch_geometric.utils import from_networkx, dense_to_sparse
from rdkit import Chem
from .chem import mol2smiles, smiles2mol
from rdkit.Chem import PandasTools
from torch_geometric.data import Data
from rdkit.Chem import rdmolops

DEFAULT_ATOM_TYPE_SET = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I",]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]


NUM_ATOM_FEATURES = sum([len(l) + 1 for l in [
    DEFAULT_ATOM_TYPE_SET,
    DEFAULT_HYBRIDIZATION_SET,
    DEFAULT_TOTAL_NUM_Hs_SET,
    DEFAULT_TOTAL_DEGREE_SET]])


NUM_BOND_FEATURES = sum([len(l) + 1 for l in [
    DEFAULT_BOND_TYPE_SET,
    DEFAULT_BOND_STEREO_SET]]) + 1


def one_hot_encoding(value, value_set):
    num_choices = len(value_set) + 1
    one_hot_vector = [0.0] * num_choices
    try:
        index = value_set.index(value)
        one_hot_vector[index] = 1.0
    except ValueError:
        one_hot_vector[-1] = 1.0
    return one_hot_vector


def get_atom_features(atom):
    features = one_hot_encoding(atom.GetSymbol(), DEFAULT_ATOM_TYPE_SET)
    features += one_hot_encoding(atom.GetHybridization(), DEFAULT_HYBRIDIZATION_SET)
    features += one_hot_encoding(atom.GetTotalNumHs(), DEFAULT_TOTAL_NUM_Hs_SET)
    features += one_hot_encoding(atom.GetTotalDegree(), DEFAULT_TOTAL_DEGREE_SET)
    return features


def get_bond_features(bond):
    features = one_hot_encoding(bond.GetBondType(), DEFAULT_BOND_TYPE_SET)
    features += one_hot_encoding(bond.GetBondType(), DEFAULT_BOND_STEREO_SET)
    features += [float(bond.IsInRing())]
    return features


def mol2pyg(mol):
    if isinstance(mol, str):
        mol = smiles2mol(mol)

    assert mol is not None, print(mol2smiles(mol))

    A = rdmolops.GetAdjacencyMatrix(mol)
    node_features, edge_features = [], []

    for idx in range(A.shape[0]):
        atom = mol.GetAtomWithIdx(idx)
        atom_features = get_atom_features(atom)
        node_features.append(atom_features)

    for bond in mol.GetBonds():
        bond_features = get_bond_features(bond)
        start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_features.append(bond_features)
        edge_features.append(bond_features)  # add for reverse edge

    edge_index, eattr = dense_to_sparse(torch.Tensor(A))
    node_features = torch.Tensor(node_features)
    edge_features = torch.Tensor(edge_features)
    return Data(edge_index=edge_index, x=node_features, edge_attr=edge_features)

def get_drug_smiles_and_names(sdf_filename='dataset/raw/drug-structures.sdf'):
    # Get smiles
    smiles = []
    supplier = Chem.SDMolSupplier(sdf_filename)
    for mol in supplier:
       smiles.append(Chem.MolToSmiles(mol))

    # Get names
    df = PandasTools.LoadSDF(sdf_filename)

    return smiles, df['ID'].tolist()
