from rdkit import Chem


def mol2smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def smiles2mol(smi):
    return Chem.MolFromSmiles(smi, sanitize=True)


def mols2smiles(mols):
    return [mol2smiles(m) for m in mols]


def smiles2mols(smiles):
    return [smiles2mol(s) for s in smiles]