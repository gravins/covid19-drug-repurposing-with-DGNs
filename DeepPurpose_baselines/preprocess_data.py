import torch
import pdb
import dill
import copy
import tqdm
import random
import itertools
import numpy as np
import pandas as pd
from utils import *
from models.proteins_utils import mol2pyg, get_drug_smiles_and_names

# Set random seed
seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_drug_and_gene_embs(df, d, d2smile, d2g, prot_embs):
    # Get drug embedding
    try:
        drug_emb = (torch.from_numpy(df.loc[d].values).type(torch.FloatTensor)).unsqueeze(0)
    except KeyError:
        return None, None, None, None

    # Get proteins embedding
    genes_emb = []
    genes_ids = []
    missing_genes = 0
    for g in d2g[d]:
        try:
            genes_emb.append(torch.FloatTensor(prot_embs[int(g)]).unsqueeze(0))
            genes_ids.append(g)
        except KeyError:
            missing_genes += 1
    return drug_emb, genes_emb, genes_ids, missing_genes


def protein_features_map(entrezid, drug_emb_path, saving_filename='dataset'):
    '''
    :param: entrezid: list of entrez gene ids
    :param: prot_fun: proteins aggregation function
    :param: prot_per_drug: percentage of proteins considering associated to a drug
    :param: single_prot: if True each association protein-drug is considered as instance for the problem (prot_fun is ignored),
                         else the combination of protreins with the drug is considered as instance
    :param at_least_prot: if > 0 then  the key "tag" is added to the output containing 1 if the drug is associated with at least "at_least_prot" proteins, 0 otherwise.
                          Ignored if single_prot == True.
    :param: saving_filename: name of files used to store the result
    '''
    # Load protein features
    prot_embs = np.delete(np.load("dataset/embeddings_feat.npz")['arr_0'], list(range(1, 40168)), axis=1)
    print('Proteins embedding shape :', prot_embs.shape)

    tmp = prot_embs[:,[0]]
    prot_embs = np.delete(prot_embs, [0], axis=1)
    prot_embs = {int(k):v for k,v in zip(tmp, prot_embs)}

    # Load drug embeddings
    df = pd.read_csv(drug_emb_path, index_col='DrugBankIDs')
    if 'names' in df.columns and 'smiles' in df.columns:
        df = df.drop(['names', 'smiles'], axis='columns')

    d2g = drug2genes()            # dict that maps drugs with genes
    g2d = read_gene2drug_files()  # dict that maps genes with drugs
    d2smile = {name:smile for smile, name in zip(*get_drug_smiles_and_names())}         # dict that maps drugs with smiles

    data = []

    missing_drugs = 0
    missing_prots = []
    for d in tqdm.tqdm(d2g.keys()):
        drug_emb, genes_emb, genes_ids, missing = get_drug_and_gene_embs(df, d, d2smile, d2g, prot_embs)

        if drug_emb is None:
            missing_drugs += 1
            continue

        missing_prots.append(missing)
        if len(genes_emb) < 1:
            continue

        # Considering only the percentage of proteins for each drug expressed by prot_per_drug
        permuted_ids = list(range(len(genes_emb)))
        random.shuffle(permuted_ids)
        max_prot = len(genes_emb)
        considered_genes = [genes_ids[gid] for gid in permuted_ids[:max_prot]]

        # Add a triplet for each protein-drug association
        for gid in considered_genes:
            sample = {'y': 1., 'drug': d, 'gene': gid}
            data.append(sample)

        # Create negative sample: given a protein linked with a drug, select a new random drug not connected with it
        drugs = list(d2g.keys())
        for idx in permuted_ids[:max_prot]:
            g = genes_ids[idx]
            i = random.randint(0,len(drugs)-1)
            while drugs[i] in g2d[g] or drugs[i] not in df.index:
                # Search until find a drug not connected with the gene
                i = random.randint(0,len(drugs)-1)

            drug_emb = drugs[i]
            gid = int(g)

            neg_sample = {'y': 0., 'drug': d, 'gene':gid}
            data.append(neg_sample)

    dill.dump(data, open(saving_filename+'.p', 'wb'))
    print('Missing drugs: ', missing_drugs)
    print('Missing proteins per drug on average: ', np.mean(missing_prots))


if __name__ == "__main__":

    entrezid = dill.load(open('dataset/entrezid.p', 'rb'))
    inp_data = 'dataset/drug_embeddings.csv'

    # 1 prot -- 1 drug
    saving_filename = 'dataset/triplets_dataset_single_prot_seq_emb'
    protein_features_map(entrezid, inp_data, saving_filename=saving_filename) 