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
from model.proteins_utils import mol2pyg, get_drug_smiles_and_names
from torch_geometric.data import Data

# Set random seed
seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Protein aggregation functions
prot_sum = lambda x: torch.sum(x, dim=0).unsqueeze(0)
prot_mean = lambda x: torch.mean(x, dim=0).unsqueeze(0)
prot_min = lambda x: torch.min(x, dim=0).values.unsqueeze(0)
prot_max = lambda x: torch.max(x, dim=0).values.unsqueeze(0)
prot_sum_max = lambda x: torch.cat((torch.sum(x, dim=0).unsqueeze(0), torch.max(x, dim=0).values.unsqueeze(0)), dim=1)
prot_mean_max = lambda x: torch.cat((torch.mean(x, dim=0).unsqueeze(0), torch.max(x, dim=0).values.unsqueeze(0)), dim=1)

def get_drug_and_gene_embs(d, d2smile, d2g, prot_embs, df=None):
    if df is not None:
        # Get drug embedding
        try:
            drug_emb = (torch.from_numpy(df.loc[d].values).type(torch.FloatTensor)).unsqueeze(0)
        except KeyError:
            return None, None, None, None

    if e2e:
        drug_emb = mol2pyg(d2smile[d])
        if not drug_emb.__contains__('x'):
            print(d)
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


def protein_features_map(entrezid, drug_emb_path, prot_fun=prot_sum_max, prot_per_drug=1., single_prot=False, at_least_prot=0, e2e=False, saving_filename='dataset', baseline=False):
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
    if baseline:
        # Load protein and drug features
        prot_embs = dill.load(open('dataset/prot_embedding_baseline.p', 'rb'))
        df = pd.read_csv('dataset/drug_embedding_baseline_FTT-L3-R1024.csv', index_col='DrugBankIDs') # drugbank id, name, smile, ft_1, ..., ft_1024

    else:
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
    buckets = {'1':[[],0], '2':[[],0], '3':[[],0], '4-9':[[],0], '10-300':[[],0]} #TODO stratified neg. samples
    for d in tqdm.tqdm(d2g.keys()):
        drug_emb, genes_emb, genes_ids, missing = get_drug_and_gene_embs(d, d2smile, d2g, prot_embs, df)

        if drug_emb is None:
            missing_drugs += 1
            continue

        missing_prots.append(missing)
        if len(genes_emb) < 1:
            continue

        # Considering only the percentage of proteins for each drug expressed by prot_per_drug
        permuted_ids = list(range(len(genes_emb)))
        random.shuffle(permuted_ids)
        max_prot = max(int(len(genes_emb) * prot_per_drug), 1)
        x = torch.cat(genes_emb, dim=0)
        x = x[permuted_ids[:max_prot]]

        nprot = x.size(0)#########################################################
        if x.size(0) >= 10:  #TODO stratified neg. samples
            buckets['10-300'][1] += 1
        elif x.size(0) >= 4:
            buckets['4-9'][1] += 1
        else:
            buckets[str(x.size(0))][1] += 1
        if not single_prot:
            x = prot_fun(x)  # combine proteins linked with a drug

            if e2e:
                sample = copy.deepcopy(drug_emb)
                sample['y'] = 1.
                sample['drug'] = d
                sample['gene_emb'] = x
                sample['n_associated_prots'] = nprot ########################################
                if at_least_prot:
                    sample['tag'] = int(len(d2g[d]) >= at_least_prot)
            else:
                sample = {'y':1., 'drug':d}
                sample.update({'x': torch.cat((x,drug_emb),dim=1).squeeze(0)})
                if at_least_prot:
                    sample.update({'tag': int(len(d2g[d]) >= at_least_prot)})

            data.append(sample)

        else:
            # Add a triplet for each protein-drug association
            for emb in x:
                if e2e:
                    sample = copy.deepcopy(drug_emb)
                    sample['y'] = 1.
                    sample['drug'] = d
                    sample['gene_emb'] = emb.unsqueeze(0)
                    sample['n_associated_prots'] = 1
                else:
                    sample = {'y': 1., 
                              'drug': d,
                              'x': torch.cat((emb, drug_emb.squeeze(0)),dim=0),
                              'drug_emb': drug_emb.squeeze(0),
                              'gene_emb': emb,
                              'n_associated_prots': 1}
                data.append(sample)
        
        #TODO stratified neg. samples
        for bk in buckets:
            if single_prot and bk != '1':
                continue

            if bk == "4-9": prot_number = random.randint(4,9)
            elif bk == "10-300": prot_number = random.randint(10,300)
            else: prot_number = int(bk)
            
            multi_gene = []
            for _ in range(prot_number):
                g = random.sample(prot_embs.keys(),1)[0]
                while str(g) in d2g[d]:
                    g = random.sample(prot_embs.keys(),1)[0]
 
                x = torch.FloatTensor(prot_embs[g]).unsqueeze(0)
                multi_gene.append(x)

            x = torch.cat(multi_gene, dim=0)
            x = prot_fun(x) if not single_prot else x
 
            if e2e:
                neg_sample = copy.deepcopy(drug_emb)
                neg_sample['y'] = 0.
                neg_sample['drug'] = d
                neg_sample['gene_emb'] = x
                neg_sample['n_associated_prots'] = prot_number ########################################
                if at_least_prot and not single_prot:
                    neg_sample['tag'] = int(prot_number >= at_least_prot)
            else:
                neg_sample = {'y': 0., 
                              'drug': d,
                              'x': torch.cat((x, drug_emb),dim=1).squeeze(0),
                              'drug_emb': drug_emb.squeeze(0),
                              'gene_emb': x.squeeze(0),
                              'n_associated_prots': prot_number} ############################
                if not single_prot and at_least_prot:
                    neg_sample.update({'tag': int(len(d2g[d]) >= at_least_prot)})
            buckets[bk][0].append(neg_sample)
            
        '''
        # Create negative sample: given a protein linked with a drug, select a new random drug not connected with it
        drugs = list(d2g.keys())
        for idx in permuted_ids[:max_prot]:
            g = genes_ids[idx]
            i = random.randint(0,len(drugs)-1)
            while drugs[i] in g2d[g] or drugs[i] not in df.index:
                # Search until find a drug not connected with the gene
                i = random.randint(0,len(drugs)-1)

            if e2e:
                drug_emb = mol2pyg(d2smile[drugs[i]])
            else:
                drug_emb = (torch.from_numpy(df.loc[drugs[i]].values).type(torch.FloatTensor)).unsqueeze(0)

            x = torch.FloatTensor(prot_embs[int(g)]).unsqueeze(0)
            x = prot_fun(x) if not single_prot else x

            if e2e:
                neg_sample = copy.deepcopy(drug_emb)
                neg_sample['y'] = 0.
                neg_sample['drug'] = d
                neg_sample['gene_emb'] = x
                if not single_prot and at_least_prot:
                    neg_sample['tag'] = int(len(d2g[d]) >= at_least_prot)
            else:
                neg_sample = {'y': 0., 'drug': d}
                neg_sample.update({'x': torch.cat((x,drug_emb),dim=1).squeeze(0)})
                if not single_prot and at_least_prot:
                    neg_sample.update({'tag': int(len(d2g[d]) >= at_least_prot)})
            data.append(neg_sample)

            if not single_prot:
                break
        '''
    if single_prot:         #TODO stratified neg. samples
        random.shuffle(buckets['1'][0])
        data += buckets['1'][0][:len(data)]
    else:
        for bk in buckets:
            random.shuffle(buckets[bk][0])
            data += buckets[bk][0][:buckets[bk][1]]
    for bk in buckets:
        print(bk,':',buckets[bk][1])

    dill.dump(data, open(saving_filename+'.p', 'wb'))
    print('Missing drugs: ', missing_drugs)
    print('Missing proteins per drug on average: ', np.mean(missing_prots))


def create_protein_subset_testset(drug_list, drug_emb_path, prot_fun=prot_sum_max, e2e=False, saving_path="perturbed_test_set.p"):
    # Load protein embeddings
    prot_embs = np.delete(np.load("dataset/embeddings_feat.npz")['arr_0'], list(range(1, 40168)), axis=1)
    tmp = prot_embs[:,[0]]
    prot_embs = np.delete(prot_embs, [0], axis=1)
    prot_embs = {int(k):v for k,v in zip(tmp, prot_embs)}

    # Load drug embeddings
    df = pd.read_csv(drug_emb_path, index_col='DrugBankIDs')
    if 'names' in df.columns and 'smiles' in df.columns:
        df = df.drop(['names', 'smiles'], axis='columns')

    d2g = drug2genes()  # dict that maps drugs with genes
    d2smile = {name:smile for smile, name in zip(*get_drug_smiles_and_names())}         # dict that maps drugs with smiles

    test_set={'0.75':{'data':[], 'batch':[]},
              '0.50':{'data':[], 'batch':[]},
              '0.25':{'data':[], 'batch':[]}}

    missing_drugs = 0
    missing_prots = []
    for i, d in tqdm.tqdm(enumerate(drug_list)):
        drug_emb, genes_emb, genes_ids, missing = get_drug_and_gene_embs(d, d2smile, d2g, prot_embs, df)
        
        if drug_emb is None:
            missing_drugs += 1
            continue

        missing_prots.append(missing)
        if len(genes_emb) < 1:
            continue

        if len(genes_emb) > 3:
            x = torch.cat(genes_emb, dim=0)

            # Considering only the percentage of proteins for each drug:
            # - split protein embeddings in 4 random folds
            # - combine them excluding 1, 2,or 3 folds
            # - combine the proteins in the remaining data and store
            permuted_ids = list(range(len(genes_emb)))
            random.shuffle(permuted_ids)
            folds_dim = len(permuted_ids) / 4
            folds = [permuted_ids[int(folds_dim * i):int(folds_dim * (i + 1))] for i in range(4)]

            for j, percentage in enumerate(['0.25', '0.50', '0.75']):
                j += 1
                for idxs in itertools.combinations(folds, j):
                    fold_ids = []
                    for idx in idxs:
                        fold_ids += idx
                    xx = x[fold_ids]
                    xx = prot_fun(xx)  # combine proteins linked with a drug

                    if e2e:
                        sample = copy.deepcopy(drug_emb)
                        sample['y'] = 1.
                        sample['drug'] = d
                        sample['gene_emb'] = xx

                    else:
                        sample = {'y': 1., 'drug': d}
                        sample.update({'x': torch.cat((xx,drug_emb),dim=1).squeeze(0)})

                    test_set[percentage]['data'].append(sample)
                    test_set[percentage]['batch'].append(i)

    dill.dump(test_set, open(saving_path, "wb"))
    print(len(test_set['0.75']['batch']))
    print(len(test_set['0.50']['batch']))
    print(len(test_set['0.25']['batch']))

    print('Missing drugs: ', missing_drugs)
    print('Missing proteins per drug on average: ', np.mean(missing_prots))


def covid19_test_dataset(covid_genes, drug_emb_path='drug_rep.csv', remove_known=True, add_negative=True, prot_fun=prot_sum_max, prot_per_drug=1., single_prot=False, at_least_prot=0, e2e=True, saving_filename='test_dataset'):
    '''
    :param: covid_genes: set of covid entrez gene id
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

    complete_prot_embs = {int(k):v for k,v in zip(tmp, prot_embs)}
    prot_embs = {int(k):v for k,v in zip(tmp, prot_embs) if int(k) in covid_genes}
    print('missing protein embeddings: ', len(covid_genes) - len(prot_embs), ' over ', len(covid_genes), ' proteins\t[missing prot. ids: ', covid_genes.difference(prot_embs), ']')

    # Load drugs
    if not e2e:
        tested_drugs = pd.read_csv(drug_emb_path, index_col='DrugBankIDs')
        if 'names' in tested_drugs.columns and 'smiles' in tested_drugs.columns:
            df = tested_drugs.drop(['names', 'smiles'], axis='columns')
    else:
        tested_drugs = pd.read_csv(drug_emb_path)
        df = None

    if remove_known:
        # Remove genes already known for the interaction
        remove_ = []
        d2g = drug2genes()
        genes = set([str(k) for k in prot_embs.keys()])
        for k in tested_drugs['drugbankID']:
            try:
                remove_ += list(genes.difference(genes.difference(d2g[k])))
            except:
                pass
        print(remove_)
        for k in set(remove_):
            del prot_embs[int(k)]
        print('removed', len(set(remove_)), 'known prots')

    covid_d2g = {k: set(prot_embs.keys()) for k in tested_drugs['drugbankID']} # dict that maps drugs with genes

    d2smile = {drugbank_id:smile for _, (drugbank_id, smile) in tested_drugs[['drugbankID', 'smile']].iterrows()}  # dict that maps drugs with smiles
    print(len(prot_embs))
    data = []
    missing_drugs = 0
    missing_prots = []
    for d in tqdm.tqdm(covid_d2g.keys()):
        drug_emb, genes_emb, genes_ids, missing = get_drug_and_gene_embs(d, d2smile, covid_d2g, prot_embs, df)

        if drug_emb is None:
            missing_drugs += 1
            continue

        missing_prots.append(missing)
        if len(genes_emb) < 1:
            continue

        # Considering only the percentage of proteins for each drug expressed by prot_per_drug
        permuted_ids = list(range(len(genes_emb)))
        random.shuffle(permuted_ids)
        max_prot = max(int(len(genes_emb) * prot_per_drug), 1)
        x = torch.cat(genes_emb, dim=0)
        x = x[permuted_ids[:max_prot]]

        if not single_prot:
            x = prot_fun(x)  # combine proteins linked with a drug

            if e2e:
                sample = copy.deepcopy(drug_emb)
                sample['y'] = 1.
                sample['drug'] = d
                sample['gene_emb'] = x
                if at_least_prot:
                    sample['tag'] = int(len(covid_d2g[d]) >= at_least_prot)
            else:
                sample = {'y':1., 'drug':d}
                sample.update({'x': torch.cat((x,drug_emb),dim=1).squeeze(0)})
                if at_least_prot:
                    sample.update({'tag': int(len(covid_d2g[d]) >= at_least_prot)})

            data.append(sample)

        else:
            # Add a triplet for each protein-drug association
            for emb in x:
                if e2e:
                    sample = copy.deepcopy(drug_emb)
                    sample['y'] = 1.
                    sample['drug'] = d
                    sample['gene_emb'] = emb.unsqueeze(0)
                else:
                    sample = {'y': 1., 'drug': d}
                    sample.update({'x': torch.cat((emb.unsqueeze(0),drug_emb),dim=1).squeeze(0)})
                data.append(sample)

        if add_negative:
           # Get proteins not linked with the drug and different from covid-19 proteins
           g = random.sample(complete_prot_embs.keys(),1)[0]
           while g in prot_embs.keys() or (d in d2g and str(g) in d2g[d]):
               g = random.sample(complete_prot_embs.keys(),1)[0]

           x = torch.FloatTensor(complete_prot_embs[g]).unsqueeze(0)
           x = prot_fun(x) if not single_prot else x

           neg_sample = copy.deepcopy(drug_emb)
           neg_sample['y'] = 0.
           neg_sample['drug'] = d
           neg_sample['gene_emb'] = x
           data.append(neg_sample)

    dill.dump(data, open(saving_filename+'.p', 'wb'))
    print('Missing drugs: ', missing_drugs)
    print('Missing proteins per drug on average: ', np.mean(missing_prots))


def extract_covid_genes(path, col):
    df = pd.read_csv(path, sep='\t', dtype={col:str})
    df = df.drop(df[(df[col].str.contains('-')) | (df[col]=='')].index)
    return set(df[col].astype(int))


if __name__ == "__main__":


    e2e = False
    #path, col = 'dataset/BIOGRID-PROJECT-covid19_coronavirus_project-GENES-4.2.193.projectindex.csv', 'EntrezGeneID'
    #path, col = 'dataset/BIOGRID-PROJECT-covid19_coronavirus_project-PTM-4.2.193.ptmtab.csv', 'EntrezGeneID'
    #path, col = 'dataset/BIOGRID-PROJECT-covid19_coronavirus_project-INTERACTIONS-4.2.193.tab3.csv', 'Entrez Gene Interactor B'
    #path, col = 'dataset/human_covid_host_genes.tab', 'EntrezGeneID'
    #covid_genes = extract_covid_genes(path, col)
    #covid19_test_dataset(covid_genes) #prot_per_drug=0.0030395137)
    #exit()
    AE = ['_AE_16', '_AE_64', '_AE_96', '_AE_96_denoising']
    mode = ['single_prot', 'stratified', 'standard', 'subset']

    mode = mode[0] #-1]
    ae = None ## AE[1]
    baseline = True

    if (baseline and e2e) or (baseline and ae):
        raise ValueError('Cannot use baseline embeddings with end to end configuration or AE embeddigns')

    if ae and e2e:
        raise ValueError('Cannot use AE embeddings with end to end configuration')
    entrezid = dill.load(open('dataset/entrezid.p', 'rb'))
    inp_data = 'dataset/drug_embeddings.csv'
    if ae:
        inp_data = inp_data.replace('deddings', 's'+ae)
    
    if mode == 'single_prot':
        # 1 prot -- 1 drug
        saving_filename = 'dataset/triplets_dataset_single_prot'
        if ae:
            saving_filename += ae
        elif e2e:
            saving_filename += '_e2e'
        elif baseline:
            saving_filename += '_baseline'
        protein_features_map(entrezid, inp_data,  prot_fun=prot_sum_max, prot_per_drug=1., single_prot=True, e2e=e2e, saving_filename=saving_filename, baseline=baseline) 
    elif mode == 'stratified':
        # 1 drug -- all associated drugs  + tag
        saving_filename = 'dataset/triplets_dataset_tagged'
        if ae:
            saving_filename += ae
        if e2e:
            saving_filename += '_e2e_MEAN+MAX_CORRETTO-provaprova'
        #protein_features_map(entrezid, inp_data, prot_fun=prot_sum_max, at_least_prot=4, e2e=e2e, saving_filename=saving_filename) 
        protein_features_map(entrezid, inp_data, prot_fun=prot_mean_max, at_least_prot=4, e2e=e2e, saving_filename=saving_filename) 
    elif mode == 'standard':
        # 1 drug -- all associated drugs
        saving_filename = 'dataset/triplets_dataset'
        if ae:
            saving_filename += ae
        if e2e:
            saving_filename += '_e2e'
        protein_features_map(entrezid, inp_data, prot_fun=prot_sum_max, e2e=e2e, saving_filename=saving_filename)  
    else:
        # Create testset with a subset of proteins
        filename = 'results_e2e_strat_neg_distrib/test_drug_name_triplets_stratified_e2e.p' #test_drug_name.p' ## path of the list of drug's names considered as a testset
        drug_list = dill.load(open(filename,'rb'))
        saving_filename = 'perturbed_test_set_triplet_strat'
        if ae:
            saving_filename += ae
        if e2e:
            saving_filename += '_e2e'
        create_protein_subset_testset(drug_list, inp_data, e2e=e2e, saving_path=saving_filename + ".p")
