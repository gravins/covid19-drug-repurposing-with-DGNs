from sklearn.model_selection import StratifiedShuffleSplit
import itertools
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score, precision_score, 
                             recall_score, accuracy_score)
import os
import dill
import pandas
import tqdm 

def load_data(path):
    if os.path.exists(path):
        print(f'Found existing file at path {path}, loading it...')
        return dill.load(open(path, 'rb'))
    else:
        original_data = dill.load(open('triplets_dataset_single_prot_seq_emb.p','rb'))
        prot_seq = pandas.read_csv('protein_sequences.csv')

        train_id, test_id = dill.load(open('tr_ts_id.p','rb'))
        train = [original_data[i] for i in train_id]
        test = [original_data[i] for i in test_id]

        tr_folds, vl_folds = dill.load(open('folds.p','rb'))
        train_folds = []
        valid_folds = []
        for tr, vl in zip(tr_folds, vl_folds):
            train_folds.append([train[i] for i in tr])
            valid_folds.append([train[i] for i in vl])

        def convert_data(data):
            X_drugs = []
            X_proteins = []
            y = [] 
            for od in tqdm.tqdm(data):
                try:
                    prot = prot_seq[prot_seq['EntrezGeneID']==int(od['gene'])]['Seq'].item()
                except ValueError: 
                    print(od['gene'])
                    continue
                assert isinstance(prot, str)

                X_proteins.append(prot)
                X_drugs.append(od['smile'])
                y.append(od['y'])
            return X_drugs, X_proteins, y
        
        final_data = {
            'train':convert_data(train), # X_drug_train, X_prot_train, y_train
            'test': convert_data(test)
        }

        for i, (tr, vl) in enumerate(zip(train_folds, valid_folds)):
            final_data[f'fold {i}'] = (convert_data(tr), convert_data(vl))

        dill.dump(final_data, open(path,'wb'))
        return final_data

def split(y, eval_size=0.1, n_splits=1, seed=9):
    """
    :param y: the target variable for supervised learning problems 
    :param eval_size: evaluation set size, type: float or int. Default 0.1
    :param groups: attribute associated to each sample, whose value distribution is preserved in each fold.
                   If None the stratified splitting over the target varable is performed
    :param n_splits: number of re-shuffling and splitting iterations
    :param seed: seed used for random number generator
    :return two lists containing the fold's index list
    """

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=seed)
    tr_folds, eval_folds = [], []
    for tr_ids, eval_ids in splitter.split(np.zeros(len(y)), y):
        tr_folds.append(tr_ids)
        eval_folds.append(eval_ids)

    if len(tr_folds) == 1:
        tr_folds, eval_folds = tr_folds[0], eval_folds[0]

    return tr_folds, eval_folds


def extraploate_hyperparameters(conf):
    # Extrapolates all hyper-parameters name and values
    labels, terms = zip(*conf.items())
    conf_list = {i: dict(zip(labels, term)) for i, term in enumerate(itertools.product(*terms))}
    return conf_list


def compute_score(y_pred, y_true, labels=None):
    return {"confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred) if (0 in y_true) and (1 in y_true) else -1,
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)}


def aggregate_scores(f, scores):
    res = {}
    for k in scores[0].keys():
        if k == 'confusion_matrix':
            res[k] = f([s[k] for s in scores], axis=0)
        else:
            res[k] = f([s[k] for s in scores])
    return res

