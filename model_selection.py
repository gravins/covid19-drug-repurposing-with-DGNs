import torch

import os
import ray
import dill
import tqdm
import itertools
import pandas as pd
from _split import *
from datetime import datetime
from collections import defaultdict
from torch_geometric.data import DataLoader
from train import train_and_eval, format_time
from sklearn.model_selection import StratifiedShuffleSplit


def split(y, eval_size=0.1, groups=None, n_splits=1, seed=9):
    """
    :param y: the target variable for supervised learning problems 
    :param eval_size: evaluation set size, type: float or int. Default 0.1
    :param groups: attribute associated to each sample, whose value distribution is preserved in each fold.
                   If None the stratified splitting over the target varable is performed
    :param n_splits: number of re-shuffling and splitting iterations
    :param seed: seed used for random number generator
    :return two lists containing the fold's index list
    """

    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=seed) if not groups \
                else DAStratifiedSplit(n_splits=n_splits, test_size=eval_size, random_state=seed)

    tr_folds, eval_folds = [], []
    for tr_ids, eval_ids in splitter.split(np.zeros(len(y)), y, groups):
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


def max_tasks():
    return max(1, (os.cpu_count() / int(os.getenv("OMP_NUM_THREADS", 1)))*0.75)  # 75% of available threads


def model_selection(k, configurations, model, data, epochs, groups=None, batch_size=512, csv_path='dataframe.csv', device='cpu', seed=9, max_concurrent_tasks=max_tasks()):
    """
    Perform a model selection phase through standard validation or k-fold model selection. 
    All the results are saved into a DataFrame and the best configuration is returned.

    :param k: int or float. If int, k represent the number of folds in the k-fold model selection, 
                            if float, k in (0,1) represent the percentage of the validation set size
    :param configurations: dict. Dictionary of parameters, to each parameters is associated a list of possible values
    :param data: array-like. Problem's data, it is a list of dicts with at least x and y keys e.g. [{'x':vx1, 'y':vy1, ... } ... {'x':vxn, 'y':vyn, ... }]
    """

    assert ray.is_initialized() == True, "Ray is not initialized"
    assert k > 0 and k != 1, "k must be greater than 0"

    if k > 1:
        eval_size = int(len(data) // k) 
        if not isinstance(k,int):
           k = int(k)
    else:
        eval_size = k
        k = 1

    y = [d['y'] for d in data]
    tr_folds, vl_folds = split(y, eval_size, groups, k, seed)

    conf_list = extraploate_hyperparameters(configurations)

    final_results = defaultdict(list)
    ids_to_configs = {}
    for train_ids, evaluation_ids in tqdm.tqdm(zip(tr_folds, vl_folds)):
        # Start training
        for conf_id, conf in conf_list.items():
            tr_data = [data[i] for i in train_ids]
            vl_data = [data[i] for i in evaluation_ids]

            tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
            vl_loader = DataLoader(vl_data, batch_size=batch_size)

            remote_id = train_and_eval.remote(model, conf, epochs, tr_loader, vl_loader, device=device, mode="Validation", save_best=False)
            ids_to_configs[remote_id] = conf_id

            # Collect results if the number of started tasks > max_concurrent_tasks
            while len(ids_to_configs) > max_concurrent_tasks:
                # Return first task done
                done_ids, _ = ray.wait(list(ids_to_configs.keys()), num_returns=1)
                i = done_ids[0]

                res = ray.get(i)

                conf_id = ids_to_configs.pop(i)
                final_results[conf_id].append(res['history'][res['best_epoch']])

    # Wait and collect results
    while ids_to_configs:
        # Return first task done
        done_ids, _ = ray.wait(list(ids_to_configs.keys()), num_returns=1)
        i = done_ids[0]

        res = ray.get(i)

        conf_id = ids_to_configs.pop(i)
        final_results[conf_id].append(res['history'][res['best_epoch']])

    # Store results into a dataframe
    df = defaultdict(list)
    for conf_id in final_results.keys():
        # Add to the dataframe the model configuration
        df['conf id'].append(conf_id) 
        for key, value in conf_list[conf_id].items():
            df[key].append(value)

        # Add to the dataframe the model results
        res = final_results[conf_id]
        tmp = defaultdict(list)
        for i in range(k):
            for key in res[i]['Training'].keys():
                if 'time' in key:
                    tmp['tr '+key].append(
                        datetime.strptime(res[i]['Training'][key], "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S"))
                    tmp['vl '+key].append(
                        datetime.strptime(res[i]['Validation'][key], "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S"))
                else:
                    tmp['tr '+key].append(res[i]['Training'][key])
                    tmp['vl '+key].append(res[i]['Validation'][key])
        for key in tmp.keys():
            # Compute the avg and std over the folds
            if 'time' in key:
                mean = format_time(np.mean(tmp[key]).seconds)
            else:            
                mean = round(np.mean(tmp[key]), 6)
                std = (round(np.std(tmp[key]), 6) if 'confusion_matrix' not in key 
                       else None)

            df[key + ' mean'].append(mean)
            if 'time' not in key:
                df[key + ' std'].append(std)

    df = pd.DataFrame(df).sort_values('vl roc_auc mean', ascending=False)
    df.to_csv(csv_path, index=False)

    # Return best configuration
    imax = df.loc[df['vl roc_auc mean'].idxmax(), 
                  'conf id']
    best_conf = conf_list[imax]
    return best_conf
