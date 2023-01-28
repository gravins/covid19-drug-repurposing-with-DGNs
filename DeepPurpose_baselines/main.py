import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from DeepPurpose import DTI as models
from DeepPurpose import utils as dp_utils
import numpy as np
from utils import *
import argparse
import random
import torch
import tqdm
import time
import dill
from collections import defaultdict
import ray
import datetime
from train import train_ray
from torch import sigmoid

ray.init() # local ray initialization

# Set random seed
seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


## train_X_drugs[0]: CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N
## train_X_proteins[0]: MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL
## train_y[0]: 0.1

if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', dest='model', default='GraphDTA', choices=['GraphDTA', 'DeepDTA'])
    parser.add_argument('--epochs', dest='epochs', default=10000, type=int)
    parser.add_argument('--batch', dest='batch', default=512, type=int)
    parser.add_argument('--path', dest='path', default='.', type=str)
    
    args = parser.parse_args()

    if args.model == 'GraphDTA':
        # GraphDTA
        ## Other supported models DGL_GCN, DGL_NeuralFP, DGL_GIN_AttrMasking, DGL_GIN_ContextPred, DGL_AttentiveFP  
        configurations = {
            'drug_encoding': ['DGL_GCN'],
            'target_encoding': ['CNN'],
            'cnn_target_filters': [[32,64,96]],
            'cnn_target_kernels': [[4,8,12]],

            'gnn_num_layers': [3],
            'gnn_hid_dim_drug':[192],
            'hidden_dim_drug':[192],
            'gnn_activation': torch.relu
        } 
    elif args.model == 'DeepDTA':
        # DeepDTA
        configurations = {
            'drug_encoding': ['CNN'],
            'target_encoding': ['CNN'],

            'cnn_drug_filters': [[32,64,96]],
            'cnn_target_filters': [[32,64,96]],
            'cnn_drug_kernels': [[4,6,8]],
            'cnn_target_kernels': [[4,8,12]],
        }

    else:
        raise NotImplementedError()

    configurations.update({
        'cls_hidden_dims': [[512, 128, 32], [512, 64],
                            [256, 64, 16], [256,32], 
                            [128, 64, 32], [128,16]],
        'lr': [2e-5, 2e-4, 2e-3],
    })

    data = load_data(os.path.join(args.path, 'data.p'))

    folds = [k for k in data.keys() if 'fold' in k]
    results = {}

    conf_list = extraploate_hyperparameters(configurations)
    ray_ids = []
    for conf_id in conf_list:
        for f_id in tqdm.tqdm(folds):

            ray_ids.append(train_ray.remote(
                conf = conf_list[conf_id],
                conf_id = conf_id,
                f_id = f_id, 
                data = data, 
                batch = args.batch,
                epochs = args.epochs
            ))

    results = defaultdict(list)
    # Wait and collect results
    for id_ in tqdm.tqdm(ray_ids):
        res = ray.get(id_)

        (f_id, conf_id, y_pred_tr, y_pred_val, score_tr, score_val) = res

        results[conf_id].append({
            'conf': conf_list[conf_id],
            'f_id': f_id,
            'y_pred_tr': y_pred_tr, 
            'y_pred_val': y_pred_val, 
            'score_tr': score_tr, 
            'score_val': score_val}
        )

        dill.dump(results, open(os.path.join(args.path, f'model_selection_results_{args.model}.p'), 'wb'))
        
    for conf_id in results.keys():
        n_folds = len(results[conf_id])
        results[conf_id] = {
            'fold_res': results[conf_id], 
            'conf':results[conf_id][0]['conf'], 
            'avg tr_score': aggregate_scores(np.mean, [results[conf_id][i]['score_tr'] for i in range(n_folds)]), 
            'avg vl_score': aggregate_scores(np.mean, [results[conf_id][i]['score_val'] for i in range(n_folds)]), 
            'std tr_score': aggregate_scores(np.std, [results[conf_id][i]['score_tr'] for i in range(n_folds)]), 
            'std vl_score': aggregate_scores(np.std, [results[conf_id][i]['score_val'] for i in range(n_folds)])
        }

    dill.dump(results, open(os.path.join(args.path, f'model_selection_results_{args.model}.p'), 'wb'))

    
    vl_scores_list = [(results[cid]['avg vl_score'], cid) for cid in results.keys()]
    results['best score'] = max(vl_scores_list, key=lambda x: x[0]['roc_auc'])[1] # it is the conf_id key in the result dict
    dill.dump(results, open(os.path.join(args.path, f'model_selection_results_{args.model}.p'), 'wb'))

    # Get best conf and train again on the whole train/test
    best_conf = results[results['best score']]['conf']

    print(f'\n best conf \n {best_conf}\n')

    train = dp_utils.data_process(data['train'][0], data['train'][1], data['train'][2], 
                                  best_conf['drug_encoding'], best_conf['target_encoding'], 
                                  split_method='no_split')

    test = dp_utils.data_process(data['test'][0], data['test'][1], data['test'][2], 
                                 best_conf['drug_encoding'], best_conf['target_encoding'], 
                                 split_method='no_split')

    config = dp_utils.generate_config(
        drug_encoding = best_conf['drug_encoding'], 
        target_encoding = best_conf['target_encoding'], 
        batch_size = args.batch,
        train_epoch = args.epochs, 
        LR = best_conf['lr'],
        decay = 0.1, # similar to AdamW
        cnn_drug_filters = best_conf['cnn_drug_filters'] if 'cnn_drug_filters' in best_conf else None,
        cnn_drug_kernels = best_conf['cnn_drug_kernels'] if 'cnn_drug_kernels' in best_conf else None,
        cnn_target_filters = best_conf['cnn_target_filters'],
        cnn_target_kernels = best_conf['cnn_target_kernels'],
        cuda_id = None,
        gnn_hid_dim_drug = best_conf['gnn_hid_dim_drug'] if 'gnn_hid_dim_drug' in best_conf else None,
        gnn_num_layers = best_conf['gnn_num_layers'] if 'gnn_num_layers' in best_conf else None,
        gnn_activation = best_conf['gnn_activation'] if 'gnn_activation' in best_conf else None,        
        cls_hidden_dims = best_conf['cls_hidden_dims'] 
    )

    model = models.model_initialize(**config)
    model.train(train, test, verbose = True)
    y_pred_test = model.predict(test)
    print('test')
    print(compute_score((sigmoid(torch.tensor(y_pred_test)) > 0.5).float().tolist(), test['Label'].to_list())) 

    y_pred_train = model.predict(train)
    print('train')
    print(compute_score((sigmoid(torch.tensor(y_pred_train)) > 0.5).float().tolist(), train['Label'].to_list())) 

    dill.dump(model, open(os.path.join(args.path, f'final_{args.model}_model.p'), 'wb'))

    elapsed = time.time() - t0
    print(str(datetime.timedelta(seconds=int(round((elapsed))))))