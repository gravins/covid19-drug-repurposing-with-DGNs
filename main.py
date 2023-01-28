import torch

import os
import ray
import time
import dill
import random
import argparse
import datetime
import numpy as np
from utils import *
from train import *
from dataset import *
from models.baseline import DotProd
from models.mlp import MLP
from models.e2e import GraphAutoEncoder_e2e
from torch_geometric.data import DataLoader
from model_selection import model_selection, split

# Set random seed
seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#ray.init() # local ray initialization
ray.init(address=os.environ.get("ip_head"), _redis_password=os.environ.get("redis_password"))  # clustering ray initialization

print('Settings:')
print('\tKMP_SETTING:', os.environ.get('KMP_SETTING'))
print('\tOMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS'))
print('\tKMP_BLOCKTIME:', os.environ.get('KMP_BLOCKTIME'))
print('\tMALLOC_CONF:', os.environ.get('MALLOC_CONF'))
print('\tLD_PRELOAD:', os.environ.get('LD_PRELOAD'))

if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest='dataset_path')
    parser.add_argument('--name', dest='name')
    parser.add_argument('--mode', dest='mode', default='Validation', choices=['Validation', 'Test'])
    parser.add_argument('--k', dest='k', default=5, type=float)
    parser.add_argument('--epochs', dest='epochs', default=10000, type=int)
    parser.add_argument('--batch', dest='batch', default=512, type=int)
    parser.add_argument('--csv', dest='csv_path')
    parser.add_argument('--e2e', dest='e2e', action='store_true')
    parser.add_argument('--baseline', dest='baseline', required=False, choices=['MLP', 'DotProd'])
    parser.add_argument('--tag', dest='grouped_stratification', action='store_true')

    args = parser.parse_args()

    assert args.baseline or args.e2e, "Params e2e and baseline cannot be set simultaniously"

    data = DrugRepurposing(args.dataset_path)

    # Split data into Traingin and Test
    if args.grouped_stratification:
        train_ids, test_ids = split(y=data['y'], eval_size=0.2, groups=data['tag'], seed=seed)
        groups = data.getitem_from_key(train_ids, 'tag')
    else:
        train_ids, test_ids = split(y=data['y'], eval_size=0.2, seed=seed)
        groups = None

    dill.dump((test_ids, data.getitem_from_key(test_ids, 'drug')), open('test_drug_name_'+args.name+'.p', 'wb'))

    x, _ = data[0]
    if not args.baseline:
        model = MLP() if not args.e2e else GraphAutoEncoder_e2e(protein_size=x.size(0))
    else:
        model = MLP() if args.baseline == 'MLP' else DotProd()

    train_dataloader = DataLoader(data[train_ids], batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(data[test_ids], batch_size=args.batch)
    device = torch.device("cpu")

    if args.mode == 'Validation' and args.k is not None:
        # Run model selection
        if not args.k:
            args.k = 0.2

        if args.e2e:
            # Structural-Similarity-based prediction Network (SSN)
            configurations = {
                    'mlp_output_size': [1],
                    'mlp_hidden_size': [[512, 128, 32], [512, 64],
                                        [256, 64, 16], [256,32], 
                                        [128, 64, 32], [128,16]],
                    'mlp_batchnorm': [False, True],
                    'ae_batchnorm': [False],
                    'denoising': [True],
                    'gnn_num_layers': [3],
                    'gnn_hidden_size':[192],
                    'gnn_output_size':[192],
                    'lr': [2e-5, 2e-4, 2e-3]
            }

        elif args.baseline == 'DotProd':
            
            prot_dim = data.data[0]['gene_emb'].size(0)
            drug_dim = data.data[0]['drug_emb'].size(0)
            
            # Baseline DotProd(MLP(node2vec emb), MLP(MorganFP))
            configurations = { 
                    'prot_dim': [prot_dim], 
                    'drug_dim': [drug_dim],
                    'hidden_dim': [4096, 3000, 2048, 1024, 512, 256, 128, 64, 32],
                    'lr': [2e-5, 2e-4, 2e-3]
            }
        
        else:
            # Chemical-Similarity-based prediction Network (CSN)
            # and baseline concat(node2vec emb, MorganFP) + MLP
            configurations = { 
                    'input_size': [x.size(0)], 'output_size': [1],
                    'hidden_size': [[512, 128, 32], [512, 64], [512],
                                    [256, 64, 16], [256,32], [256],
                                    [128, 64, 32], [128,16], [128]],
                    'use_batchnorm': [True, False],
                    'lr': [2e-5, 2e-4, 2e-3]
            }

        max_tasks = 100
        conf = model_selection(args.k, configurations, model, data[train_ids], args.epochs, groups, args.batch, args.csv_path, device=device, max_concurrent_tasks=max_tasks)

    else:
        conf = {
                'mlp_output_size': [1],
                'mlp_hidden_size': [512, 64],
                'mlp_batchnorm': [False],
                'ae_batchnorm': [False],
                'denoising': [True],
                'gnn_num_layers': [3],
                'gnn_hidden_size':[192],
                'gnn_output_size':[192],
                'lr': 2e-3
        }
        
        '''
        conf = {
            'input_size': x.size(0), 'output_size': 1,
            'hidden_size': [512, 128, 32],
            'lr': 2e-5
        }
        '''

    # Train the best model on the whole training set and evaluate on the test set
    remote_id = train_and_eval.remote(model, conf, args.epochs, train_dataloader, test_dataloader, device,
                                      mode='Test', save_best=True, path_save_best="best_"+args.name+".pth")
    history = ray.get(remote_id)

    dill.dump({"epochs": args.epochs,
                "batch_size": args.batch,
                'model_structure': conf,
                'model_selection': configurations if args.mode == 'Validation' else None,
                "history": history}, open(args.name + "_history.p", "wb"))

    title = args.name.replace("_", " ")
    make_plots(history['history'], title, args.name, 'Test')

    elapsed = time.time() - t0
    print(str(datetime.timedelta(seconds=int(round((elapsed))))))
