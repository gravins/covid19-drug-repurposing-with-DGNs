#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import dill
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from train import evaluate
from dataset import DrugRepurposing
from torch_scatter import scatter_std
from torch_geometric.data import DataLoader

from models.mlp import MLP
from models.e2e import GraphAutoEncoder_e2e

# Set random seed
seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', dest='dataset_path')
    parser.add_argument('--model-path', dest='model_path')
    parser.add_argument('--exp-name', dest='exp_name')
    parser.add_argument('--mode', dest='mode', choices=['Repurposing', 'SubsetTest'])
    parser.add_argument('--p', dest='percentage', default=None, choices=['0.75', '0.50', '0.25'])
    parser.add_argument('--csv', dest='csv_path', default='repurposing_results.csv')
    args = parser.parse_args()


    custom_test = (lambda data: data[args.percentage]['data']  if args.mode == 'SubsetTest'
                   else lambda data: data)
    data = DrugRepurposing(args.dataset_path, transform=custom_test)
    evaluation_dataloader = DataLoader(data[range(len(data))], batch_size=512)
    
    # Load the model
    model = torch.load(args.model_path)

    device = torch.device("cpu")
    model = model.to(device)
    print('Total parameters: ', sum(p.numel() for p in model.parameters()))
    print('Trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.mode == 'SubsetTest':
        # Run subset test
        # test the robustness of the model by giving as input a subsample of proteins per each drug in the set
        
        scores, (_, y_pred, _) = evaluate(model, evaluation_dataloader, device)

        # Make the tags consequent numbers starting from 0
        tags = dill.load(open(args.dataset_path,'rb'))[args.percentage]['batch']
        batches, prev, index = [], None, -1
        for t in tags:
            if not prev or t != prev:
                index += 1
                prev = t
            batches.append(index)
        batches = torch.tensor(batches).to(device)

        avg_std = torch.mean(scatter_std(torch.tensor(y_pred).squeeze(1), batches).float()).detach().cpu().item()

        print('Scores:', scores)
        print('avg(STD):', avg_std)

        with open(args.exp_name + '.txt', 'w') as f:
            f.write(args.exp_name + '\n')
            f.flush()
            f.write('Scores: ' + str(scores) + '\n')
            f.flush()
            f.write('Avg(std): ' + str(avg_std) +'\n')
            f.flush()
            f.close()
    else:
        scores, (_, y_pred, y_pred_confidence) = evaluate(model, evaluation_dataloader, device=device)
        err = False
        try:
            drug_names = data['drug'] 
        except KeyError:
            err = True
            print("'drug' key is missing in the data. No drugs' names information will be added to the DataFrame.")

        df = {'Predicted class': y_pred,
              'Probability of class 1': y_pred_confidence}
        if not err:
            df.update({'Drug name': drug_names})

        df = pd.DataFrame(df, columns=sorted(df.keys()))
        df.to_csv(args.csv_path)

    elapsed = time.time() - t0
    print(str(datetime.timedelta(seconds=int(round((elapsed))))))


