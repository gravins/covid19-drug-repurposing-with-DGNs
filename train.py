import torch

import os
import ray
import time
import dill
import random
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

from ray.services import get_node_ip_address

# Set random seed
seed = 9
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def scoring(y_true, y_pred, labels=None):
    return {"confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred) if (0 in y_true) and (1 in y_true) else -1,
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)}


def train(model, optimizer, train_dataloader, device='cpu'):
    model.train()
    for batch in train_dataloader:
        # Reset gradients from previous step
        model.zero_grad()

        # Perform a forward pass
        loss, preds = model.forward(batch, device)

        # Perform a backward pass to calculate the gradients
        loss.backward()

        # Update parameters
        optimizer.step()


def evaluate(model, eval_dataloader, device='cpu'):
    t0 = time.time()
    model.eval()
    y_true, y_preds, y_preds_confidence = [], [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            # Perform the forward pass
            loss, preds = model.forward(batch, device)
        y_preds_confidence += torch.sigmoid(preds).detach().cpu().squeeze().tolist()
        preds = (torch.sigmoid(preds) > 0.5).float()
        y_preds += preds.detach().cpu().squeeze().tolist()
        y_true += batch['y'].detach().cpu().squeeze().tolist() if 'y' in batch else []

    scores = {'loss': loss.detach().cpu().item(),
              'time': format_time(time.time() - t0)}
    if 'y' in batch:
        # Compute scores
        scores.update(scoring(y_true, y_preds))

    return scores, (y_true, y_preds, y_preds_confidence)


@ray.remote(num_cpus=int(os.getenv('OMP_NUM_THREADS',1)))
def train_and_eval(model, config, epochs, train_dataloader, eval_dataloader=None, device="cpu", mode="Validation",\
                   save_best=False, path_save_best='best_epoch_model.pth'):
    history = []
    total_time = time.time()

    print('ip_addr:', get_node_ip_address(), 'train', config)

    model.set_params(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.to(device)

    max_auc = -1
    best_epoch = 0
    for e in range(epochs):
        t0 = time.time()

        train(model, optimizer, train_dataloader, device)
        
        # Evaluate the model on the training set
        train_scores, _ = evaluate(model, train_dataloader, device)
        tr_time = format_time(time.time() - t0)
        
        # Evaluate the model on the evaluation set
        if eval_dataloader:
            eval_scores, _ = evaluate(model, eval_dataloader, device)

        # Record all statistics from this epoch
        train_scores['time'] = tr_time
        h = {'epoch': e + 1,
             'Training': train_scores}
        if eval_dataloader:
            h.update({mode: eval_scores})

            if eval_scores['roc_auc'] >= max_auc:
                max_auc = eval_scores['roc_auc']
                best_epoch = e

                # Save model with highest evaluation score
                if save_best:
                    torch.save(model, path_save_best) ## then later the_model = torch.load(PATH)

        history.append(h)

    print(config, " - Total training took {:} (hh:mm:ss)".format(format_time(time.time()-total_time)))

    return {'history': history, 'best_epoch': best_epoch}
