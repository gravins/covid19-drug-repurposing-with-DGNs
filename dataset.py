import torch
import dill
import random
import numpy as np
from torch.utils.data import Dataset


class DrugRepurposing(Dataset):
    def __init__(self, dill_path, transform=None):
         '''
         :param dill_path: dill path of a list of dicts containing at least 2 keys: x, y
         '''
         self.data = dill.load(open(dill_path, 'rb')) # list of dict containing at least the keys x, y

         if transform is not None:
              self.data = transform(self.data)

         print('Dataset Loaded!')

    def __len__(self):
         return len(self.data)

    def shuffle(self, seed=9):
         random.seed = seed
         random.shuffle(self.data)

    def __getitem__(self, idx):
         if isinstance(idx, str):
            return self.getitem_from_key(idx=range(len(self.data)), key=idx)

         if torch.is_tensor(idx):
            idx = idx.tolist()

         if isinstance(idx, list) or isinstance(idx, np.ndarray) or isinstance(idx, range):
            return [self.data[i] for i in idx]
         elif isinstance(idx, int):
            return self.data[idx]['x'], self.data[idx]['y']
         else:
            raise ValueError("idx must be int or list, not ", type(idx))


    def getitem_from_key(self, idx, key):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list) or isinstance(idx, np.ndarray) or isinstance(idx, range):
            return [self.data[i][key] for i in idx]
        elif isinstance(idx, int):
            return self.data[idx][key]
        else:
            raise ValueError("idx must be int or list, not ", type(idx))

