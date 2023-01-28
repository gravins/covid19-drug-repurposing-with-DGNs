import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProd(torch.nn.Module):
    def __init__(self, prot_dim=None, drug_dim=None, hidden_dim=None):
        super().__init__()
        self.prot_dim=None
        self.drug_dim=None
        self.hidden_dim = None
        self.prot_emb = None
        self.drug_emb = None
        self.loss_fun = torch.nn.BCEWithLogitsLoss() 
        if prot_dim and drug_dim and hidden_dim:
            self.set_params({'prot_dim': prot_dim,
                             'drug_dim': drug_dim,
                             'hidden_dim': hidden_dim})

    def set_params(self, config):
        self.prot_dim = config['prot_dim']
        self.drug_dim = config['drug_dim']
        self.hidden_dim = config['hidden_dim']

        self.prot_emb = nn.Linear(self.prot_dim, self.hidden_dim)
        self.drug_emb = nn.Linear(self.drug_dim, self.hidden_dim)



    def forward(self, data, device):
        assert self.drug_emb is not None and self.prot_emb is not None, 'The model was not initialized'

        prot, drug = data['gene_emb'].to(device), data['drug_emb'].to(device)
        prot = self.prot_emb(prot)
        drug = self.drug_emb(drug)

        x = (prot * drug).sum(dim=1) # this is the equivalent of dot product, but with a batch of tensors

        loss = self.loss(x, data['y'].to(device)) if 'y' in data else None
        return loss, x

    def loss(self, x, y):
        return self.loss_fun(x,y)
 

'''
class BaseMLP(torch.nn.Module):
    def __init__(self, 
                 input_size=None, 
                 hidden_size=None, 
                 output_size=None):

        super().__init__()

        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        self.lin = None
        self.loss_fun = torch.nn.BCEWithLogitsLoss() 
        
        if input_size and hidden_size and output_size:
            self.set_params({'input_size': input_size,
                            'hidden_size': hidden_size,
                            'output_size': output_size})

    def set_params(self, config):
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']

        self.lin = nn.ModuleList()

        if isinstance(self.hidden_size, list):
            self.lin.append(nn.Linear(self.input_size, self.hidden_size[0]))
            self.lin.append(nn.ReLU())
            for i in range(1, len(self.hidden_size)):
                self.lin.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
                self.lin.append(nn.ReLU())
            self.lin.append(nn.Linear(self.hidden_size[-1], self.output_size))

        else:
            self.lin.append(nn.Linear(self.input_size, self.hidden_size))
            self.lin.append(nn.ReLU())
            self.lin.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, data, device):
        assert self.lin, 'The model was not initialized'

        x = data['x'].to(device)

        for layer in self.lin:
            x = layer(x)

        loss = self.loss(x.squeeze(1), data['y'].to(device)) if 'y' in data else None
        return loss, x

    def loss(self, x, y):
        return self.loss_fun(x,y)
'''
