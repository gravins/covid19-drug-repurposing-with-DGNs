import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, 
                 input_size=None, 
                 hidden_size=None, 
                 output_size=None,
                 use_batchnorm=None):

        super().__init__()

        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        self.use_batchnorm = None
        self.lin = None
        self.loss_fun = torch.nn.BCEWithLogitsLoss() 
        
        if input_size and hidden_size and output_size:
            self.set_params({'input_size': input_size,
                            'hidden_size': hidden_size,
                            'output_size': output_size,
                            'use_batchnorm':use_batchnorm})

    def set_params(self, config):
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.use_batchnorm = config.get('use_batchnorm', False)

        self.lin = nn.ModuleList()

        if isinstance(self.hidden_size, list):
            self.lin.append(nn.Linear(self.input_size, self.hidden_size[0]))
            for i in range(1, len(self.hidden_size)):
                self.lin.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
                if self.use_batchnorm:
                    self.lin.append(nn.BatchNorm1d(self.hidden_size[i]))
                self.lin.append(nn.ReLU())
            self.lin.append(nn.Linear(self.hidden_size[-1], self.output_size))

        else:
            self.lin.append(nn.Linear(self.input_size, self.hidden_size))
            if self.use_batchnorm:
                self.lin.append(nn.BatchNorm1d(self.hidden_size))
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




