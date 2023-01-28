import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
from .gae import GraphAutoEncoder, GNN, AutoEncoder

from .proteins_utils import NUM_ATOM_FEATURES, NUM_BOND_FEATURES
#NUM_PROTEIN_FEATURES = int(768/2)  # Single protein setting 
NUM_PROTEIN_FEATURES = 768         # Multi proteins setting

class GraphAutoEncoder_e2e(torch.nn.Module):
    def __init__(self,
                 input_size=NUM_ATOM_FEATURES,
                 edge_features_size=NUM_BOND_FEATURES,
                 protein_size=NUM_PROTEIN_FEATURES,
                 ae_hidden_layer=None, ae_batchnorm=False, ae_activation=F.relu, denoising=False,
                 gnn_num_layers=None, gnn_hidden_size=None, gnn_output_size=None,
                 mlp_hidden_size=None, mlp_output_size=None, mlp_batchnorm=False):

        super().__init__()

        self.input_size = None
        self.edge_features_size = None
        self.ae_hidden_layer = None
        self.ae_batchnorm = None
        self.ae_activation = None
        self.denoising = None
        self.gnn_num_layers = None
        self.gnn_hidden_size = None
        self.gnn_output_size = None
        self.mlp_hidden_size = None
        self.mlp_output_size = None
        self.mlp_batchnorm = None
        self.protein_size = None

        self.ae = None
        self.gnn = None
        self.gae = None
        self.lin = None
        self.mlp_loss_fun = torch.nn.BCEWithLogitsLoss() 
        
        if input_size and edge_features_size and gnn_num_layers and gnn_hidden_size and \
           gnn_output_size and mlp_hidden_size and mlp_output_size:

            self.set_params({'input_size': input_size,
                             'edge_features_size': edge_features_size,
                             'ae_hidden_layer': ae_hidden_layer,
                             'ae_batchnorm': ae_batchnorm,
                             'ae_activation': ae_activation,
                             'denoising': denoising,
                             'gnn_num_layers' : gnn_num_layers,
                             'gnn_hidden_size': gnn_hidden_size,
                             'gnn_output_size': gnn_output_size,
                             'mlp_hidden_size': mlp_hidden_size,
                             'mlp_output_size': mlp_output_size,
                             'mlp_batchnorm': mlp_batchnorm,
                             'protein_size': protein_size})

    def set_params(self, config):
        self.input_size = config.get('input_size', None) or NUM_ATOM_FEATURES
        self.edge_features_size = config.get('edge_features_size', None) or NUM_BOND_FEATURES
        self.ae_hidden_layer = config.get('ae_hidden_layer', None)
        self.ae_batchnorm = config.get('ae_batchnorm', None) or False
        self.ae_activation = config.get('ae_activation', None) or F.relu
        self.denoising = config.get('denoising', None) or False
        self.gnn_num_layers = config['gnn_num_layers']
        self.gnn_hidden_size = (config['gnn_output_size'] if config['gnn_hidden_size'] is None
                               else config['gnn_hidden_size'])
        self.gnn_output_size = (config['gnn_hidden_size'] if config['gnn_output_size'] is None
                               else config['gnn_output_size'])
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.mlp_output_size = config['mlp_output_size']
        self.mlp_batchnorm = config.get('mlp_batchnorm', None) or False
        self.protein_embedding_size = config.get('protein_size', None) or NUM_PROTEIN_FEATURES
        
        # graph embedder
        self.ae = AutoEncoder(self.gnn_output_size, self.ae_hidden_layer, self.ae_batchnorm, self.ae_activation)
        self.gnn = GNN(self.gnn_num_layers, self.gnn_hidden_size, self.gnn_output_size, self.input_size, self.edge_features_size)
        self.gae = GraphAutoEncoder(self.gnn, self.ae, self.denoising)

        # final MLP
        self.lin = nn.ModuleList()

        inp = self.ae_hidden_layer[-1] if self.ae_hidden_layer else self.gnn_output_size // 2
        inp += self.protein_embedding_size
        if isinstance(self.mlp_hidden_size, list):
            self.lin.append(nn.Linear(inp, self.mlp_hidden_size[0]))
            for i in range(1, len(self.mlp_hidden_size)):
                self.lin.append(nn.Linear(self.mlp_hidden_size[i - 1], self.mlp_hidden_size[i]))
                if self.mlp_batchnorm:
                    self.lin.append(nn.BatchNorm1d(self.mlp_hidden_size[i]))
                self.lin.append(nn.ReLU())
            self.lin.append(nn.Linear(self.mlp_hidden_size[-1], self.mlp_output_size))

        else:
            self.lin.append(nn.Linear(inp, self.mlp_hidden_size))
            if self.mlp_batchnorm:
                self.lin.append(nn.BatchNorm1d(self.mlp_hidden_size))
            self.lin.append(nn.ReLU())
            self.lin.append(nn.Linear(self.mlp_hidden_size, self.mlp_output_size))

    def forward(self, graphs_batch, device):
        assert self.lin, 'The model was not initialized'

        loss, _, _, x = self.gae(graphs_batch.to(device))

        # concat prototeins and drug embeddings
        x = torch.cat((graphs_batch['gene_emb'], x),dim=1).squeeze(0).to(device)
        
        for layer in self.lin:
            x = layer(x)
        
        loss = self.loss(x.squeeze(1), graphs_batch['y'], loss) if 'y' in graphs_batch else None
        return loss, x

    def loss(self, x, y, ae_loss):
        return self.mlp_loss_fun(x,y) + ae_loss
