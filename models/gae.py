import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import GINEConv, global_add_pool
from torch_scatter import scatter_add

from .proteins_utils import NUM_ATOM_FEATURES, NUM_BOND_FEATURES


class Block(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 bottleneck=False,
                 use_batchnorm=False,
                 activation=F.relu):
        super().__init__()

        self.lin = nn.Linear(dim_input, dim_output)
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.bottleneck = bottleneck

        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(dim_output)

    def forward(self, x):
        x = self.lin(x)
        if self.bottleneck and self.use_batchnorm:
            x = self.batchnorm(x)
        if self.bottleneck:
            return x
        return self.activation(x)


class ReverseBlock(nn.Module):
    def __init__(self,
                 dim_input,
                 dim_output,
                 use_batchnorm=False,
                 activation=F.relu):
        super().__init__()

        self.lin = nn.Linear(dim_input, dim_output)
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(dim_input)

    def forward(self, x):
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x = self.activation(x)
        x = self.lin(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self,
                 dim_input,
                 hidden_layers=None,
                 use_batchnorm=False,
                 activation=F.relu):

        super().__init__()
        assert hidden_layers is None or isinstance(hidden_layers, list)

        self.dim_input = dim_input
        self.hidden_layers = hidden_layers or [dim_input // 2]

        blocks, reverse_blocks = [], []
        layers = [dim_input] + self.hidden_layers

        for i, dim_input in enumerate(layers[:-1]):
            dim_output = layers[i+1]
            is_bottleneck = (i+1) == len(layers) - 1
            block = Block(
                dim_input=dim_input,
                dim_output=dim_output,
                bottleneck=is_bottleneck,
                use_batchnorm=use_batchnorm,
                activation=activation)
            rev_block = ReverseBlock(
                dim_input=dim_output,
                dim_output=dim_input,
                use_batchnorm=use_batchnorm,
                activation=activation)
            blocks.append(block)
            reverse_blocks.append(rev_block)

        self.encoder = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(*reverse_blocks[::-1])

    def forward(self, x):
        h = self.encoder(x)
        x_rec = self.decoder(h)
        return x_rec, h


class GNN(nn.Module):
    def __init__(self,
                 num_layers,
                 dim_hidden,
                 dim_output,
                 dim_input=NUM_ATOM_FEATURES,
                 dim_edge_features=NUM_BOND_FEATURES):
        super().__init__()

        self.num_layers = num_layers
        self.dim_input = dim_input
        self.dim_edge_features = dim_edge_features
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output

        self.convs = nn.ModuleList([])
        self.edge_nets = nn.ModuleList([])

        for i in range(self.num_layers):
            dim_input = self.dim_input if i == 0 else self.dim_hidden
            dim_output = self.dim_output if i == self.num_layers - 1 else self.dim_hidden

            conv = GINEConv(nn=Block(dim_input, dim_output))
            self.convs.append(conv)

            dim_output = self.dim_input if i == 0 else self.dim_hidden
            edge_net = nn.Linear(self.dim_edge_features, dim_output, bias=False)
            self.edge_nets.append(edge_net)

        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    def aggregate_nodes(self, nodes_repr, batch):
        nodes_per_graph = scatter_add(torch.ones_like(batch), batch)
        nodes_per_graph = nodes_per_graph.repeat_interleave(nodes_per_graph.view(-1))
        nodes_per_graph = torch.sqrt(nodes_per_graph.view(-1, 1).float())
        graph_repr = global_add_pool(nodes_repr / nodes_per_graph, batch)
        return graph_repr

    def forward(self, x, edge_index, edge_attr, batch):
        nodes_repr = 0
        for conv, edge_net in zip(self.convs, self.edge_nets):
            x = conv(x, edge_index, edge_attr=edge_net(edge_attr))
            nodes_repr += x

        return self.aggregate_nodes(nodes_repr, batch)


class GraphAutoEncoder(nn.Module):
    def __init__(self, gnn, autoencoder, denoising=False):
        super().__init__()

        self.gnn = gnn
        self.autoencoder = autoencoder
        self.denoising = denoising

    def add_noise(self, repr):
        noise = torch.randn(repr.size()) * 0.2
        noisy_repr = repr + noise.to(repr.device)
        return noisy_repr

    def forward(self, graphs_batch):
        x, edge_index, edge_attr, batch = (
            graphs_batch['x'],
            graphs_batch['edge_index'],
            graphs_batch['edge_attr'],
            graphs_batch['batch'])

        graph_repr = self.gnn(x, edge_index, edge_attr, batch)
        if self.denoising:
           graph_repr_noisy = self.add_noise(graph_repr)
           graph_rec, hidden = self.autoencoder(graph_repr_noisy)
        else:
           graph_rec, hidden = self.autoencoder(graph_repr)
        loss = self.loss(graph_rec, graph_repr)
        return loss, graph_repr, graph_rec, hidden

    def loss(self, graph_rec, graph_repr):
        return F.mse_loss(graph_rec, graph_repr)
