import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import CGConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from ml.data import *


def get_gnn(gnn, n_node_feats, dim_out, n_bond_feats=None):
    if gnn == 'gcn':
        return GCN(n_node_feats, dim_out)
    elif gnn == 'gat':
        return GAT(n_node_feats, dim_out)
    elif gnn == 'gin':
        return GIN(n_node_feats, dim_out)
    elif gnn == 'mpnn':
        return MPNN(n_node_feats, n_bond_feats, dim_out)
    elif gnn == 'cgcnn':
        return CGCNN(n_node_feats, n_bond_feats, dim_out)
    elif gnn == 'tfnn':
        return TFNN(n_node_feats, n_bond_feats, dim_out)
    else:
        raise AssertionError('Unknown GNN method \'{}\' was given.'.format(gnn))


def train(model, data_loader, optimizer, criterion):
    model.train()
    train_loss = 0

    for batch in data_loader:
        batch.cuda()

        preds = model(batch)
        loss = criterion(batch.y, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.cuda()

            preds = model(batch)
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0)


def exec_gnn(dataset_train, dataset_test, n_bond_feats, gnn, exp_id, n_repeats):
    exp_id += '_' + gnn
    targets_test = numpy.array([d.y.item() for d in dataset_test]).reshape(-1, 1)
    loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=128)
    list_mae = list()
    list_r2 = list()

    for n in range(0, n_repeats):
        model = get_gnn(gnn=gnn,
                        n_node_feats=dataset_train[0].x.shape[1],
                        n_bond_feats=n_bond_feats,
                        dim_out=1).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        criterion = torch.nn.L1Loss()

        for epoch in range(0, 300):
            train_loss = train(model, loader_train, optimizer, criterion)
            print(epoch, train_loss)

        preds_test = test(model, loader_test).cpu().numpy()
        pred_results = numpy.hstack([targets_test, preds_test])

        list_mae.append(mean_absolute_error(targets_test, preds_test))
        list_r2.append(r2_score(targets_test, preds_test))

        df_results = pandas.DataFrame(pred_results)
        df_results.to_excel('results/preds_' + exp_id + '_' + str(n) + '.xlsx', index=None, header=None)

    return numpy.mean(list_mae), numpy.std(list_mae), numpy.mean(list_r2), numpy.std(list_r2)


class FNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(dim_in, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, dim_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        out = self.fc3(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for inputs, targets in data_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            preds = self(inputs)
            loss = criterion(targets, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.cuda()

                preds = self(inputs)
                list_preds.append(preds.cpu().detach())

        return torch.vstack(list_preds).numpy()


class GCN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(n_node_feats, 256)
        self.gc2 = GCNConv(256, 256)
        self.gc3 = GCNConv(256, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out


class GAT(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GAT, self).__init__()
        self.gc1 = GATConv(n_node_feats, 256)
        self.gc2 = GATConv(256, 256)
        self.gc3 = GATConv(256, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out


class GIN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GIN, self).__init__()
        self.gc1 = GINConv(nn.Linear(n_node_feats, 256))
        self.gc2 = GINConv(nn.Linear(256, 256))
        self.gc3 = GINConv(nn.Linear(256, 256))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out


class MPNN(nn.Module):
    def __init__(self, n_node_feats, n_bond_feats, dim_out):
        super(MPNN, self).__init__()
        self.fc_edge1 = nn.Sequential(nn.Linear(n_bond_feats, 64), nn.ReLU(), nn.Linear(64, n_node_feats * 64))
        self.gc1 = NNConv(n_node_feats, 64, self.fc_edge1)
        self.gn1 = LayerNorm(64)

        self.fc_edge2 = nn.Sequential(nn.Linear(n_bond_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc2 = NNConv(64, 64, self.fc_edge2)
        self.gn2 = LayerNorm(64)

        self.fc_edge3 = nn.Sequential(nn.Linear(n_bond_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc3 = NNConv(64, 64, self.fc_edge3)
        self.gn3 = LayerNorm(64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, dim_out)

    def forward(self, g):
        h = F.relu(self.gn1(self.gc1(g.x, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn3(self.gc3(h, g.edge_index, g.edge_attr)))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out


class CGCNN(nn.Module):
    def __init__(self, n_node_feats, n_bond_feats, dim_out):
        super(CGCNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_bond_feats)
        self.gn1 = LayerNorm(128)
        self.gc2 = CGConv(128, n_bond_feats)
        self.gn2 = LayerNorm(128)
        self.gc3 = CGConv(128, n_bond_feats)
        self.gn3 = LayerNorm(128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn3(self.gc3(h, g.edge_index, g.edge_attr)))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc2(hg))
        out = self.fc3(h)

        return out


class TFNN(nn.Module):
    def __init__(self, n_node_feats, n_bond_feats, dim_out):
        super(TFNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_bond_feats)
        self.gc2 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_bond_feats)
        self.gc3 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_bond_feats)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gc1(hx, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc2(hg))
        out = self.fc3(h)

        return out
