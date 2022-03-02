import numpy
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Batch


n_smp = 4
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
y_var = None


def set_y_vars(dataset):
    targets = numpy.array([x.y.item() for x in dataset]).reshape(-1, 1)
    p_dist = pairwise_distances(targets, targets)
    y_vars = torch.tensor(numpy.var(p_dist, axis=1), dtype=torch.float)

    for i in range(0, len(dataset)):
        dataset[i].y_var = y_vars[i].view(1, 1)

    global y_var
    y_var = torch.tensor(numpy.var(p_dist), dtype=torch.float)

    return y_vars


def collate(batch):
    n_data = len(batch)
    targets = [x.y for x in batch]
    anc = list()
    smp = list()
    y_dists = list()
    y_vars = list()

    for i in range(0, n_data):
        y_vars.append(batch[i].y_var)

        for j in range(0, n_data):
            anc.append(batch[i])
            smp.append(batch[j])
            y_dists.append(torch.norm(targets[i] - targets[j]).view(1, 1))

    return Batch.from_data_list(anc), Batch.from_data_list(smp), torch.tensor(y_dists).view(-1, 1).cuda(), torch.tensor(y_vars).view(-1, 1).cuda()


def train(emb_net, data_loader, optimizer, coeff_cos_sim):
    emb_net.train()
    sum_loss = 0

    for i, (anc, smp, y_dists, y_vars) in enumerate(data_loader):
        anc.batch = anc.batch.cuda()
        smp.batch = smp.batch.cuda()

        emb_anc = F.normalize(emb_net(anc), p=2, dim=1)
        emb_smp = F.normalize(emb_net(smp), p=2, dim=1)
        z_dists = torch.norm(emb_anc - emb_smp, dim=1).view(-1, 1)

        loss = 0
        pos = 0
        for j in range(0, y_vars.shape[0]):
            r2 = torch.sum((y_dists[pos:pos + n_smp, :] - z_dists[pos:pos + n_smp, :]) ** 2) / y_var
            cos = torch.mean(cos_sim(emb_anc, emb_smp))
            loss += r2 - coeff_cos_sim * cos
            pos += n_smp
        loss /= y_vars.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    return sum_loss / len(data_loader)


def test(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            embs = F.normalize(emb_net(batch), p=2, dim=1)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)
