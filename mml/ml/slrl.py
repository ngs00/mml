import random
import torch
from torch_geometric.data import Batch
import torch.nn.functional as F


def get_pairs(batch):
    num_data = len(batch)
    pos_list = list()
    neg_list = list()

    for anc in batch:
        target = anc.y
        idx = random.sample(range(0, num_data), 2)

        if abs(target - batch[idx[0]].y) < abs(target - batch[idx[1]].y):
            pos_list.append(batch[idx[0]])
            neg_list.append(batch[idx[1]])
        else:
            pos_list.append(batch[idx[1]])
            neg_list.append(batch[idx[0]])

    return pos_list, neg_list


def train(emb_net, optimizer, data_loader):
    emb_net.train()
    train_loss = 0

    for i, (anc, pos, neg) in enumerate(data_loader):
        anc.batch = anc.batch.cuda()
        pos.batch = pos.batch.cuda()
        neg.batch = neg.batch.cuda()

        emb_anc = F.normalize(emb_net(anc), p=2, dim=1)
        emb_pos = F.normalize(emb_net(pos), p=2, dim=1)
        emb_neg = F.normalize(emb_net(neg), p=2, dim=1)

        dist_ratio_x = torch.norm(emb_anc - emb_pos, dim=1) / (torch.norm(emb_anc - emb_neg, dim=1) + 1e-5)
        dist_ratio_x = -torch.exp(-dist_ratio_x + 1)
        dist_ratio_y = torch.norm(anc.y - pos.y, dim=1) / (torch.norm(anc.y - neg.y, dim=1) + 1e-5)
        dist_ratio_y = -torch.exp(-dist_ratio_y + 1)

        loss = torch.mean((dist_ratio_x - dist_ratio_y)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def test(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            embs = emb_net(batch)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def collate(batch):
    pos_list, neg_list = get_pairs(batch)

    return Batch.from_data_list(batch), Batch.from_data_list(pos_list), Batch.from_data_list(neg_list)
