import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.autograd import Variable
import time
import torch.nn.functional as F

def pairwise_distance(x, y):
    if len(y.size()) == 1:
        y = y.view(1, -1)
    xp = -2 * torch.mm(x, y.t())
    xp += torch.sum(x * x, 1).expand(xp.t().size()).t()
    xp += torch.sum(y * y, 1).expand(xp.size())
    return torch.sqrt(xp)

def pairwise_dist3(x, y):
    if len(y.size()) == 1:
        y = y.view(1, -1)
    s = -3* (x*x).t() @ y
    s += 3 * x.t() @ (y * y)
    s += (x * x * x).expand(s.t().size()).t()
    s += - (y * y * y).expand(s.size())
    # pow3 = torch.pow(torch.abs(s), 0.33)
    pow3 = torch.pow(torch.abs(s+0.000001), 0.33)
    pow3.data *= torch.from_numpy(np.sign(s.data.cpu().numpy())).cuda()
    return pow3

def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = ((rx.t() + ry) - 2*zz)
    sq = torch.sqrt(P)
    sq.data *= (zz.data / torch.abs(zz.data))
    return torch.log(1 + torch.exp(-1     *sq))

class RankNet(nn.Module):
    def __init__(self, dim, hidden_size=1, use_cuda=True):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(dim, 50)
        self.fc2 = nn.Linear(50, 1)
        self.use_cuda = use_cuda
        self.gamma = 1
    def forward(self, data, ones, zeros):
        loss = torch.zeros(1)
        if self.use_cuda:
            loss = loss.cuda()
            # ones = ones.
        loss = Variable(loss)
        x = Variable(data)
        # print(self.fc1.weight.size(), x.size())
        # print(ones)
        scores = self.fc1(x)
        scores = self.fc2(F.tanh(scores))
        rp = torch.randperm(ones.size(0)).cuda()
        rp2 = torch.randperm(zeros.size(0)).cuda()
        # print(scores.size())
        dist = pairwise_dist3(scores.t(), scores.t())
        logs = torch.log(1+torch.exp(-self.gamma*dist))
        t2 = time.time()
        # for i in range(data.size(0)):
        #     for j in range(data.size(1)):
        #         gg = torch.log(1 + torch.exp(-self.gamma * (scores[i] - scores[j])))
        #         ddd = logs[i,j] - gg
        #         if ddd.data[0] > 0.01:
        #             print('oops --> ', ddd.data[0], dist[i,j].data[0], scores[i].data[0], scores[j].data[0])
        #
        #         # ddd = logs[i,j] - torch.log(1 + torch.exp(-self.gamma * (scores[i] - scores[j])))[0]
        #
        #         # if torch.abs(ddd) > 0.
        # print(' other', time.time() - t2)
        torch.cuda.synchronize()
        # print(ones.size(), zeros.size())
        count = 0
        # for i in rp[:100]:
        #     for j in rp2[:100]:
        #         # if scores[i] > scores[j]:
        #         # print(torch.log(1 + torch.exp(-self.gamma * (scores[i] - scores[j]))).size())
        #         # torch.cuda.synchronize()
        #         # print(scores[i], scores[j])
        #         count += 1
        #         loss += torch.log(1 + torch.exp(-self.gamma * (scores[i] - scores[j])))[0]
        #         # print('we ', i, ' ', j)
        #         # print(torch.log(1 + torch.exp(-self.gamma * (scores[i] - scores[j])))[0] - logs[i,j])
        #         # print(scores[i] - scores[j], dist[i,j])
        #         # print('oo')
        # print(loss / count)
        # print('mins ', torch.min(logs).data[0], torch.max(logs).data[0], torch.sum(logs).data[0] / (logs.size(0) * logs.size(1)))
        return torch.sum(logs[:5, 0]) / (logs.size(0) * logs.size(1))

import pandas as pd
top1000_data = pd.read_pickle('top1000_data.pd.pkl')
train_data = pd.read_pickle('train_data.pd.pkl')
test_data = pd.read_pickle('test_data.pd.pkl')

# ---

# m = train_data.as_matrix()
# train_data.loc[:, train_data.columns != 'rel']

cols = ['ext_doc_id', 'query_id', 'doc_len', 'query_len', 'abs_discounting', 'abs_discounting^10', 'abs_discounting^2', 'abs_discounting^3', 'abs_discounting^4', 'abs_discounting^5', 'abs_discounting^6', 'abs_discounting^7', 'abs_discounting^8', 'abs_discounting^9', 'bm25', 'bm25^10', 'bm25^2', 'bm25^3', 'bm25^4', 'bm25^5', 'bm25^6', 'bm25^7', 'bm25^8', 'bm25^9', 'dirichlet_prior', 'dirichlet_prior^10', 'dirichlet_prior^2', 'dirichlet_prior^3', 'dirichlet_prior^4', 'dirichlet_prior^5', 'dirichlet_prior^6', 'dirichlet_prior^7', 'dirichlet_prior^8', 'dirichlet_prior^9', 'jelinek_mercer', 'jelinek_mercer^10', 'jelinek_mercer^2', 'jelinek_mercer^3', 'jelinek_mercer^4', 'jelinek_mercer^5', 'jelinek_mercer^6', 'jelinek_mercer^7', 'jelinek_mercer^8', 'jelinek_mercer^9', 'plm-circle', 'plm-circle^10', 'plm-circle^2', 'plm-circle^3', 'plm-circle^4', 'plm-circle^5', 'plm-circle^6', 'plm-circle^7', 'plm-circle^8', 'plm-circle^9', 'plm-cosine', 'plm-cosine^10', 'plm-cosine^2', 'plm-cosine^3', 'plm-cosine^4', 'plm-cosine^5', 'plm-cosine^6', 'plm-cosine^7', 'plm-cosine^8', 'plm-cosine^9', 'plm-gauss', 'plm-gauss^10', 'plm-gauss^2', 'plm-gauss^3', 'plm-gauss^4', 'plm-gauss^5', 'plm-gauss^6', 'plm-gauss^7', 'plm-gauss^8', 'plm-gauss^9', 'plm-passage', 'plm-passage^10', 'plm-passage^2', 'plm-passage^3', 'plm-passage^4', 'plm-passage^5', 'plm-passage^6', 'plm-passage^7', 'plm-passage^8', 'plm-passage^9', 'plm-triangle', 'plm-triangle^10', 'plm-triangle^2', 'plm-triangle^3', 'plm-triangle^4', 'plm-triangle^5', 'plm-triangle^6', 'plm-triangle^7', 'plm-triangle^8', 'plm-triangle^9', 'tfidf', 'tfidf^10', 'tfidf^2', 'tfidf^3', 'tfidf^4', 'tfidf^5', 'tfidf^6', 'tfidf^7', 'tfidf^8', 'tfidf^9', 'rel']

train_data = train_data[cols]
test_data = test_data[cols]
top1000 = cols[:-1]
top1000_data = top1000_data[top1000]

# ---

final_list = []
final_rels = []
for array in train_data.groupby('query_id').apply(lambda x: x.as_matrix()):
    final_array = []
    rels = []
    for arr in array:
        final_array.append(arr[4:-1])
        rels.append(arr[-1])
    final_rels.append(np.array(rels, dtype=np.float32))
    final_list.append(np.array(final_array, dtype=np.float32))
# print(final_list[0], type(final_list[0]))
# final_list[0]
tensor_list = [torch.from_numpy(x[:, [0,10, 20, 30, 40, 50, 60, 70, 80, 90]]).cuda() for x in final_list]
# print(tensor_list[0])
rels = [torch.from_numpy(x).cuda() for x in final_rels]
# print(rels[0])

indexes = [(torch.nonzero(x), torch.nonzero(1-x)) for x in rels]
# print(tensor_list[0].size())
CUDA = True
model = RankNet(10)
if CUDA:
    model = model.cuda()

# means = [torch.mean(x, 0) for x in tensor_list]
# tensor_list = [tensor_list[i] / means[i] for i in range(len(tensor_list))]
model.train()

optimizer = optim.Adam(params=model.parameters())

for e in range(10):
    model.train()

    start = time.time()
    for q in range(len(tensor_list)):
        r = np.random.randint(0,len(tensor_list))
        ones, zeros = indexes[r]
        if ones.size() == torch.Size([]):
            continue
        optimizer.zero_grad()
        loss = model(tensor_list[r], ones, zeros)
        print(loss.data[0])
        loss.backward()
        optimizer.step()
    print('t     ', time.time() - start)
