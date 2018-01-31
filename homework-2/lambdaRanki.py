import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.autograd import Variable
import time
import torch.nn.functional as F
from collections import defaultdict
from lambdaRankUtils import *
from sklearn.model_selection import KFold
np.random.seed(32)
torch.manual_seed(32)

# takes as input two row vectors
def pairwise_dist2(x, y):
    y = y.view(1, -1)
    x = x.view(1, -1)
    s = -y.expand(x.size(1), y.size(1))
    s += x.expand(s.t().size()).t()
    return s


def RankNetLoss(scores, rels, gamma):
    dist = pairwise_dist2(scores.t(), scores.t())
    logs = torch.log(1 + torch.exp(-gamma *  dist ))
    # logs = dist

    mask = pairwise_dist2(rels, rels)
    mask = mask.clamp(min=0)
    # for i in range(mask.size(0)):
    #     for j in range(mask.size(1)):
    #         if dist[i,j].data != (scores[i] - scores[j]).data:
    #             print('weee')
    #             raise ' wedas'
    # logs.data = logs.data * mask

    tot_ones = torch.sum(mask)
    if tot_ones == 0:
        raise 'zzzz'
        print('weeee ', tot_ones)
        return torch.sum(logs)
    # print(torch.max(logs))
    # print(tot_ones,torch.sum(logs[mask > 0]).data[0])
    try:

        loss = torch.sum(logs[mask > 0] / tot_ones)
    except:
        print(logs)
    return loss


class RankNet(nn.Module):
    def __init__(self, dim, hidden_size=1, use_cuda=True):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(dim, 500)
        # self.fc2 = nn.Linear(2500, 2500)
        self.fc2 = nn.Linear(500, 1)
        self.use_cuda = use_cuda

    def forward(self, data):
        x = Variable(data)
        scores = self.fc1(x)
        scores = self.fc2(F.relu(scores))
        # scores = self.fc3(F.relu(scores))
        return F.sigmoid(scores)


def get_data(df, features, mean=None, std=None, no_rel=False):
    final_list = []
    final_rels = []
    final_doc_ids = []
    query_ids = []
    for array in df.groupby('query_id').apply(lambda x: x.as_matrix()):
        final_array = []
        rels = []
        docs = []
        i = 0
        # print(array.shape)
        for arr in array:
            if no_rel:
                final_array.append(arr[2:])
            else:
                final_array.append(arr[2:-1])
            rels.append(arr[-1])
            docs.append(arr[0])
            if i == 0:
                query_ids.append(arr[1])
            i += 1
        final_rels.append(np.array(rels, dtype=np.float32))
        final_list.append(np.array(final_array, dtype=np.float32))
        final_doc_ids.append(docs)
    # print(final_list[0], type(final_list[0]))
    # final_list[0]
    tensor_list = [(torch.from_numpy(x[:, features])).cuda() for x in final_list]

    if mean is None:
        mean = torch.zeros(tensor_list[0].mean(0).size()).cuda()
        std = torch.zeros(mean.size()).cuda()
        tot = 0
        for i in range(len(tensor_list)):
            tot += tensor_list[i].size(0)
            mean += tensor_list[i].sum(0)
        mean = mean / tot
        # tot = 0
        for i in range(len(tensor_list)):
            # tot += tensor_list[i].size(0)
            std += ((tensor_list[i] - mean)*(tensor_list[i] - mean)).sum(0)
        std = torch.sqrt(std / tot)
    tensor_list = [(x - mean) / std for x in tensor_list]

    # print(tensor_list[0])
    rels = [torch.from_numpy(x).cuda() for x in final_rels]
    # print(rels[0])
    # indexes = [(torch.nonzero(x), torch.nonzero(1 - x)) for x in rels]

    return tensor_list, rels, mean, std, final_doc_ids, query_ids



def save_test_ranking(e, scores, doc_ids, query_ids, test='test'):

    data = defaultdict(list)
    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
    count = 0
    for i, query_id in enumerate(query_ids):

        if count % 15 == 0:
            print('Finished {}%...'.format(count / 15 * 10))
        count += 1
        doc_result = []
        for j, ext_doc_id in enumerate(doc_ids[i]):
            #             print(int_doc_id)
            #             ext_doc_id, _ = index.document(int_doc_id)

            doc_score = scores[i][j]
            #             if doc_score != 0:
            #             print(query_id)
            data[query_id].append((doc_score, ext_doc_id))

        with open('results/RankNet.run', 'w') as f_out:
            write_run(
                model_name='RankNet',
                data=data,
                out_f=f_out,
                max_objects_per_query=1000)

def train(model, tensor_list, rels):
    model.train()
    loss = Variable(torch.zeros(1).cuda())
    start = time.time()

    kk = 0
    optimizer.zero_grad()
    for q in range(len(tensor_list)):

        r = np.random.randint(0,len(tensor_list))
        # print(tensor_list[r].size(), rels[r].size())
        if torch.sum(rels[q]) == 0:
            continue
            # raise ' we'
        scores = model(tensor_list[q]) #/ len(tensor_list)


        loss = RankNetLoss(scores, rels[q], gamma=1) #/ len(tensor_list)
        ranking, indices = torch.sort(scores[:, 0], descending=True)
        if kk == 0:
            # print(indices)
            kk += 1
    loss.backward()
    print('loss = ', loss.data[0])
    # print('prima', torch.sum(model.fc1.weight), torch.sum(model.fc2.bias))
    optimizer.step()
    # print('dopo ', torch.sum(model.fc1.weight), torch.sum(model.fc2.bias))
    return loss.data[0]
    # print('t     ', time.time() - start, ' --> ', loss.data[0])

measures = ['ndcg_cut_10', 'map_cut_1000', 'recall_1000', 'P_5']


def test(model, tensor_list, doc_ids, query_ids, validation):
    model.eval()
    loss = Variable(torch.zeros(1).cuda())
    conf = torch.zeros(2, 2)
    scores_list = []
    kk = 0
    for q in range(len(tensor_list)):
        optimizer.zero_grad()
        scores = model(tensor_list[q])
        # scores_list.append(scores)
        # loss += RankNetLoss(scores, rels[q], gamma=1) / len(tensor_list)
        ranking, indices = torch.sort(scores[:, 0], descending=True)
        if kk == 0:
            # print(indices)
            kk += 1
        # print(torch.max(ranking).data[0], torch.min(ranking).data[0])

        # predicted_rels = torch.round(ranking).clamp(min=0, max=1)
        # predicted_ranking = rels[q][indices.data]
        # print(ranking[:5])
        scores_list.append(scores.data[:, 0])
        # conf += confusion_matrix(predicted_rels.data, predicted_ranking)
    # print('loss test = ', loss.data[0])
    save_test_ranking(e, scores_list, doc_ids, query_ids)
    results = trec_eval('RankNet', measures, validation=validation)
    # conf /= torch.sum(conf)
    # results, precision, recall = conf[0, 0] + conf[1, 1], conf[1, 1] + conf[1, 0], conf[1, 1] + conf[0, 1]
    # print('t   test  ', time.time() - start, ' --> ', loss.data[0])
    # return 0,0,0,0
    return {measure: results[measure]['all'] for measure in measures}



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
# features = [i*10 for i in range(10)]
features = range(102)
tensor_list, rels, mean, std, train_doc_ids, train_query_ids = get_data(train_data, features)
test_list, test_rels, _, _, test_doc_ids, test_query_ids = get_data(test_data, features, mean=mean, std=std)
top1000_list, _, _, _, top1000_doc_ids, top1000_query_ids = get_data(top1000_data, features, mean=mean, std=std, no_rel=True)


test_1000_list = [top1000_list[top1000_query_ids.index(idx)] for _, idx in enumerate(test_query_ids)]
test_1000_doc_ids = [top1000_doc_ids[top1000_query_ids.index(idx)] for _, idx in enumerate(test_query_ids)]
test_1000_query_ids = [top1000_query_ids[top1000_query_ids.index(idx)] for _, idx in enumerate(test_query_ids)]
# recall_k = lambda x: torch.sum(x) / x.size(0)
# print([recall_k(x) for x in test_rels])



# print(tensor_list[0].size())
CUDA = True
model = RankNet(len(features))
if CUDA:
    model = model.cuda()

# means = [torch.mean(x, 0) for x in tensor_list]
# tensor_list = [tensor_list[i] / means[i] for i in range(len(tensor_list))]
model.train()

optimizer = optim.SGD(params=model.parameters(), lr=0.1, momentum=0.5, weight_decay=0.001)


k = 10
listlen = len(tensor_list)
validlen = listlen - int(listlen / k)
best_results = None
best_model = None
best_ndcg = -10000
best_losses = None

epochs = 100
import itertools
split = lambda x, p: [x[i] for i in p]
splits = KFold(n_splits=10, shuffle=True).split(tensor_list)
for train_split, test_split in splits:
    # permutation = np.random.permutation(listlen).tolist()
    # print(permutation)
    train_list = split(tensor_list, train_split)
    train_rels = split(rels, train_split)
    valid_list = split(tensor_list, test_split)
    valid_rels = split(rels, test_split)
    losses = []
    for e in range(epochs):
        loss = train(model, train_list, train_rels)
        losses.append(loss)
        # results = test(model, test_list, test_rels, test_doc_ids, test_query_ids)
        # print('train -> ', results)
        # if e % 25 == 0:
        #     results = test(model, e, valid_list, None, train_doc_ids[permutation][:valid_list],
        #                     train_query_ids[permutation][:valid_list])
        #     print('test -> ', results)
    results = test(model, valid_list, split(train_doc_ids, test_split), split(train_query_ids, test_split), validation=False)
    if results['ndcg_cut_10'] > best_ndcg:
        best_ndcg = results['ndcg_cut_10']
        best_results = results
        best_model = model
        best_losses = losses
    print(results)

# test_1000_list, test_1000_doc_ids, test_1000_query_ids = [], [], []


results = test(model,  test_1000_list, test_1000_doc_ids, test_1000_query_ids, validation=True)
# results = test(model,  top1000_list, top1000_doc_ids, top1000_query_ids)
print('test -> ', results)
