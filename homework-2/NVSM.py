import io
import time

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.cuda as cuda
from torch import optim
from gensim.models import KeyedVectors
from torch.autograd import Variable
import torch.nn.functional as F
import collections
import pyndri
import pickle
import os
import copy
import gensim
import logging
import pyndri
import pyndri.compat
import sys

CUDA = cuda.is_available()
epochs = 1
n_gram = 5
num_batches = 100

if CUDA:
    torch.FloatTensor = torch.cuda.FloatTensor
    torch.LongTensor = torch.cuda.LongTensor
    torch.ByteTensor = torch.cuda.ByteTensor

if not os.path.exists('saved'):
    os.makedirs('saved')


class NVSM(nn.Module):
    def __init__(self, documents, t2i, i2t, embeddings, document_emb_size, word_emb_size, batch_size,
                 n_gram, z, lamb, m):
        super(NVSM, self).__init__()
        self.documents = documents
        self.num_documents = len(documents)
        self.rd = Variable(init.xavier_normal(torch.Tensor(document_emb_size, self.num_documents)), requires_grad=True)
        self.rv = Variable(torch.stack([torch.FloatTensor(embeddings[word]) for idx, word in i2t.items()], 1),
                           requires_grad=True)
        self.proj = Variable(init.xavier_normal(torch.Tensor(document_emb_size, word_emb_size)), requires_grad=True)
        self.document_emb_size = document_emb_size
        self.word_emb_size = word_emb_size
        self.batch_size = batch_size
        self.n_gram = n_gram
        self.z = z
        self.lamb = lamb
        self.beta = Variable(torch.rand((document_emb_size, 1)), requires_grad=True)
        self.t2i = t2i
        self.i2t = i2t
        self.m = m
        if CUDA:
            self.rd = self.rd.cuda()
            self.rv = self.rv.cuda()
            self.proj = self.proj.cuda()
            self.beta = self.beta.cuda()

    # for word in vocabulary.token2id:
    #   # REMOVE WITH REAL MODEL
    #   if word in embeddings.vocab:
    #     self.embeddings[word] = Variable(torch.FloatTensor(embeddings.word_vec(word)), requires_grad = True)

    def f(self, x):
        return self.proj @ x

    def norm(self, x):
        return x / torch.norm(x)

    def g(self, x):
        return x.mean(1)

    def t(self, x, mean, std):
        return F.hardtanh((x - mean) / torch.sqrt(std) + self.beta.view(-1))

    def p(self, x, doc):
        return F.sigmoid(x @ doc)

    def sample_batch(self):
        for i in range(self.batch_size):
            sample_doc_idx = int(torch.rand(1)[0] * self.num_documents)
            sample_doc_idx_var = Variable(torch.LongTensor([sample_doc_idx]), requires_grad=False)
            if CUDA:
                sample_doc_idx_var = sample_doc_idx_var.cuda()
            sample_doc = documents[sample_doc_idx]
            sample_doc_emb = self.rd.index_select(1, sample_doc_idx_var)
            sample_word_idx = int(torch.rand(1)[0] * (len(sample_doc) - 6))
            sample_word_ids = Variable(torch.LongTensor(sample_doc[sample_word_idx:sample_word_idx + self.n_gram]),
                                       requires_grad=False)
            if CUDA:
                sample_word_ids = sample_word_ids.cuda()
            sample_word_emb = self.rv.index_select(1, sample_word_ids)
            if i == 0:
                word_tensor = sample_word_emb
                document_tensor = sample_doc_emb
            else:
                word_tensor = torch.cat([word_tensor, sample_word_emb])
                document_tensor = torch.cat([document_tensor, sample_doc_emb])
                # # word_tensor[i * self.word_emb_size:(i + 1) * self.word_emb_size, :self.n_gram] = sample_word_emb
                # document_tensor[i * self.document_emb_size:(i + 1) * self.document_emb_size] = sample_doc_embedding
        return word_tensor, document_tensor

    def forward(self):
        out = Variable(torch.zeros(self.batch_size), requires_grad=True)
        if CUDA:
            out = out.cuda()
        words, documents = self.sample_batch()
        words_processed = self.g(words)
        for i in range(self.batch_size):
            if i == 0:
                t_tensor = self.f(self.norm(words_processed[i * self.word_emb_size:(i + 1) * self.word_emb_size]))
            elif i == 1:
                res = self.f(self.norm(words_processed[i * self.word_emb_size:(i + 1) * self.word_emb_size]))
                print(t_tensor.shape)
                print(res.shape)
                t_tensor = torch.stack([t_tensor, res], 1)
            else:
                res = self.f(self.norm(words_processed[i * self.word_emb_size:(i + 1) * self.word_emb_size]))
                print(t_tensor.shape)
                print(res.shape)
                t_tensor = torch.cat([t_tensor, res.view(-1, 1)], 1)
        mean = t_tensor.mean(1)
        std = t_tensor.std(1)
        for i in range(self.batch_size):
            t = self.t(t_tensor[:, i], mean, std)
            log_p = self.z * torch.log(
                self.p(t, documents[i * self.document_emb_size:(i + 1) * self.document_emb_size]))
            # torch.sum([torch.log(1.0 - )]))
            nsample_indices = Variable((torch.rand(self.z) * self.num_documents).long(), requires_grad=False)
            if CUDA:
                nsample_indices = nsample_indices.cuda()
            nsample_documents = self.rd.index_select(1, nsample_indices)
            nsample_log = torch.sum(torch.log(1 - self.p(t.view(1, -1), nsample_documents)))
            log_p_wave = (self.z + 1) / (2 * self.z) * (log_p + nsample_log)
            if i == 0:
                out = log_p_wave
            else:
                out = torch.cat([out, log_p_wave])
        # add RV
        loss = 1 / self.batch_size * torch.sum(out) + self.lamb / (2 * self.m) * (
            torch.sum(self.rd) + torch.sum(self.rv) + torch.sum(self.proj))
        return loss

    def score(self, document_id, query):
        if CUDA:
            query_token_ids = Variable(torch.LongTensor([t2i(word) for word in query]).cuda(), requires_grad=False)
        else:
            query_token_ids = Variable(torch.LongTensor([t2i(word) for word in query]), requires_grad=False)

        words = self.rv.index_select(1, query_token_ids)
        words_processed = self.g(words)
        words_proj = self.f(words_processed)
        document_proj = self.rd[:, document_id]
        return F.cosine_similarity(words_proj, document_proj)


def train_word2vec(iterations):
    word2vec_init = gensim.models.Word2Vec(
        size=300,  # Embedding size
        window=5,  # One-sided window size
        sg=True,  # Skip-gram.
        min_count=5,  # Minimum word frequency.
        sample=1e-3,  # Sub-sample threshold.
        hs=False,  # Hierarchical softmax.
        negative=10,  # Number of negative examples.
        iter=1,  # Number of iterations.
        workers=8,  # Number of workers.
    )

    with pyndri.open('index/') as index:
        dictionary = pyndri.extract_dictionary(index)
        sentences = pyndri.compat.IndriSentences(index, dictionary)

        # Build vocab
        word2vec_init.build_vocab(sentences, trim_rule=None)
        models = [word2vec_init]

        for epoch in range(iterations):
            start_time = time.time()
            print('Epoch {} started..'.format(epoch + 1))

            model = copy.deepcopy(models[-1])
            model.train(sentences, total_examples=len(sentences), epochs=model.iter)

            models.append(model)
            print('Epoch {} finished in {}'.format(epoch + 1, time.time() - start_time))
    return models[-1]


if os.path.exists('saved/model.pkl'):
    model = pickle.load(open('saved/model.pkl', 'rb'))
else:
    model = train_word2vec(15)
    pickle.dump(model, open('saved/model.pkl', 'wb'))

index = pyndri.Index('index/')

if os.path.exists('saved/t2i.pkl') and os.path.exists('saved/i2t.pkl') \
        and os.path.exists('saved/documents.pkl') and os.path.exists('saved/word_vectors.pkl'):
    documents = pickle.load(open('saved/documents.pkl', 'rb'))
    t2i = pickle.load(open('saved/t2i.pkl', 'rb'))
    i2t = pickle.load(open('saved/i2t.pkl', 'rb'))
    word_vectors = pickle.load(open('saved/word_vectors.pkl', 'rb'))
else:
    dictionary = pyndri.extract_dictionary(index)
    word_vectors = KeyedVectors.load_word2vec_format('saved/reduced_vectors_google.txt')
    documents = []
    t2i = collections.defaultdict(lambda: len(t2i))
    i2t = {}

    for doc_idx in range(index.document_base(), index.document_base() + 10000):
        document_id, document = index.document(doc_idx)
        proc_doc = []
        for word_id in document:
            if word_id > 0 and dictionary.id2token[word_id] in word_vectors.vocab:
                i2t[t2i[dictionary.id2token[word_id]]] = dictionary.id2token[word_id]
                proc_doc.append(t2i[dictionary.id2token[word_id]])
        if len(proc_doc) > n_gram:
            documents.append(proc_doc)

            # C binary format
    for word in t2i:
        if word not in word_vectors.vocab:
            print(word)

    t2i = dict(t2i)
    i2t = dict(i2t)

    pickle.dump(documents, open('saved/documents.pkl', 'wb'))
    documents = documents[:1000]
    pickle.dump(t2i, open('saved/t2i.pkl', 'wb'))
    pickle.dump(i2t, open('saved/i2t.pkl', 'wb'))
    pickle.dump(word_vectors, open('saved/word_vectors.pkl', 'wb'))

# print(sum([len(document) for document in documents]))

# 228795
#
num_documents = index.maximum_document() - index.document_base()

# with torch.autograd.profiler.profile() as prof:
model = NVSM(documents, t2i, i2t, word_vectors, 256, 300, 1000, n_gram, 10, 0.01, 10)
if CUDA:
    model = model.cuda()
model.train()

optimizer = optim.Adam(params=[model.proj, model.rd, model.beta])

total_loss = 0.0
losses = []
for i in range(epochs):
    loss = 0.0
    start_time = time.time()
    for batch in range(num_batches):
        batch_start_time = time.time()
        optimizer.zero_grad()
        loss = model()
        total_loss += loss.data[0]
        loss += loss.data[0]
        loss.backward()
        optimizer.step()
        print('Time for 1 batch is: {}'.format(time.time() - batch_start_time))
        # print(prof)
    print('Loss is: {}'.format(loss.data[0]))
    losses.append(loss.data[0])
    # print('Average loss is: {}'.format(total_loss/(i+1)))
    print('Time for epoch, ', i + 1, ' is: {}'.format(time.time() - start_time))
    torch.save(model, 'model.pth.tar')

print(losses)
