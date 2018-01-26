import io
import time

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.cuda as cuda
from gensim.models import KeyedVectors
from torch.autograd import Variable
import torch.nn.functional as F
import collections
import pyndri

CUDA = cuda.is_available()

def hard_tanh(x):
    return min(max(-1, x), 1)

if CUDA:
	torch.FloatTensor = torch.cuda.FloatTensor
	torch.Tensor = torch.cuda.Tensor


class NVSM(nn.Module):
	def __init__(self, num_documents, documents, vocabulary, embeddings, document_emb_size, word_emb_size, batch_size, n_gram, z, lamb):
		super(NVSM, self).__init__()
		self.documents = documents
		self.num_documents = num_documents
		self.embeddings = {}
		self.rd = Variable(init.xavier_normal(torch.Tensor(document_emb_size,num_documents)), requires_grad=True)
		# self.rv = torch.FloatTensor([embeddings[word] for word in vocabulary])
		self.proj = Variable(init.xavier_normal(torch.Tensor(document_emb_size,word_emb_size)), requires_grad = True)
		self.document_emb_size = document_emb_size
		self.word_emb_size = word_emb_size
		self.batch_size = batch_size
		self.n_gram = n_gram
		self.z = z
		self.lamb = lamb
		self.beta = Variable(torch.rand((word_emb_size, 1)), requires_grad = True)
		self.vocabulary = vocabulary
		for word in vocabulary.token2id:
			# REMOVE WITH REAL MODEL
			if word in embeddings.vocab:
				self.embeddings[word] = Variable(torch.FloatTensor(embeddings.word_vec(word)), requires_grad = True)

	def f(self, x):
		return self.proj * x

	def norm(self, x):
		return x/torch.norm(x)

	def g(self, x):
		return x.mean(1)

	def t(self, x):
		return hard_tanh((words_sample - words_sample.mean(1))/torch.sqrt(words_sample.std(1)) + self.beta)

	def p(self, x, doc):
		return F.sigmoid(x*doc)

	def sample_batch(self):
		word_tensor = torch.zeros((self.batch_size * self.word_emb_size, self.n_gram))
		document_tensor = torch.zeros(self.batch_size * self.document_emb_size)
		for i in range(self.batch_size):
			sample_doc_idx =  int(torch.rand(1)[0] * self.num_documents)
			sample_doc = documents[sample_doc_idx]
			sample_doc_embedding = self.rd[:, sample_doc_idx]
			sample_word_idx = int(torch.rand(1)[0] * (len(sample_doc) - 5))
			# REMOVE WITH REAL MODEL
			sample_word_emb = torch.stack([self.embeddings[self.vocabulary.id2token[word]]
												 for word in sample_doc[sample_word_idx:sample_word_idx+self.n_gram]
												 if self.vocabulary.id2token[word] in self.embeddings])
			word_tensor[i*self.word_emb_size:(i+1)*self.word_emb_size, self.n_gram] = sample_word_emb
			document_tensor[i*self.document_emb_size:(i+1)*self.document_emb_size] = sample_doc_embedding
		return Variable(word_tensor, requires_grad = True), Variable(document_tensor, requires_grad=True)

	def forward(self):
		out = Variable(torch.zeros(self.batch_size), requires_grad=True)
		words, documents = self.sample_batch()
		words_processed = self.g(words)
		for i in range(self.batch_size):
			processed_words = self.t(self.f(self.norm(words_processed[i*self.word_emb_size:(i+1)*self.word_emb_size])))
			log_p = z * torch.log(self.p(processed_words, documents[i*self.document_emb_size:(i+1)*self.document_emb_size]))
				# torch.sum([torch.log(1.0 - )]))
			log_p_sample = torch.sum(torch.stack([torch.log(1.0 - self.p(processed_words, 
				self.rd[:, torch.rand((word_emb_size, 1)) * self.num_documents] for i in range(z)))]))
			log_p_wave = (z+1)/(2*z) * (log_p + log_p_sample)
			out[i] = log_p_wave
		# add RV
		loss = 1/self.batch_size * torch.sum(out) + self.lamb/(2*m) * (torch.sum(self.rd) + torch.sum(self.proj))

def parse_topics(file_or_files,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and not isinstance(file_or_files, tuple):
        if hasattr(file_or_files, '__iter__'):
            file_or_files = list(file_or_files)
        else:
            file_or_files = [file_or_files]

    for f in file_or_files:
        assert isinstance(f, io.IOBase)

        for line in f:
            assert(isinstance(line, str))

            line = line.strip()

            if not line:
                continue

            topic_id, terms = line.split(delimiter, 1)

            if topic_id in topics and (topics[topic_id] != terms):
                    logging.error('Duplicate topic "%s" (%s vs. %s).',
                                  topic_id,
                                  topics[topic_id],
                                  terms)

            topics[topic_id] = terms

            if max_topics > 0 and len(topics) >= max_topics:
                break

    return topics

with open('./ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

index = pyndri.Index('index/')

documents = []
for doc_idx in range(index.document_base(), index.maximum_document()):
	document_id, document = index.document(doc_idx)
	document = [word_id for word_id in document if word_id > 0]
	documents.append(document)

start_time = time.time()
num_documents = index.maximum_document() - index.document_base()

dictionary = pyndri.extract_dictionary(index)

word_vectors = KeyedVectors.load_word2vec_format('reduced_vectors_google.txt')  # C binary format

model = NVSM(num_documents, documents, dictionary, word_vectors, 300, 300, 256, 5)
model.train()
model.forward()