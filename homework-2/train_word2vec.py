import copy
import gensim
import logging
import pyndri
import pyndri.compat
import sys
import time
import pickle

# def train_word2vec(iterations):
#     word2vec_init = gensim.models.Word2Vec(
#         size=300,  # Embedding size
#         window=5,  # One-sided window size
#         sg=True,  # Skip-gram.
#         min_count=5,  # Minimum word frequency.
#         sample=1e-3,  # Sub-sample threshold.
#         hs=False,  # Hierarchical softmax.
#         negative=10,  # Number of negative examples.
#         iter=1,  # Number of iterations.
#         workers=8,  # Number of workers.
#     )
#
#     with pyndri.open('index/') as index:
#         dictionary = pyndri.extract_dictionary(index)
#         sentences = pyndri.compat.IndriSentences(index, dictionary)
#
#         # Build vocab
#         word2vec_init.build_vocab(sentences, trim_rule=None)
#         models = [word2vec_init]
#
#         for epoch in range(iterations):
#             start_time = time.time()
#             print('Epoch {} started..'.format(epoch+1))
#
#             model = copy.deepcopy(models[-1])
#             model.train(sentences, total_examples=len(sentences), epochs=model.iter)
#
#             models.append(model)
#             print('Epoch {} finished in {}'.format(epoch+1, time.time()-start_time))
#     return models[-1]
#
# model = train_word2vec(15)
# pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print('we')
