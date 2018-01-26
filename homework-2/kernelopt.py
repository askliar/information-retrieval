
# coding: utf-8

# In[2]:


# IMPORTANT OLD VERSION!! DON'T DELETE!
#     # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
#     for query_id, query_terms in tokenized_queries.items():
        
#         if (int(query_id) - 50) % 15 == 0:
#             print('Finished {}%...'.format((int(query_id) - 50) / 15 * 10))
            
#         document_score = defaultdict(float)
        
#         for query_term_id in query_terms:
#             for int_doc_id in inverted_index[query_term_id]:
#                 ext_doc_id = index.document(int_doc_id)
#                 document_term_freq = inverted_index[query_term_id][int_doc_id]
#                 document_score[ext_doc_id] += score_fn(int_doc_id, query_term_id, document_term_freq)
                
#         query_result = [(doc_score, ext_doc_id) for ext_doc_id, doc_score in document_score.items()]
#         data[query_id] = query_result

# IMPORTANT OLD VERSION!! DON'T DELETE!
# import numpy as np

# def tfidf(int_document_id, query_term_id, document_term_freq):
#     """
#     Scoring function for a document and a query term
    
#     :param int_document_id: the document id
#     :param query_token_id: the query term id (assuming you have split the query to tokens)
#     :param document_term_freq: the document term frequency of the query term 
#     """
#     tf = document_term_freq/document_lengths[int_document_id]
#     df = len(inverted_index[query_term_id])
#     idf = num_documents/df
#     score = np.log(1 + tf) * np.log(idf)
#     return score

# # combining the two functions above: 
# run_retrieval('tfidf', tfidf)

# # TODO implement the rest of the retrieval functions 

# # TODO implement tools to help you with the analysis of the results.


# # Searching Unstructured and Structured Data #
# ## Assignment 1: Retrieval models [100 points] ##
# **TA**: Nikos Voskarides (n.voskarides@uva.nl)

# In this assignment you will get familiar with basic information retrieval concepts. You will implement and evaluate different information retrieval ranking models and evaluate their performance.
# 
# We provide you with a Indri index. To query the index, you'll use a Python package ([pyndri](https://github.com/cvangysel/pyndri)) that allows easy access to the underlying document statistics.
# 
# For evaluation you'll use the [TREC Eval](https://github.com/usnistgov/trec_eval) utility, provided by the National Institute of Standards and Technology of the United States. TREC Eval is the de facto standard way to compute Information Retrieval measures and is frequently referenced in scientific papers.
# 
# This is a **groups-of-three assignment**, the deadline is **Monday, 22/1, at 23:59**. Code quality, informative comments and convincing analysis of the results will be considered when grading. Submission should be done through blackboard, questions can be asked on the course [Piazza](https://piazza.com/class/ixoz63p156g1ts).
# 
# ### Technicalities (must-read!) ###
# 
# The assignment directory is organized as follows:
#    * `./assignment.ipynb` (this file): the description of the assignment.
#    * `./index/`: the index we prepared for you.
#    * `./ap_88_90/`: directory with ground-truth and evaluation sets:
#       * `qrel_test`: test query relevance collection (**test set**).
#       * `qrel_validation`: validation query relevance collection (**validation set**).
#       * `topics_title`: semicolon-separated file with query identifiers and terms.
# 
# You will need the following software packages (tested with Python 3.5 inside [Anaconda](https://conda.io/docs/user-guide/install/index.html)):
#    * Python 3.5 and Jupyter
#    * Indri + Pyndri (Follow the installation instructions [here](https://github.com/nickvosk/pyndri/blob/master/README.md))
#    * gensim [link](https://radimrehurek.com/gensim/install.html)
#    * TREC Eval [link](https://github.com/usnistgov/trec_eval)

# ### TREC Eval primer ###
# The TREC Eval utility can be downloaded and compiled as follows:
# 
#     git clone https://github.com/usnistgov/trec_eval.git
#     cd trec_eval
#     make
# 
# TREC Eval computes evaluation scores given two files: ground-truth information regarding relevant documents, named *query relevance* or *qrel*, and a ranking of documents for a set of queries, referred to as a *run*. The *qrel* will be supplied by us and should not be changed. For every retrieval model (or combinations thereof) you will generate a run of the top-1000 documents for every query. The format of the *run* file is as follows:
# 
#     $query_identifier Q0 $document_identifier $rank_of_document_for_query $query_document_similarity $run_identifier
#     
# where
#    * `$query_identifier` is the unique identifier corresponding to a query (usually this follows a sequential numbering).
#    * `Q0` is a legacy field that you can ignore.
#    * `$document_identifier` corresponds to the unique identifier of a document (e.g., APXXXXXXX where AP denotes the collection and the Xs correspond to a unique numerical identifier).
#    * `$rank_of_document_for_query` denotes the rank of the document for the particular query. This field is ignored by TREC Eval and is only maintained for legacy support. The ranks are computed by TREC Eval itself using the `$query_document_similarity` field (see next). However, it remains good practice to correctly compute this field.
#    * `$query_document_similarity` is a score indicating the similarity between query and document where a higher score denotes greater similarity.
#    * `$run_identifier` is an identifier of the run. This field is for your own convenience and has no purpose beyond bookkeeping.
#    
# For example, say we have two queries: `Q1` and `Q2` and we rank three documents (`DOC1`, `DOC2`, `DOC3`). For query `Q1`, we find the following similarity scores `score(Q1, DOC1) = 1.0`, `score(Q1, DOC2) = 0.5`, `score(Q1, DOC3) = 0.75`; and for `Q2`: `score(Q2, DOC1) = -0.1`, `score(Q2, DOC2) = 1.25`, `score(Q1, DOC3) = 0.0`. We can generate run using the following snippet:

# In[3]:


import logging
import sys
import os
import numpy as np
def write_run(model_name, data, out_f,
              max_objects_per_query=sys.maxsize,
              skip_sorting=False):
    """
    Write a run to an output file.
    Parameters:
        - model_name: identifier of run.
        - data: dictionary mapping topic_id to object_assesments;
            object_assesments is an iterable (list or tuple) of
            (relevance, object_id) pairs.
            The object_assesments iterable is sorted by decreasing order.
        - out_f: output file stream.
        - max_objects_per_query: cut-off for number of objects per query.
    """
    for subject_id, object_assesments in data.items():
        if not object_assesments:
            logging.warning('Received empty ranking for %s; ignoring.',
                            subject_id)

            continue

        # Probe types, to make sure everything goes alright.
        # assert isinstance(object_assesments[0][0], float) or \
        #     isinstance(object_assesments[0][0], np.float32)
        assert isinstance(object_assesments[0][1], str) or             isinstance(object_assesments[0][1], bytes)

        if not skip_sorting:
            object_assesments = sorted(object_assesments, reverse=True)

        if max_objects_per_query < sys.maxsize:
            object_assesments = object_assesments[:max_objects_per_query]

        if isinstance(subject_id, bytes):
            subject_id = subject_id.decode('utf8')

        for rank, (relevance, object_id) in enumerate(object_assesments):
            if isinstance(object_id, bytes):
                object_id = object_id.decode('utf8')

            out_f.write(
                '{subject} Q0 {object} {rank} {relevance} '
                '{model_name}\n'.format(
                    subject=subject_id,
                    object=object_id,
                    rank=rank + 1,
                    relevance=relevance,
                    model_name=model_name))
            
# The following writes the run to standard output.
# In your code, you should write the runs to local
# storage in order to pass them to trec_eval.
write_run(
    model_name='example',
    data={
        'Q1': ((1.0, 'DOC1'), (0.5, 'DOC2'), (0.75, 'DOC3')),
        'Q2': ((-0.1, 'DOC1'), (1.25, 'DOC2'), (0.0, 'DOC3')),
    },
    out_f=sys.stdout,
    max_objects_per_query=1000)


# Now, imagine that we know that `DOC1` is relevant and `DOC3` is non-relevant for `Q1`. In addition, for `Q2` we only know of the relevance of `DOC3`. The query relevance file looks like:
# 
#     Q1 0 DOC1 1
#     Q1 0 DOC3 0
#     Q2 0 DOC3 1
#     
# We store the run and qrel in files `example.run` and `example.qrel` respectively on disk. We can now use TREC Eval to compute evaluation measures. In this example, we're only interested in Mean Average Precision and we'll only show this below for brevity. However, TREC Eval outputs much more information such as NDCG, recall, precision, etc.
# 
#     $ trec_eval -m all_trec -q example.qrel example.run | grep -E "^map\s"
#     > map                   	Q1	1.0000
#     > map                   	Q2	0.5000
#     > map                   	all	0.7500
#     
# Now that we've discussed the output format of rankings and how you can compute evaluation measures from these rankings, we'll now proceed with an overview of the indexing framework you'll use.

# In[4]:


# %%bash
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run > trec_results_tdfidf
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run > trec_results_tdfidf
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run > trec_results_tdfidf
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run > trec_results_tdfidf

# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run  | grep -E "^map_cut_1000" > map_cut_1000
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run  | grep -E "^ndcg_cut_10\s" > ndcg_cut_10
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run  | grep -E "^P_5\s" > P_5
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run  | grep -E "^recall_1000\s" > recall_1000
# ./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel tfidf.run 


# In[5]:


# %matplotlib
# import matplotlib.pyplot as plt

# y = np.array(list(document_lengths.values()))
# x = np.arange(len(y))
# print(sorted(y))
# plt.plot(y, x)
# plt.show()


# In[6]:


# print([len(results[a]) for a in results])
# print([(i, results[i]) for i in measures])


# ### Pyndri primer ###
# For this assignment you will use [Pyndri](https://github.com/cvangysel/pyndri) [[1](https://arxiv.org/abs/1701.00749)], a python interface for [Indri](https://www.lemurproject.org/indri.php). We have indexed the document collection and you can query the index using Pyndri. We will start by giving you some examples of what Pyndri can do:
# 
# First we read the document collection index with Pyndri:

# In[7]:


import pyndri

index = pyndri.Index('index/')


# The loaded index can be used to access a collection of documents in an easy manner. We'll give you some examples to get some idea of what it can do, it is up to you to figure out how to use it for the remainder of the assignment.
# 
# First let's look at the number of documents, since Pyndri indexes the documents using incremental identifiers we can simply take the lowest index and the maximum document and consider the difference:

# In[8]:


print("There are %d documents in this collection." % (index.maximum_document() - index.document_base()))


# Let's take the first document out of the collection and take a look at it:

# In[9]:


example_document = index.document(index.document_base())
print(example_document)


# Here we see a document consists of two things, a string representing the external document identifier and an integer list representing the identifiers of words that make up the document. Pyndri uses integer representations for words or terms, thus a token_id is an integer that represents a word whereas the token is the actual text of the word/term. Every id has a unique token and vice versa with the exception of stop words: words so common that there are uninformative, all of these receive the zero id.
# 
# To see what some ids and their matching tokens we take a look at the dictionary of the index:

# In[10]:


token2id, id2token, _ = index.get_dictionary()
print(list(id2token.items())[:15])


# Using this dictionary we can see the tokens for the (non-stop) words in our example document:

# In[11]:


print([id2token[word_id] for word_id in example_document[1] if word_id > 0])


# The reverse can also be done, say we want to look for news about the "University of Massachusetts", the tokens of that query can be converted to ids using the reverse dictionary:

# In[12]:


query_tokens = index.tokenize("University of Massachusetts")
print("Query by tokens:", query_tokens)
query_id_tokens = [token2id.get(query_token,0) for query_token in query_tokens]
print("Query by ids with stopwords:", query_id_tokens)
query_id_tokens = [word_id for word_id in query_id_tokens if word_id > 0]
print("Query by ids without stopwords:", query_id_tokens)


# Naturally we can now match the document and query in the id space, let's see how often a word from the query occurs in our example document:

# In[13]:


matching_words = sum([True for word_id in example_document[1] if word_id in query_id_tokens])
print("Document %s has %d word matches with query: \"%s\"." % (example_document[0], matching_words, ' '.join(query_tokens)))
print("Document %s and query \"%s\" have a %.01f%% overlap." % (example_document[0], ' '.join(query_tokens),matching_words/float(len(example_document[1]))*100))


# While this is certainly not everything Pyndri can do, it should give you an idea of how to use it. Please take a look at the [examples](https://github.com/cvangysel/pyndri) as it will help you a lot with this assignment.
# 
# **CAUTION**: Avoid printing out the whole index in this Notebook as it will generate a lot of output and is likely to corrupt the Notebook.

# ### Parsing the query file
# You can parse the query file (`ap_88_89/topics_title`) using the following snippet:

# In[14]:


import collections
import io
import logging
import sys

def parse_topics(file_or_files,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and             not isinstance(file_or_files, tuple):
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
    print(parse_topics([f_topics]))


# ### Task 1: Implement and compare lexical IR methods [35 points] ### 
# 
# In this task you will implement a number of lexical methods for IR using the **Pyndri** framework. Then you will evaluate these methods on the dataset we have provided using **TREC Eval**.
# 
# Use the **Pyndri** framework to get statistics of the documents (term frequency, document frequency, collection frequency; **you are not allowed to use the query functionality of Pyndri**) and implement the following scoring methods in **Python**:
# 
# - [TF-IDF](http://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) and 
# - [BM25](http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) with k1=1.2 and b=0.75. **[5 points]**
# - Language models ([survey](https://drive.google.com/file/d/0B-zklbckv9CHc0c3b245UW90NE0/view))
#     - Jelinek-Mercer (explore different values of 𝛌 in the range [0.1, 0.5, 0.9]). **[5 points]**
#     - Dirichlet Prior (explore different values of 𝛍 [500, 1000, 1500]). **[5 points]**
#     - Absolute discounting (explore different values of 𝛅 in the range [0.1, 0.5, 0.9]). **[5 points]**
#     - [Positional Language Models](http://sifaka.cs.uiuc.edu/~ylv2/pub/sigir09-plm.pdf) define a language model for each position of a document, and score a document based on the scores of its PLMs. The PLM is estimated based on propagated counts of words within a document through a proximity-based density function, which both captures proximity heuristics and achieves an effect of “soft” passage retrieval. Implement the PLM, all five kernels, but only the Best position strategy to score documents. Use 𝛔 equal to 50, and Dirichlet smoothing with 𝛍 optimized on the validation set (decide how to optimize this value yourself and motivate your decision in the report). **[10 points]**
#     
# Implement the above methods and report evaluation measures (on the test set) using the hyper parameter values you optimized on the validation set (also report the values of the hyper parameters). Use TREC Eval to obtain the results and report on `NDCG@10`, Mean Average Precision (`MAP@1000`), `Precision@5` and `Recall@1000`.
# 
# For the language models, create plots showing `NDCG@10` with varying values of the parameters. You can do this by chaining small scripts using shell scripting (preferred) or execute trec_eval using Python's `subprocess`.
# 
# Compute significance of the results using a [two-tailed paired Student t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) **[5 points]**. Be wary of false rejection of the null hypothesis caused by the [multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem). There are multiple ways to mitigate this problem and it is up to you to choose one.
# 
# Analyse the results by identifying specific queries where different methods succeed or fail and discuss possible reasons that cause these differences. This is *very important* in order to understand who the different retrieval functions behave.
# 
# **NOTE**: Don’t forget to use log computations in your calculations to avoid underflows. 

# **IMPORTANT**: You should structure your code around the helper functions we provide below.

# In[15]:


import time

with open('./ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

index = pyndri.Index('index/')
start_time = time.time()
num_documents = index.maximum_document() - index.document_base()

dictionary = pyndri.extract_dictionary(index)

tokenized_queries = {
    query_id: [dictionary.translate_token(token)
               for token in index.tokenize(query_string)
               if dictionary.has_token(token)]
    for query_id, query_string in queries.items()}
#TODO: Explain why we need it
query_term_counts = {
    query_id: collections.Counter(query_terms) 
    for query_id, query_terms in tokenized_queries.items()}

query_term_ids = set(
    query_term_id
    for query_term_ids in tokenized_queries.values()
    for query_term_id in query_term_ids)

print('Gathering statistics about', len(query_term_ids), 'terms.')

# inverted index creation.

document_lengths = {}
unique_terms_per_document = {}

inverted_index = collections.defaultdict(dict)
inverted_index_positions = collections.defaultdict(dict)
collection_frequencies = collections.defaultdict(int)

total_terms = 0

document_ids = []
for int_doc_id in range(index.document_base(), index.maximum_document()):
    # CHECK IF WE CAN DO THAT
    if len(index.document(int_doc_id)[1]) > 0:
        document_ids.append(int_doc_id)
        ext_doc_id, doc_token_ids = index.document(int_doc_id)

        document_bow = collections.Counter(
            token_id for token_id in doc_token_ids
            if token_id > 0)

        token_positions = collections.defaultdict(list)

        # Check with the TAs if we can change that
        [token_positions[token_id].append(i) for i, token_id in enumerate(doc_token_ids)]

        document_length = sum(document_bow.values())

        document_lengths[int_doc_id] = document_length
        total_terms += document_length

        unique_terms_per_document[int_doc_id] = len(document_bow)

        for query_term_id in query_term_ids:
            assert query_term_id is not None

            document_term_frequency = document_bow.get(query_term_id, 0)

            if document_term_frequency == 0:
                continue

            collection_frequencies[query_term_id] += document_term_frequency
            inverted_index[query_term_id][int_doc_id] = document_term_frequency
            inverted_index_positions[query_term_id][int_doc_id] = token_positions[query_term_id]

inverted_index = dict(inverted_index)
inverted_index_positions = dict(inverted_index_positions)

avg_doc_length = total_terms / num_documents

print('Inverted index creation took', time.time() - start_time, 'seconds.')


# In[25]:


# def run_retrieval(model_name, score_fn):
#     """
#     Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.
    
#     :param model_name: the name of the model (a string)
#     :param score_fn: the scoring function (a function - see below for an example) 
#     """
#     run_out_path = '{}.run'.format(model_name)

#     if os.path.exists(run_out_path):
#         return

#     retrieval_start_time = time.time()

#     print('Retrieving using', model_name)

#     data = {}
#     a = 0
#     # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
#     for query_id in tokenized_queries:
        
#         if (int(query_id) - 50) % 15 == 0:
#             print('Finished {}%...'.format((int(query_id) - 50) / 15 * 10))
            
#         doc_result = []
#         for int_doc_id in range(index.document_base(), index.maximum_document()):
#             if a > 1:
#                 break
#             a += 1
#             ext_doc_id = index.document(int_doc_id)
#             doc_score = score_fn(int_doc_id, query_id)
#             if len(doc_result) < 1:
#                 doc_result.append((doc_score, ext_doc_id))
#         data[query_id] = doc_result

#     with open(run_out_path, 'w') as f_out:
#         write_run(
#             model_name=model_name,
#             data=data,
#             out_f=f_out,
#             max_objects_per_query=1000)
import sys
@profile
def run_retrieval(model_name, score_fn):
    """
    Runs a retrieval method for all the queries and writes the TREC-friendly results in a file.
    
    :param model_name: the name of the model (a string)
    :param score_fn: the scoring function (a function - see below for an example) 
    """
    run_out_path = '{}.run'.format(model_name)

#     if os.path.exists(run_out_path):
#         return

    retrieval_start_time = time.time()

    print('Retrieving using', model_name)
    
    data = collections.defaultdict(list)
    # The dictionary data should have the form: query_id --> (document_score, external_doc_id)
    count = 0
    for query_id in tokenized_queries:
            
        if count % 15 == 0:
            print('Finished {}%...'.format(count / 15 * 10))
        count += 1    
        doc_result = []
        for int_doc_id in document_ids:
            ext_doc_id, _ = index.document(int_doc_id)
            doc_score = score_fn(int_doc_id, query_id)
#             if doc_score != 0:
            data[query_id].append((doc_score, ext_doc_id))
        
    with open(run_out_path, 'w') as f_out:
        write_run(
            model_name=model_name,
            data=data,
            out_f=f_out,
            max_objects_per_query=1000)


# In[16]:


import numpy as np
def tfidf(int_document_id, query_id):
    """
    Scoring function for a document and a query term
    
    :param int_document_id: the document id
    :param query_token_id: the query id
    """
    
    score = 0
    
    for query_term_id in tokenized_queries[query_id]:
        # can rewrite this to skip going over documents not containing query term
        if int_document_id in inverted_index[query_term_id]:
            tf = inverted_index[query_term_id][int_document_id]
#             print(a, type(document_term_freq), document_term_freq)
            df = len(inverted_index[query_term_id])
            idf = num_documents/df
#             idf = np.exp(np.log(num_documents) - np.log(df))
            score += np.log(1 + tf) * np.log(idf)
#             score += np.exp(np.log(np.log(1 + tf)) + np.log(idf)
            
    return score

#run_retrieval('tfidf', tfidf)

# TODO implement the rest of the retrieval functions 

# TODO implement tools to help you with the analysis of the results.


# In[18]:


def bm25(int_document_id, query_id):
    k1=1.2
    b=0.75
    # TODO: check if we actually have to use k3
    k3=1.6
    score = 0
    
    # decide whether to go over set or not
    for query_term_id in set(tokenized_queries[query_id]):
        if int_document_id in inverted_index[query_term_id]:
            tf_td = inverted_index[query_term_id][int_document_id]
            # IMPORTANT OLD VERSION! DO NOT REMOVE!
#             document_term_freq = inverted_index[query_term_id][int_document_id]
#             tf_td = document_term_freq/Ld
            
            Ld = document_lengths[int_document_id]
            Lavg = avg_doc_length

            tf_td = inverted_index[query_term_id][int_document_id]

            df = len(inverted_index[query_term_id])
            idf = num_documents/df
#             idf = np.exp(np.log(num_documents) - np.log(df))
            tf_tq = query_term_counts[query_id][query_term_id]

            first_term = np.log(idf)
            second_term = ((k1+1)*tf_td)/(k1*((1-b)+b*(Ld/avg_doc_length))+tf_td)
#             second_term = np.exp( (np.log(k1+1)+np.log(tf_td)) 
#                 - np.exp( (np.log(k1) + np.log((1-b)+np.log(b)+np.log(Ld) - np.log(avg_doc_length)))+tf_td)
            # decide whether to use third term
        #             third_term = (k3+1)*tf_tq/(k3+tf_tq)

            score += first_term * second_term # * third_term
#             score += np.exp(np.log(first_term) + np.log(second_term)) # * third_term

    
    
    return score

#run_retrieval('bm25', bm25)


# In[48]:


import functools

def jelinek_mercer(int_document_id, query_id, lambd):
    
    score = 0
    for query_term_id in tokenized_queries[query_id]:
#         if int_document_id in inverted_index[query_term_id]:
        # Don't need to divide by len, because already doing that in interpolation (check NLP-1)
        # basically, just a model with unigram probability
        tf = inverted_index[query_term_id].get(int_document_id, 0)
        doc_len = document_lengths[int_document_id] 

#         first_term = lambd * tf/doc_len
        first_term = lambd*tf/doc_len
        second_term = (1 - lambd) * collection_frequencies[query_term_id]/total_terms
#         second_term = np.exp(np.log(1 - lambd) + np.log(collection_frequencies[query_term_id]) - np.log(total_terms))
        score += np.log(first_term + second_term) * query_term_counts[query_id][query_term_id]
#         score += np.exp(np.log(first_term) + np.log(second_term))
    
    return score

#for lambd_val in [0.1, 0.5, 0.9]:
#    jel_merc_func = functools.partial(jelinek_mercer, lambd = lambd_val)
#    run_retrieval('jelinek_mercer_{}'.format(lambd_val), jel_merc_func)


# In[50]:


def dirichlet_prior(int_document_id, query_id, mu):
    score = 0
    for query_term_id in tokenized_queries[query_id]:
#         if int_document_id in inverted_index[query_term_id]:
            # Don't need to divide by len, because already doing that in interpolation (check NLP-1)
            # basically, just a model with unigram probability
        tf = inverted_index[query_term_id].get(int_document_id, 0)
                    
        prob = collection_frequencies[query_term_id]/total_terms
#         prob = np.exp(np.log(collection_frequencies[query_term_id]) - np.log(total_terms))
        doc_len = document_lengths[int_document_id] 

        score += (np.log(tf+mu*prob)-np.log(doc_len + mu))*query_term_counts[query_id][query_term_id]
#         score += np.exp(np.log(tf+mu*prob) - np.log(doc_len + mu))
    
    return score

#for mu_val in [0.1, 0.5, 0.9]:
#    dir_prior_func = functools.partial(dirichlet_prior, mu=mu_val)
#    run_retrieval('dirichlet_prior_{}'.format(mu_val), dir_prior_func)


# In[51]:


def abs_discount(int_document_id, query_id, delta):
    score = 0
    for query_term_id in tokenized_queries[query_id]:
#         if int_document_id in inverted_index[query_term_id]:
            # Don't need to divide by len, because already doing that in interpolation (check NLP-1)
            # basically, just a model with unigram probability
        tf = inverted_index[query_term_id].get(int_document_id, 0)

        prob = collection_frequencies[query_term_id]/total_terms
#         prob = np.exp(np.log(collection_frequencies[query_term_id]) - np.log(total_terms))
        doc_len = document_lengths[int_document_id] 
        doc_unique_len = unique_terms_per_document[int_document_id]

        first_term = max(tf - delta, 0)
        second_term = delta*doc_unique_len*prob/doc_len
#         second_term = np.exp(np.log(delta) + np.log(doc_unique_len) + np.log(prob) - np.log(doc_len))

        score += np.log(first_term + second_term)*query_term_counts[query_id][query_term_id]
    
    return score

#for delta_val in [0.1, 0.5, 0.9]:
#    abs_discount_func = functools.partial(abs_discount, delta=delta_val)
#    run_retrieval('abs_discounting_{}'.format(delta_val), abs_discount_func)


# In[19]:


from scipy.stats import norm

def gauss_kernel(i, sigma=50):
    kernel_val = np.exp(-np.power(i, 2)/(2*np.power(sigma,2)))
    return kernel_val

cdflist = dict()

# for i in range(3000):
#     cdflist[i] = norm.cdf(i/50)
# squareroot = np.sqrt(2*np.pi*np.power(sigma,2))
# const2 = 0.5*squareroot

def compute_kernel_table(kernel, integral, limit=2500):
    kernel_table = np.zeros(limit)
    integral_table = np.zeros(limit)
    for i in range(limit):
        kernel_table[i] = kernel(i)
        integral_table[i] = integral(i)
    return kernel_table, integral_table

# def partial_integral(i):
#     return squareroot*cdflist[i] - const2
#     return np.sqrt(2*np.pi*np.power(sigma,2)) * (norm.cdf(i/sigma) - norm.cdf(0))
def gauss_partial_integral(i, sigma=50):
    return np.sqrt(2*np.pi*np.power(sigma,2)) * (norm.cdf(i/sigma) - norm.cdf(0))

def triangle_kernel(i, sigma=50):
    kernel_val = 1-np.abs(i)/sigma if np.abs(i)<=sigma else 0
    return kernel_val

def triangle_partial_integral(i, sigma=50):
    return  i - np.power(i, 2)/(2*sigma) if i < sigma else sigma/2

def cosine_kernel(i, sigma=50):
    kernel_val = 1/2 * (1+np.cos(np.abs(i)*np.pi/sigma))        if np.abs(i)<=sigma        else 0
    return kernel_val

def cosine_partial_integral(i, sigma=50):
    return sigma/2 + sigma*np.sin(np.pi*i/sigma)/(2*np.pi) if i < sigma else sigma/2

def circle_kernel(i, sigma=50):
    kernel_val = np.sqrt(1-np.power(np.abs(i)/sigma,2))        if np.abs(i)<=sigma        else 0
    return kernel_val

def circle_partial_integral(i, sigma=50):
    return sigma/2 * (np.arcsin(i/sigma) + 1/2 * np.sin(2*np.arcsin(i/sigma)))                        if i < sigma else sigma*np.pi/4

def passage_kernel(i, sigma=50):
    kernel_val = 1 if np.abs(i)<=sigma else 0
    return kernel_val

def passage_partial_integral(i, sigma=50):
    return i if i < sigma else sigma

gauss_kernel_table, gauss_integral_table = compute_kernel_table(gauss_kernel, gauss_partial_integral)
triangle_kernel_table, triangle_integral_table = compute_kernel_table(triangle_kernel, triangle_partial_integral)
cosine_kernel_table, cosine_integral_table = compute_kernel_table(cosine_kernel, cosine_partial_integral)
circle_kernel_table, circle_integral_table = compute_kernel_table(circle_kernel, circle_partial_integral)
passage_kernel_table, passage_integral_table = compute_kernel_table(passage_kernel, passage_partial_integral)


# In[20]:


passage_integral_table


# In[21]:


collection_frequencies
norm.cdf(0)


# In[26]:


# import functools
# import numpy as np
# def positional_language_model(int_document_id, query_id, kernel, kernel_integral, mu=0.1):
#     all_scores = np.zeros(document_lengths[int_document_id])
    
#     for query_term_id in tokenized_queries[query_id]:    
#         if int_document_id in inverted_index_positions[query_term_id]:
#             for i in range(document_lengths[int_document_id]):
#                 term_score = 0
#                 score = 0
#                 for query_term_pos in inverted_index_positions[query_term_id]:
#                     term_score += kernel(i, query_term_pos)
#                 prob = collection_frequencies[query_term_id]/total_terms
# #                 prob = np.exp(np.log(collection_frequencies[query_term_id]) - np.log(total_terms))
#                 doc_len = document_lengths[int_document_id] 
#                 norm_value = kernel_integral(i, document_lengths[int_document_id]-1-i)
#                 score = (term_score+mu*prob)/(norm_value + mu)
# #                 score = np.exp(np.log(term_score+mu*prob) - np.log((norm_value + mu)))
#                 all_scores[i] += score
                               
#     print(all_scores)
#     score = np.max(all_scores)
#     return score

# for kernel_func, kernel_integral in [(gauss_kernel, gauss_kernel_integral), 
#                                      (triangle_kernel, triangle_kernel_integral),
#                                      (cosine_kernel, cosine_kernel_integral),
#                                      (circle_kernel, circle_kernel_integral),
#                                      (passage_kernel, passage_kernel_integral)]:
# #     plm_func = functools.partial(positional_language_model, kernel=kernel_func, kernel_integral=kernel_integral, int_document_id=1, query_id = 1)
# #     run_retrieval('Dirichlet smoothed PLM', plm_func)
#     res = positional_language_model(kernel=kernel_func, kernel_integral=kernel_integral, int_document_id=329, query_id='51', mu=1)
#     print(res)
    
import functools
import numpy as np
@profile
def positional_language_model(int_document_id, query_id, kernel, kernel_integral, mu=0.1):
    all_scores = np.zeros(document_lengths[int_document_id])
    for query_term_id in set(tokenized_queries[query_id]):    
        for i in range(document_lengths[int_document_id]):
            term_score = 0
            score = 0
            if int_document_id in inverted_index_positions[query_term_id]:
                for query_term_pos in inverted_index_positions[query_term_id][int_document_id]:
                    term_score += kernel[np.abs(i-query_term_pos)]
                #NEXT LINE
                #TODO: check this
                term_score = min(term_score, 1.0)
            else:
                term_score = 0
            prob_query = query_term_counts[query_id][query_term_id]/len(tokenized_queries[query_id])
            prob = collection_frequencies[query_term_id]/total_terms
            doc_len = document_lengths[int_document_id] 
            norm_value = kernel_integral[i] + kernel_integral[document_lengths[int_document_id]-1-i]
            #NEXT LINE
            prob_word = (term_score+mu*prob)/(norm_value + mu)
            #NEXT LINE
            score = prob_query * np.log(prob_query/prob_word)
#                 assert prob_word <= 1.0
            all_scores[i] += score
    all_scores *= -1    
    score = np.max(all_scores)
    return score
import time
for kernel_func_table, kernel_integral_table in [(gauss_kernel_table, gauss_integral_table), 
                                     (triangle_kernel_table, triangle_integral_table),
                                     (circle_kernel_table, circle_integral_table),
                                     (cosine_kernel_table, cosine_integral_table),
                                     (passage_kernel_table, passage_integral_table)]:
    start=time.time()
    plm_func = functools.partial(positional_language_model, kernel=kernel_func_table, kernel_integral=kernel_integral_table)
    run_retrieval('plm_', plm_func)
    print(time.time() - start)
# for delta_val in [0.1, 0.5, 0.9]:
#     abs_discount_func = functools.partial(abs_discount, delta=delta_val)
#     run_retrieval('abs_discounting_{}'.format(delta_val), abs_discount_func)


# In[16]:


document_lengths


# In[70]:


from collections import defaultdict
import os
import scipy
import matplotlib.pyplot as plt

def validation_best(score_name, param_list, measures):
    result_dicts = []
    for p in param_list:
        result_dicts.append(trec_eval(score_name + '_' + str(p), measures, validation=True))
    vals = []
    
    for res in result_dicts:
#         print(res)
        vals.append(res['ndcg_cut_10']['all'])
    best = np.argmax(vals)
    return vals[best], param_list[best], result_dicts[best], result_dicts
    
def trec_eval(score_name, measures, validation=True):
    validation_str = 'validation' if validation else 'test'
    results = defaultdict(dict)
    a = os.system(str.format('./trec_eval/trec_eval -m all_trec -q ap_88_89/qrel_{} {}.run > trec_results_{}', validation_str, score_name, score_name))
    f = open('trec_results_' + score_name, 'r')
    for line in f.readlines():
        name, id, val = line.strip().split()
        if name in measures:
            val = float(val)
            results[name][id] = val

    f.close()
#     print('returning ', results)
    return results


measures = ['ndcg_cut_10', 'map_cut_1000', 'recall_1000', 'P_5']
scores = ['jelinek_mercer', 'dirichlet_prior','abs_discounting']#, 'abs_discounting']
params = {'jelinek_mercer': [0.1, 0.5, 0.9],
         'abs_discounting': [0.1, 0.5, 0.9],
         'dirichlet_prior': [0.1, 0.5, 0.9]}

ind = np.arange(len(measures))
width = 0.2
colors = ['r', 'g', 'b', 'y', 'k']
fig, ax = plt.subplots()
rs = []
#evaluation measures on with optimized hyper parameters 
for i, score in enumerate(scores):
    best_val, best_param, best_result, results = validation_best(score, params[score], measures)
#     print(best_val)
#     print(best_param)
#     print(best_result['ndcg_cut_10'])
#     print('done')
    means = []
    for measure in measures:
        means.append(sum(best_result[measure].values())) #/ len(best_result[measure])
#     plt.bar(measures, means)
    rs.append(ax.bar(ind + width*i, means, width, color=colors[i]))
ax.legend(rs, scores)
ax.set_ylabel('Scores')
ax.set_title('Scores measure')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(measures)
plt.show()
    
for i, score in enumerate(scores):
    result_dicts = []
    if score in params:
        for p in params[score]:
            result_dicts.append(trec_eval(score + '_' + str(p), measures, validation=False))
    else:
        results_dicts.append(trec_eval(score, measures, validation=False))
    print(result_dicts[i]['ndcg_cut_10'])
    plt.bar([i for i, p in enumerate(result_dicts[i]['ndcg_cut_10'])], [result_dicts[i]['ndcg_cut_10'][j] for j in result_dicts[i]['ndcg_cut_10']])
    plt.show()
    
# to extend, maybe between all pairs?
scipy.stats.ttest_rel(list(result_dicts[0]['ndcg_cut_10'].values()), list(result_dicts[1]['ndcg_cut_10'].values()))


# In[104]:


get_ipython().run_cell_magic('time', '', "# example_query_term_ids = [54, 846, 2208]\n\nexample_document_id = 97064\nexample_query_id = '101'\n# collection_frequencies = {2208: 3822, 846: 9585, 54: 46470}\n# total_terms = 42208958\n# example_query_id = \n# inverted_index[2208][97604]\n\npositional_language_model(example_document_id, example_query_id, kernel=gauss_kernel, kernel_integral=gauss_kernel_integral, mu=0.1)")


# ### Task 2: Latent Semantic Models (LSMs) [20 points] ###
# 
# In this task you will experiment with applying distributional semantics methods ([LSI](http://lsa3.colorado.edu/papers/JASIS.lsi.90.pdf) **[5 points]** and [LDA](https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf) **[5 points]**) for retrieval.
# 
# You do not need to implement LSI or LDA on your own. Instead, you can use [gensim](http://radimrehurek.com/gensim/index.html). An example on how to integrate Pyndri with Gensim for word2vec can be found [here](https://github.com/cvangysel/pyndri/blob/master/examples/word2vec.py). For the remaining latent vector space models, you will need to implement connector classes (such as `IndriSentences`) by yourself.
# 
# In order to use a latent semantic model for retrieval, you need to:
#    * build a representation of the query **q**,
#    * build a representation of the document **d**,
#    * calculate the similarity between **q** and **d** (e.g., cosine similarity, KL-divergence).
#      
# The exact implementation here depends on the latent semantic model you are using. 
#    
# Each of these LSMs come with various hyperparameters to tune. Make a choice on the parameters, and explicitly mention the reasons that led you to these decisions. You can use the validation set to optimize hyper parameters you see fit; motivate your decisions. In addition, mention clearly how the query/document representations were constructed for each LSM and explain your choices.
# 
# In this experiment, you will first obtain an initial top-1000 ranking for each query using TF-IDF in **Task 1**, and then re-rank the documents using the LSMs. Use TREC Eval to obtain the results and report on `NDCG@10`, Mean Average Precision (`MAP@1000`), `Precision@5` and `Recall@1000`.
# 
# Perform significance testing **[5 points]** (similar as in Task 1) in the class of semantic matching methods.
# 
# Perform analysis **[5 points]**

# ### Task 3:  Word embeddings for ranking [10 points] ###
# 
# First create word embeddings on the corpus we provided using [word2vec](http://arxiv.org/abs/1411.2738) -- [gensim implementation](https://radimrehurek.com/gensim/models/word2vec.html). You should extract the indexed documents using pyndri and provide them to gensim for training a model (see example [here](https://github.com/nickvosk/pyndri/blob/master/examples/word2vec.py)).
# 
# Try one of the following (increasingly complex) methods for building query and document representations:
#    * Average or sum the word vectors.
#    * Cluster words in the document using [k-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and use the centroid of the most important cluster. Experiment with different values of K for k-means.
#    * Using the [bag-of-word-embeddings representation](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1248).
#    
# Note that since we provide the implementation for training word2vec, you will be graded based on your creativity on combining word embeddings for building query and document representations.
# 
# Note: If you want to experiment with pre-trained word embeddings on a different corpus, you can use the word embeddings we provide alongside the assignment (./data/reduced_vectors_google.txt). These are the [google word2vec word embeddings](https://code.google.com/archive/p/word2vec/), reduced to only the words that appear in the document collection we use in this assignment.

# In[ ]:


get_ipython().run_cell_magic('time', '', "import copy\nimport gensim\nimport logging\nimport pyndri\nimport pyndri.compat\nimport sys\n\n\nword2vec_init = gensim.models.Word2Vec(\n    size=300,  # Embedding size\n    window=5,  # One-sided window size\n    sg=True,  # Skip-gram.\n    min_count=5,  # Minimum word frequency.\n    sample=1e-3,  # Sub-sample threshold.\n    hs=False,  # Hierarchical softmax.\n    negative=10,  # Number of negative examples.\n    iter=1,  # Number of iterations.\n    workers=8,  # Number of workers.\n)\n\nwith pyndri.open('index/') as index:\n    logging.info('Loading vocabulary.')\n    dictionary = pyndri.extract_dictionary(index)\n    sentences = pyndri.compat.IndriSentences(index, dictionary)\n\n    logging.info('Constructing word2vec vocabulary.')\n\n    # Build vocab.\n    word2vec_init.build_vocab(sentences, trim_rule=None)\n#     print(len(word2vec_init.corpus_count))\n#     print(len(dictionary))\n#     print(len(sentences))\n    models = [word2vec_init]\n\n    for epoch in range(1, 5 + 1):\n        logging.info('Epoch %d', epoch)\n\n        model = copy.deepcopy(models[-1])\n        model.train(sentences, total_examples=len(sentences), epochs=model.iter)\n\n        models.append(model)\n\n    logging.info('Trained models: %s', models)")


# ### Task 4: Learning to rank (LTR) [10 points] ###
# 
# In this task you will get an introduction into learning to rank for information retrieval, in particular pointwise learning to rank.
# 
# You will experiment with a pointwise learning to rank method, logistic regression, implemented in [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
# Train your LTR model using 10-fold cross validation on the test set.
# 
# You can explore different ways for devising features for the model. Obviously, you can use the retrieval methods you implemented in Task 1 and Task 2 as features. Think about other features you can use (e.g. query/document length). 
# One idea is to also explore external sources such as Wikipedia entities (?). Creativity on devising new features and providing motivation for them will be taken into account when grading.
# 
# For every query, first create a document candidate set using the top-1000 documents using TF-IDF, and subsequently compute features given a query and a document. Note that the feature values of different retrieval methods are likely to be distributed differently.

# In[52]:


score = 'tfidf'
measures = ['P_5']
ranking = trec_eval(score + '_' + str(p), measures, validation=False)
# sort and get the top 1000, using which score?
print(ranking['P_5'])
#top1000 = list(sorted(ranking['P_5']


# ### Task 4: Write a report [20 points; instant FAIL if not provided] ###
# 
# The report should be a PDF file created using the [sigconf ACM template](https://www.acm.org/publications/proceedings-template) and will determine a significant part of your grade.
# 
#    * It should explain what you have implemented, motivate your experiments and detail what you expect to learn from them. **[10 points]**
#    * Lastly, provide a convincing analysis of your results and conclude the report accordingly. **[10 points]**
#       * Do all methods perform similarly on all queries? Why?
#       * Is there a single retrieval model that outperforms all other retrieval models (i.e., silver bullet)?
#       * ...
# 
# **Hand in the report and your self-contained implementation source files.** Only send us the files that matter, organized in a well-documented zip/tgz file with clear instructions on how to reproduce your results. That is, we want to be able to regenerate all your results with minimal effort. You can assume that the index and ground-truth information is present in the same file structure as the one we have provided.
# 
