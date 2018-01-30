num_topics = [10, 25, 50]

# import numpy as np
import gensim
from scipy import spatial

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
                assert isinstance(object_assesments[0][1], str) or \
                       isinstance(object_assesments[0][1], bytes)

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




import pyndri

index = pyndri.Index('index/')




import collections
import io
import logging
import sys

def parse_topics(file_or_files,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and \
            not isinstance(file_or_files, tuple):
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




token2id, id2token, _ = index.get_dictionary()
print(list(id2token.items())[:15])

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

# TODO: Explain why we need it
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
                doc_token_ids = [doc_token_id for doc_token_id in doc_token_ids if doc_token_id > 0]

                document_bow = collections.Counter(doc_token_ids)

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

import sys


def run_retrieval(model_name, score_fn, top_1000=None):
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
                global document_ids
                if top_1000 is not None:
                        print('using top 1000')
                        document_ids = top_1000
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





ext2int = {index.ext_document_id(i): i for i in range(index.document_base(), index.maximum_document())}
# print(ext2int['AP890308-0144'])
from collections import defaultdict
def get_first_1000(measure):
    f = open('{}.run'.format(measure), 'r')
    top = defaultdict(list)
    top_values = defaultdict(list)

    for line in f.readlines():
        splitted = line.split(' ')
        query_id, ext_id, rank, tfidf = int(splitted[0]), splitted[2], int(splitted[3]), float(splitted[4])
        top[query_id].append(ext2int[ext_id])
        top_values[query_id].append(tfidf)
#         print(ext_id, ext2int[ext_id])
    f.close()
    return top, top_values
top_1000, _ = get_first_1000('tfidf')





# for the LdaModel the corpus should be a list of docs, where each document is represented as a list of tuples,
# where a tuple means a word and occurence
my_corpus = []
# iterate over the documents
for int_doc_id in range(index.document_base(), index.maximum_document()):
        ext_doc_id, doc_token_ids = index.document(int_doc_id)
        doc_token_ids = [doc_token_id for doc_token_id in doc_token_ids if doc_token_id > 0]
        document_bow = collections.Counter(doc_token_ids).most_common(len(doc_token_ids))
        my_corpus.append(document_bow)





############################################################################################################################
num_topics = [10, 25, 50]

