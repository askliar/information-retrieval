# Reinforcement Learning

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE)

## Description

Labs of the [Information Retrieval 1](http://studiegids.uva.nl/xmlpages/page/2017-2018/zoek-vak/vak/34035) course of the MSc in Artificial Intelligence at the University of Amsterdam. Joint work with [Gabriele Bani](https://github.com/Hiryugan) and [Bence Keresztury](https://github.com/bencee16).

#### Lab 1 - Evaluation Measures, Interleaving and Click Models

[Problem statement and Solution](https://github.com/askliar/information-retrieval/homework-1/hw1-lab.ipynb)

Commercial search engines typically use a funnel approach in evaluating a new search algorithm: they first use an offline test collection to compare the production algorithm (P) with the new experimental algorithm (E); if E outperforms P with respect to the evaluation measure of their interest, the two algorithms are then compared online through an interleaving experiment.

Therefore, one of the key questions in IR is whether offline evaluation and online evaluation outcomes agree with each other. In this lab we run experiments in which we compare different online and offline evaluation metrics in order to understand the relationship between them. Multiple click models are used in order to simulate user clicks.  

#### Lab 2 - A Study of Lexical and Semantic Language Models Applied to Information Retrieval and Learning to Rank

[Problem statement and Solution](https://github.com/askliar/information-retrieval/homework-2/hw2-lab.ipynb) - [Report](https://github.com/askliar/information-retrieval/homework-2/hw2-report.pdf)

In a typical IR task we are interested in finding a (usually ranked) list of results that satisfy the information need of a user expressed by means of a query. Many difficulties arise as the satisfaction and information need of the user cannot be directly observed, and thus queries from users can be interpreted in many ways. Moreover, the query is merely a linguistic representation of the actual information need of the user and the gap between them can not be measured either. 

In this lab we study three families of models that are used to measure and rank the relevance of a set of documents given a query: lexical models, semantic models, and machine learning-based re-ranking algorithms that build on top of the former models. 

## Running

Refer to each notebook name and run Jupyter notebook with a following command:
``` 
jupyter notebook #notebook#.ipynb
```

## Dependencies

- NumPy
- PyTorch
- Matplotlib
- Scipy
- Pyndri
- Gensim
- Trec Eval
- Pandas

## Copyright

Copyright © 2018 Andrii Skliar.

<p align=“justify”>
This project is distributed under the <a href="LICENSE">MIT license</a>. This was developed as part of the Reinforcement Learning course taught by Herke van Hoof at the University of Amsterdam. Please follow the <a href="http://student.uva.nl/en/content/az/plagiarism-and-fraud/plagiarism-and-fraud.html">UvA regulations governing Fraud and Plagiarism</a> in case you are a student.
</p>
