# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:03:21 2018

@author: Wheelspawn
"""

import pandas as pd
import numpy as np
from tm_preprocessor import Preprocessor

from gensim import corpora, models

from sklearn.decomposition import LatentDirichletAllocation as LDA

my_dir = 'data/'

mit=pd.read_csv(my_dir+'MIT-IEEE-eBooks.csv',encoding = "ISO-8859-1", header=0, sep=",")
springer=pd.read_csv(my_dir+'Springer-eBooks.csv',encoding = "ISO-8859-1", header=0, sep=",")
wiley=pd.read_csv(my_dir+'Wiley-IEEE-eBooks.csv',encoding = "ISO-8859-1", header=0, sep=",")

mit_titles=list(np.transpose(np.array(mit["Title"])))
springer_titles=list(np.transpose(np.array(springer["Book Title"])))
wiley_titles=list(np.transpose(np.array(wiley["Title"])))

mit_topics=len(set(list(np.transpose(np.array(mit["Subjects"])))))
springer_topics=len(set(list(np.transpose(np.array(springer["Subject Classification"])))))
wiley_topics=len(set(list(np.transpose(np.array(wiley["Subjects"])))))

mit_topics=30
springer_topics=30
wiley_topics=30

s=Preprocessor(documents=springer_titles)
s.remove_digits_punctuactions()

springer_dict = gensim.corpora.Dictionary(s.corpus)
springer_dict.filter_extremes(no_below=10, no_above=0.75, keep_n=100000)
bow_corpus = [springer_dict.doc2bow(doc) for doc in s.corpus]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# lda = LDA(n_topics=springer_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(bow_corpus)
