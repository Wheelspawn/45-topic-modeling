#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:18:28 2018

@author: jieluyao
"""
###############################################################################
# Name: Supervised learning for text classification of lexis articles
# Author: Jielu Yao
# Date: 14 April 2018
# Input files: lexis_articles.csv
# Output files: machine_aboutcm.xlsxw
###############################################################################


#cd /Volumes/Work/RA/Bryce/school_activism/task6_supervised_learning

# Prerequisite and setting up the environment.
import pandas as pd
import re

class StemTokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer() 
    def __call__(self, doc):
        return [self.wnl.stem(t).lower() for t in re.split(r'[\W\d]',doc) if t not in string.punctuation and len(t) > 1 and t != 'said']
    

docs=["tm_preprocessor/data/MIT-IEEE-eBooks.csv",
      "tm_preprocessor/data/Springer-eBooks.csv",
      "tm_preprocessor/data/Wiley-IEEE-eBooks.csv"]

publishers=['MIT','Springer','Wiley']

for i in range(len(docs)):

    # Load the dataset
    # cm_all = pd.DataFrame.from_csv("cm_all.csv")
    cm_all = pd.read_csv(docs[i])
    
    # #### Custom tokenizer for creating document term matrix 
    #- Use of stemming from NLTK (Natural Language Toolkit) package
    
    from nltk.stem.porter import PorterStemmer 
    import string
    
    # Use a CountVectorizer to create the document term matrix
    from sklearn.feature_extraction.text import CountVectorizer
    
    count = CountVectorizer(tokenizer=StemTokenizer(),stop_words='english', max_df=0.99, min_df=0.01) 
    df_bag = count.fit_transform(cm_all['Title'].values.astype('U'))
    
    feature_names = count.get_feature_names()
    
    feature_names
    
    len(feature_names)
    
    ################################################################################    
    # Unsupervised learning
    ###############################################################################
    import sklearn.decomposition as dec
    
    lda = dec.LatentDirichletAllocation(n_topics=10)
    lda.fit(df_bag)
    n_top_words = 5
    
    print('\n')
    print(publishers[i])
    
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d: " % topic_idx, end='')
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
    
    '''



###############################################################################    
# Supervised learning
###############################################################################

### Classification: Is this article about campus mobilization?
###############################################################################
coder = cm_all[cm_all.about_cm.notnull()]
machine = cm_all[cm_all.about_cm.isnull()]

coder.shape
machine.shape

# Model selection 
from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['Book Title'].values.astype('U'))
ms_train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['Book Title'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.about_cm)
ms_pred_MNB_aboutcm = ms_clf_MNB.predict(ms_test_tfidf)

import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['about_cm'], ms_pred_MNB_aboutcm)
print("Accuracy:", mt.accuracy_score(ms_test['about_cm'],ms_pred_MNB_aboutcm))
print("AUC:", mt.roc_auc_score(ms_test['about_cm'],ms_pred_MNB_aboutcm))
print(mt.classification_report(ms_test.about_cm, ms_pred_MNB_aboutcm))

## SVM
from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.about_cm)
ms_pred_svc_aboutcm = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['about_cm'], ms_pred_svc_aboutcm)
print("Accuracy:", mt.accuracy_score(ms_test['about_cm'],ms_pred_svc_aboutcm))
print("AUC:", mt.roc_auc_score(ms_test['about_cm'],ms_pred_svc_aboutcm))
print(mt.classification_report(ms_test.about_cm, ms_pred_svc_aboutcm))

## Random forest
from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.about_cm)
ms_pred_forest_aboutcm = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['about_cm'], ms_pred_forest_aboutcm)
print("Accuracy:", mt.accuracy_score(ms_test['about_cm'],ms_pred_forest_aboutcm))
print("AUC:", mt.roc_auc_score(ms_test['about_cm'],ms_pred_forest_aboutcm))
print(mt.classification_report(ms_test.about_cm, ms_pred_forest_aboutcm))

# Given the accuracy report, we use MultinomialNB to classify the articles 
train_bag = count.fit_transform(coder['Book Title'].values.astype('U'))
train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_bag)
train_tfidf.shape

## Running ML algorithms: MultinomialNB
clf_MNB = MultinomialNB().fit(train_tfidf, coder.about_cm)
test_bag = count.transform(machine['Book Title'].values.astype('U'))
test_bag.shape
test_tfidf = tfidf_transformer.transform(test_bag)
test_tfidf.shape
pred_MNB_aboutcm = clf_MNB.predict(test_tfidf)

# Add prediction result to the machine dataframe
machine['about_cm_pred_MNB'] = pred_MNB_aboutcm

# Save predicted values
writer = pd.ExcelWriter('machine_aboutcm.xlsx')
machine.to_excel(writer,'Sheet1')
writer.save()


### Classification: Were.the.police.mentioned.in.the.MAIN.event?
###############################################################################
coder_police = cm_all[cm_all.police.notnull()]
machine_police = cm_all[cm_all.police.isnull()]

# Model selection 
#from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder_police, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['Book Title'].values.astype('U'))
ms_train_bag.shape

#from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['Book Title'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
#from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.police)
ms_pred_MNB_police = ms_clf_MNB.predict(ms_test_tfidf)

#import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['police'], ms_pred_MNB_police)
print("Accuracy:", mt.accuracy_score(ms_test['police'],ms_pred_MNB_police))
print("AUC:", mt.roc_auc_score(ms_test['police'],ms_pred_MNB_police))
print(mt.classification_report(ms_test.police, ms_pred_MNB_police))

## SVM
#from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.police)
ms_pred_svc_police = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['police'], ms_pred_svc_police)
print("Accuracy:", mt.accuracy_score(ms_test['police'],ms_pred_svc_police))
print("AUC:", mt.roc_auc_score(ms_test['police'],ms_pred_svc_police))
print(mt.classification_report(ms_test.police, ms_pred_svc_police))

## Random forest
#from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.police)
ms_pred_forest_police = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['police'], ms_pred_forest_police)
print("Accuracy:", mt.accuracy_score(ms_test['police'],ms_pred_forest_police))
print("AUC:", mt.roc_auc_score(ms_test['police'],ms_pred_forest_police))
print(mt.classification_report(ms_test.police, ms_pred_forest_police))

# Given the accuracy report (similar results), we use MultinomialNB to 
# classify the articles
train_bag = count.fit_transform(coder_police['text'].values.astype('U'))
train_bag.shape

#from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_bag)
train_tfidf.shape

## Running ML algorithms: MultinomialNB
clf_MNB = MultinomialNB().fit(train_tfidf, coder_police.police)
test_bag = count.transform(machine_police['text'].values.astype('U'))
test_bag.shape
test_tfidf = tfidf_transformer.transform(test_bag)
test_tfidf.shape
pred_MNB_police = clf_MNB.predict(test_tfidf)

# Add prediction result to the machine dataframe
machine_police['police_pred_MNB'] = pred_MNB_police


### Classification: Q15
###############################################################################
coder_q15 = cm_all[cm_all.q15.notnull()]
machine_q15 = cm_all[cm_all.q15.isnull()]

# Model selection 
from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder_q15, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['text'].values.astype('U'))
ms_train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['text'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.q15)
ms_pred_MNB_q15 = ms_clf_MNB.predict(ms_test_tfidf)

import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['q15'], ms_pred_MNB_q15)
print("Accuracy:", mt.accuracy_score(ms_test['q15'],ms_pred_MNB_q15))
print("AUC:", mt.roc_auc_score(ms_test['q15'],ms_pred_MNB_q15))
print(mt.classification_report(ms_test.q15, ms_pred_MNB_q15))

#Accuracy: 0.7333333333333333
#AUC: 0.5

## SVM
from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.q15)
ms_pred_svc_q15 = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q15'], ms_pred_svc_q15)
print("Accuracy:", mt.accuracy_score(ms_test['q15'],ms_pred_svc_q15))
print("AUC:", mt.roc_auc_score(ms_test['q15'],ms_pred_svc_q15))
print(mt.classification_report(ms_test.q15, ms_pred_svc_q15))

#Accuracy: 0.7333333333333333
#AUC: 0.5

## Random forest
from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.q15)
ms_pred_forest_q15 = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q15'], ms_pred_forest_q15)
print("Accuracy:", mt.accuracy_score(ms_test['q15'],ms_pred_forest_q15))
print("AUC:", mt.roc_auc_score(ms_test['q15'],ms_pred_forest_q15))
print(mt.classification_report(ms_test.q15, ms_pred_forest_q15))

#Accuracy: 0.7333333333333333
#AUC: 0.5568181818181819

# Get predicted q15: use RandomForest to classify the articles
train_bag = count.fit_transform(coder_q15['text'].values.astype('U'))
train_bag.shape

#from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_bag)
train_tfidf.shape

## Running ML algorithms: Random Forest
clf_random = RandomForestClassifier().fit(train_tfidf, coder_q15.q15)
test_bag = count.transform(machine_q15['text'].values.astype('U'))
test_bag.shape
test_tfidf = tfidf_transformer.transform(test_bag)
test_tfidf.shape
pred_random_q15 = clf_random.predict(test_tfidf)

# Add prediction result to the machine dataframe
machine_q15['q15_pred_random'] = pred_random_q15

# Save to excel files
writer = pd.ExcelWriter('q15.xlsx')
machine_q15.to_excel(writer,'Sheet1')
coder_q15.to_excel(writer,'Sheet2')
writer.save()

### Classification: Q16
###############################################################################
coder_q16 = cm_all[cm_all.q16.notnull()]
machine_q16 = cm_all[cm_all.q16.isnull()]

# Model selection 
from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder_q16, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['text'].values.astype('U'))
ms_train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['text'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.q16)
ms_pred_MNB_q16 = ms_clf_MNB.predict(ms_test_tfidf)

import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['q16'], ms_pred_MNB_q16)
print("Accuracy:", mt.accuracy_score(ms_test['q16'],ms_pred_MNB_q16))
print("AUC:", mt.roc_auc_score(ms_test['q16'],ms_pred_MNB_q16))
print(mt.classification_report(ms_test.q16, ms_pred_MNB_q16))

#Accuracy: 0.638095238095238
#AUC: 0.5

## SVM
from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.q16)
ms_pred_svc_q16 = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q16'], ms_pred_svc_q16)
print("Accuracy:", mt.accuracy_score(ms_test['q16'],ms_pred_svc_q16))
print("AUC:", mt.roc_auc_score(ms_test['q16'],ms_pred_svc_q16))
print(mt.classification_report(ms_test.q16, ms_pred_svc_q16))

#Accuracy: 0.638095238095238
#AUC: 0.5

## Random forest
from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.q16)
ms_pred_forest_q16 = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q16'], ms_pred_forest_q16)
print("Accuracy:", mt.accuracy_score(ms_test['q16'],ms_pred_forest_q16))
print("AUC:", mt.roc_auc_score(ms_test['q16'],ms_pred_forest_q16))
print(mt.classification_report(ms_test.q16, ms_pred_forest_q16))

#Accuracy: 0.6190476190476191
#AUC: 0.5192458758837392

# Get predicted q16: use RandomForest to classify the articles
train_bag = count.fit_transform(coder_q16['text'].values.astype('U'))
train_bag.shape

#from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_bag)
train_tfidf.shape

## Running ML algorithms: Random Forest
clf_random = RandomForestClassifier().fit(train_tfidf, coder_q16.q16)
test_bag = count.transform(machine_q16['text'].values.astype('U'))
test_bag.shape
test_tfidf = tfidf_transformer.transform(test_bag)
test_tfidf.shape
pred_random_q16 = clf_random.predict(test_tfidf)

# Add prediction result to the machine dataframe
machine_q16['q16_pred_random'] = pred_random_q16

# Save to excel files
writer = pd.ExcelWriter('q16.xlsx')
machine_q16.to_excel(writer,'Sheet1')
coder_q16.to_excel(writer,'Sheet2')
writer.save()

### Classification: Q17
###############################################################################
coder_q17 = cm_all[cm_all.q17.notnull()]
machine_q17 = cm_all[cm_all.q17.isnull()]

# Model selection 
from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder_q17, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['text'].values.astype('U'))
ms_train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['text'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.q17)
ms_pred_MNB_q17 = ms_clf_MNB.predict(ms_test_tfidf)

import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['q17'], ms_pred_MNB_q17)
print("Accuracy:", mt.accuracy_score(ms_test['q17'],ms_pred_MNB_q17))
print("AUC:", mt.roc_auc_score(ms_test['q17'],ms_pred_MNB_q17))
print(mt.classification_report(ms_test.q17, ms_pred_MNB_q17))

#Accuracy: 0.5714285714285714
#AUC: 0.5778061224489796

## SVM
from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.q17)
ms_pred_svc_q17 = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q17'], ms_pred_svc_q17)
print("Accuracy:", mt.accuracy_score(ms_test['q17'],ms_pred_svc_q17))
print("AUC:", mt.roc_auc_score(ms_test['q17'],ms_pred_svc_q17))
print(mt.classification_report(ms_test.q17, ms_pred_svc_q17))

#Accuracy: 0.4666666666666667
#AUC: 0.5

## Random forest
from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.q17)
ms_pred_forest_q17 = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q17'], ms_pred_forest_q17)
print("Accuracy:", mt.accuracy_score(ms_test['q17'],ms_pred_forest_q17))
print("AUC:", mt.roc_auc_score(ms_test['q17'],ms_pred_forest_q17))
print(mt.classification_report(ms_test.q17, ms_pred_forest_q17))

#Accuracy: 0.5142857142857142
#AUC: 0.5102040816326531

# Get predicted q17: use MNB to classify the articles
train_bag = count.fit_transform(coder_q17['text'].values.astype('U'))
train_bag.shape

#from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_bag)
train_tfidf.shape

## Running ML algorithms: MultinomialNB
clf_MNB = MultinomialNB().fit(train_tfidf, coder_q17.q17)
test_bag = count.transform(machine_q17['text'].values.astype('U'))
test_bag.shape
test_tfidf = tfidf_transformer.transform(test_bag)
test_tfidf.shape
pred_MNB_q17 = clf_MNB.predict(test_tfidf)

# Add prediction result to the machine dataframe
machine_q17['q17_pred_MNB'] = pred_MNB_q17

# Save to excel files
writer = pd.ExcelWriter('q17.xlsx')
machine_q17.to_excel(writer,'Sheet1')
coder_q17.to_excel(writer,'Sheet2')
writer.save()

### Classification: Q18
###############################################################################
coder_q18 = cm_all[cm_all.q18.notnull()]
machine_q18 = cm_all[cm_all.q18.isnull()]

# Model selection 
from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder_q18, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['text'].values.astype('U'))
ms_train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['text'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.q18)
ms_pred_MNB_q18 = ms_clf_MNB.predict(ms_test_tfidf)

import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['q18'], ms_pred_MNB_q18)
print("Accuracy:", mt.accuracy_score(ms_test['q18'],ms_pred_MNB_q18))
print("AUC:", mt.roc_auc_score(ms_test['q18'],ms_pred_MNB_q18))
print(mt.classification_report(ms_test.q18, ms_pred_MNB_q18))

#Accuracy: 0.9142857142857143
#AUC: 0.5

## SVM
from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.q18)
ms_pred_svc_q18 = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q18'], ms_pred_svc_q18)
print("Accuracy:", mt.accuracy_score(ms_test['q18'],ms_pred_svc_q18))
print("AUC:", mt.roc_auc_score(ms_test['q18'],ms_pred_svc_q18))
print(mt.classification_report(ms_test.q18, ms_pred_svc_q18))

#Accuracy: 0.9142857142857143
#AUC: 0.5

## Random forest
from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.q18)
ms_pred_forest_q18 = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['q18'], ms_pred_forest_q18)
print("Accuracy:", mt.accuracy_score(ms_test['q18'],ms_pred_forest_q18))
print("AUC:", mt.roc_auc_score(ms_test['q18'],ms_pred_forest_q18))
print(mt.classification_report(ms_test.q18, ms_pred_forest_q18))

#Accuracy: 0.9142857142857143
#AUC: 0.5

### Classification: Q23 police
###############################################################################
coder_q23 = cm_all[cm_all.police.notnull()]
machine_q23 = cm_all[cm_all.police.isnull()]

# Model selection 
from sklearn.model_selection import train_test_split
ms_train, ms_test = train_test_split(coder_q23, test_size = 0.3)

ms_train_bag = count.fit_transform(ms_train['text'].values.astype('U'))
ms_train_bag.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
ms_train_tfidf = tfidf_transformer.fit_transform(ms_train_bag)
ms_train_tfidf.shape

ms_test_bag = count.transform(ms_test['text'].values.astype('U'))
ms_test_bag.shape
ms_test_tfidf = tfidf_transformer.transform(ms_test_bag)
ms_test_tfidf.shape

## MultinomialNB
from sklearn.naive_bayes import MultinomialNB
ms_clf_MNB = MultinomialNB().fit(ms_train_tfidf, ms_train.police)
ms_pred_MNB_q23 = ms_clf_MNB.predict(ms_test_tfidf)

import sklearn.metrics as mt 
mt.confusion_matrix(ms_test['police'], ms_pred_MNB_q23)
print("Accuracy:", mt.accuracy_score(ms_test['police'],ms_pred_MNB_q23))
print("AUC:", mt.roc_auc_score(ms_test['police'],ms_pred_MNB_q23))
print(mt.classification_report(ms_test.police, ms_pred_MNB_q23))

#Accuracy: 0.8666666666666667
#AUC: 0.5

## SVM
from sklearn.svm import SVC
ms_clf_svc = SVC().fit(ms_train_tfidf,ms_train.police)
ms_pred_svc_q23 = ms_clf_svc.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['police'], ms_pred_svc_q23)
print("Accuracy:", mt.accuracy_score(ms_test['police'],ms_pred_svc_q23))
print("AUC:", mt.roc_auc_score(ms_test['police'],ms_pred_svc_q23))
print(mt.classification_report(ms_test.police, ms_pred_svc_q23))

#Accuracy: 0.8666666666666667
#AUC: 0.5

## Random forest
from sklearn.ensemble import RandomForestClassifier
ms_clf_forest = RandomForestClassifier().fit(ms_train_tfidf,ms_train.police)
ms_pred_forest_q23 = ms_clf_forest.predict(ms_test_tfidf)

mt.confusion_matrix(ms_test['police'], ms_pred_forest_q23)
print("Accuracy:", mt.accuracy_score(ms_test['police'],ms_pred_forest_q23))
print("AUC:", mt.roc_auc_score(ms_test['police'],ms_pred_forest_q23))
print(mt.classification_report(ms_test.police, ms_pred_forest_q23))

#Accuracy: 0.8666666666666667
#AUC: 0.5302197802197802

# Get predicted q23: use RandomForest to classify the articles
train_bag = count.fit_transform(coder_q23['text'].values.astype('U'))
train_bag.shape

#from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_bag)
train_tfidf.shape

## Running ML algorithms: Random Forest
clf_random = RandomForestClassifier().fit(train_tfidf, coder_q23.police)
test_bag = count.transform(machine_q23['text'].values.astype('U'))
test_bag.shape
test_tfidf = tfidf_transformer.transform(test_bag)
test_tfidf.shape
pred_random_q23 = clf_random.predict(test_tfidf)

# Add prediction result to the machine dataframe
machine_q23['q23_pred_random'] = pred_random_q23

# Save to excel files
writer = pd.ExcelWriter('q23.xlsx')
machine_q23.to_excel(writer,'Sheet1')
coder_q23.to_excel(writer,'Sheet2')
writer.save()

'''