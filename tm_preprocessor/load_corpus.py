# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:03:21 2018

@author: Wheelspawn
"""

import pandas as pd
import numpy as np
from tm_preprocessor import Preprocessor

my_dir = 'data/'

mit=pd.read_csv(my_dir+'MIT-IEEE-eBooks.csv',encoding = "ISO-8859-1", header=0, sep=",")
springer=pd.read_csv(my_dir+'Springer-eBooks.csv',encoding = "ISO-8859-1", header=0, sep=",")
wiley=pd.read_csv(my_dir+'Wiley-IEEE-eBooks.csv',encoding = "ISO-8859-1", header=0, sep=",")

mit_titles=list(np.transpose(np.array(mit["Title"])))
springer_titles=list(np.transpose(np.array(springer["Book Title"])))
wiley_titles=list(np.transpose(np.array(wiley["Title"])))

p=Preprocessor(documents=mit_titles)
p.remove_digits_punctuactions()
print(p.get_word_ranking())
