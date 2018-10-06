# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:03:21 2018

@author: Wheelspawn
"""


import pandas as ps

my_dir = 'data/'

mit=pd.read_csv(my_dir+'MIT-IEEE-eBooks.csv')
springer=pd.read_csv(my_dir+'Springer-eBooks.csv')
wiley=pd.read_csv(my_dir+'Wiley-IEEE-eBooks.csv')