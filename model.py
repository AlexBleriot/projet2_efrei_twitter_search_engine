# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:13:57 2020

@author: Caroline Azeufack
"""

import json
import os

from joblib import dump

import pandas as pd
import re
import string

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

print("--- Load data ---")
print(".......")
tweet = pd.read_csv("tweets.csv")

print("--- Focus on <text> column ---")
print(".......")
tweets = tweet['text']

def text_clean(text):
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text) # remove the punctuation
    #text=re.sub('\w*\d\w*','',text) # remove digit
    text= re.sub('\n','',text)
    return text
cleaned= lambda x: text_clean(x)
cleaned_tweets=pd.DataFrame(tweets.apply(cleaned))


# Tokenization of each document
print("--- Tokenization of each document ---")
print(".......")
sentences = cleaned_tweets['text']
tokenized_sent = []
for s in sentences:
    tokenized_sent.append(word_tokenize(s.lower()))
 
print("--- TaggedDocument ---")
print(".......")   
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
taggeData=pd.DataFrame(tagged_data)
taggeData.to_csv("tagged_data.csv")


## Train doc2vec model
print("--- Train doc2vec model ---")
print(".......")  
model = Doc2Vec(tagged_data, vector_size = 100, window = 2, min_count = 1, epochs = 100)

#computing meta data
result = []
test_doc = word_tokenize("trump")
test_doc_vector = model.infer_vector(test_doc)
similar_doc=model.docvecs.most_similar(positive = [test_doc_vector], topn=20)



print("Serializing model to: {}".format(MODEL_PATH))
dump(model, MODEL_PATH)














"""
# Saving model to disk
print("--- Saving model to disk ---")
print(".......") 
pickle.dump(model, open('model.pkl','wb'))
"""




"""
# Loading model to test
print("--- Loading model to test ---")
print(".......") 
model = pickle.load(open('model.pkl','rb'))
test_doc = word_tokenize("trump")
test_doc_vector = model.infer_vector(test_doc)
similar_doc=model.docvecs.most_similar(positive = [test_doc_vector], topn=20)

for i in range(0,len(similar_doc)):
    sen = list(tagged_data[int(similar_doc[i][0])])
    finalString = ' '.join(sen[0])
    print(finalString.capitalize())
    print('\n')
"""



