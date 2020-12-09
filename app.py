from flask import Flask, request, render_template

import ast
import pandas as pd
import re
import string

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from joblib import load

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

app = Flask(__name__)
model = load(MODEL_PATH)


def tranform(sen):
	sen = sen.values
	finalString = ' '.join(map(str, sen))
	elt = ast.literal_eval(finalString)
	finalelt = ' '.join(map(str, elt))
	return finalelt
     
@app.route('/')
def home():   
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
	tweet = pd.read_csv("tweets.csv")
	tweets = tweet['text']

	output = []
	score = []
	
	if request.method == 'POST':
	    enter = request.form['search']
	    test_doc = word_tokenize(enter.lower())
	    test_doc_vector = model.infer_vector(test_doc)
	    similar_doc=model.docvecs.most_similar(positive = [test_doc_vector], topn=20)
	    
	    for i in range(0,len(similar_doc)):
	    	index = int(similar_doc[i][0])
	    	score = similar_doc[i][1]
	    	sen = tweets[index]
	    	tup = (score, sen)
    		output.append(tup)

    		
    	
	return render_template('result.html', tweets=output)
    
if __name__ == '__main__':
	app.run(host='0.0.0.0', port="5000", debug=True)
