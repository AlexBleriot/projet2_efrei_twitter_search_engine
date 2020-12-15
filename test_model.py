import unittest
import os
import json
from joblib import load
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')

#information regarding the model
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

class FlaskTest(unittest.TestCase):
    def setUp(self): #run at the beginning of the test
        pass
    def tearDown(self):# runs at the end of the test
        pass
        
        
    def test_a_find_similarTweets(self):
        model = load(MODEL_PATH)
        #tweet with id=200
        sample_tweet_id=200
        sample_tweet="Poll numbers are starting to look very good. Leading in Florida @CNN Arizona and big jump in Utah. All numbers rising, national way up. Wow"
        #create word tokens as well as remove punctuation 
        rem_tok_punc= RegexpTokenizer(r'\w+')
        tokens =rem_tok_punc.tokenize(sample_tweet)
        #convert the words to lower case
        words=[w.lower() for w in tokens]
        #involke all the english stopwords
        stop_word_list=set(stopwords.words('english'))
        #remove stops words
        sample_tweet_doc=[w for w in words if not w in stop_word_list]
        
        #infer the vector representation of sample_tweet_doc by using our model
        sample_tweet_doc_vector = model.infer_vector(sample_tweet_doc)
        similar_doc=model.docvecs.most_similar(positive = [sample_tweet_doc_vector], topn=20)
        
        #get the id of the 20 most similar tweets and store them into first_tuple_elements
        first_tuple_elements = [a_tuple[0] for a_tuple in similar_doc]
        
        #if sample_tweet_id belong to first_tuple_elements, then the test is good
        self.assertTrue(sample_tweet_id in first_tuple_elements)
        
        
    
        
if __name__ =='__main__':
    unittest.main()

