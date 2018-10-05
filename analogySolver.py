from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from nltk.corpus import words

from gensim.test.utils import common_texts, get_tmpfile
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from random import shuffle
import pickle

import gzip
import gensim 
import logging
 
#logging.basicConfig(format=’%(asctime)s : %(levelname)s : %(message)s’, level=logging.INFO)
 
data = [] 

with gzip.open ('reviews_data.txt.gz', 'rb') as f:
        for i,line in enumerate (f):
            data.append(line)
            

'''
path = get_tmpfile("word2vec.model")
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")


model = Word2Vec.load("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)

vector = model.wv['computer']

data = []
with open("./tone_data/anger/anger") as f:
	j=0
	for i in f:
		j+=1 
		data.append(i)
			#data_labels.append('anger')

with open("./tone_data/fear/fear") as f:
	j=0
	for i in f:
		j+=1 
		data.append(i)
			#data_labels.append('fear')

with open("./tone_data/love/love") as f:
	j=0
	for i in f:
		j+=1 
		data.append(i)
			#data_labels.append('love')

with open("./tone_data/sadness/sadness") as f:
	j=0
	for i in f: 
		j+=1
		data.append(i)
			#data_labels.append('sadness')

with open("./tone_data/joy/joy") as f:
	j=0
	for i in f:
		j+=1 
		data.append(i)
			#data_labels.append('joy')

with open("./tone_data/surprise/surprise") as f:
	j=0
	for i in f:
		j+=1 
		data.append(i)
			#data_labels.append('surprise')
'''
print(len(data))
shuffle(data)	
print("shuffled")
i = 0
processData = []
for line in data:
    print(i)
    i += 1
    processLine = simple_preprocess(line)
    processData.append(processLine)

# build vocabulary and train model
print("processing complete")
model = Word2Vec(
    processData,
    size=150,
    window=10,
    min_count=2,
    workers=10)
model.train(processData, total_examples=len(processData), epochs=10)
print("training complete")
 
w1 = "polite"
print(model.wv.most_similar(positive = w1))

w2 = "happy"
print(model.wv.most_similar(positive = w2))

pickle.dump(model, open("training_models/word2vec.pkl", "wb"))

'''
english_words = []
english_words = words.words()
print(len(english_words))

tf=TfidfVectorizer()

def vectorizer(data):
    
    tfidf_matrix=tf.fit_transform(english_words)
    pickle.dump(tf,open('training_models/vectorizer.joblib.pkl',"wb"), protocol=2)
    print('vectorizer saved')
    matrix=tfidf_matrix.toarray()
    print(english_words[0])
    print(matrix[0])
    return matrix

vectorizer(english_words)'''