#Data Corpus
import pickle as cp
import json
import re

data_list=[]

def sentence_to_wordlist(sentence):
    sentence_text = re.sub('[!।,$#-?.:%\/&()]','', sentence)
    words = sentence_text.split()
    return(words)

f1_size=0
f2_size=0
f3_size=0
k=0

alltxt = cp.load(open('/home/saklain/Downloads/alltxt.pkl','rb'))
for i in alltxt:
        data_list.append(sentence_to_wordlist(i))
        
print(len(alltxt))

with open('/home/saklain/Downloads/prothomaloComments.json','r') as f1:
        
        for i in f1:
            data2 = json.loads(i)
            f1_size+=1
            data_list.append(sentence_to_wordlist(data2["comment"]))
print(f1_size)

with open('/home/saklain/Downloads/scraped_prothomAloArticles.json','r') as f2:
        for i in f2:
            data2 = json.loads(i)
            f2_size+=1
            data_list.append(sentence_to_wordlist(data2["article"]))
print(f2_size)

print(len(alltxt)+f1_size+f2_size)


###########################################
###########################################

import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import Phrases
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


num_features = 300    # Word vector dimensionality                      
min_word_count = 5                          
num_workers = 4       # Number of threads
window_size = 5                                                                                          

model = word2vec.Word2Vec(data_list, workers=num_workers, 
            size=num_features, min_count = min_word_count, 
            window = window_size)

model_name = "banglaData"
model.save(model_name)
new_model = gensim.models.Word2Vec.load('banglaData')

###########################################
###########################################

import gensim
new_model = gensim.models.Word2Vec.load('banglaData')
vocab = list(new_model.wv.vocab.keys())

new_model.most_similar('নদী',  topn=15)
new_model.most_similar(positive=['দেশ'], negative=['রাজনীতি'], topn=15)
'লসিকা' in new_model.wv.vocab
new_model.wv.doesnt_match("রাত ফুটবল বিকাল সকাল".split())
print(new_model.wv.most_similar(positive=["ফিলিস্তিন","বাংলাদেশ"], negative=['ইসরাইল']))
print(new_model.wv.similarity("আমি","ভাত"))
new_model.wv.most_similar(positive=['পানি', 'মাটি'],negative=['খাবার'])
context = "আমাদের দেশের ".split()
print(context)
predicted_word = new_model.predict_output_word(context, topn = 10)
print(predicted_word)
out_tup = [word for word in predicted_word if word[0] not in context]
out_tup

###########################################
###########################################

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

from sklearn.cluster import KMeans
import time

model = gensim.models.KeyedVectors.load('banglaData')

start = time.time() 

vectors = model.wv.syn0
word_vectors = vectors[0:50000]
#print(word_vectors)
num_clusters = int(1000)

kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

end = time.time()
elapsed = end - start
print ("Time taken : ", elapsed, "seconds.")

###########################################
###########################################

word_centroid_map = dict(zip( model.wv.index2word, idx ))

for cluster in range(0,1000):
    print ("Cluster: ", cluster)
    
    words = []
    for key, item in word_centroid_map.items():
        if item == cluster:
            words.append(key)
    print (words)
    
###########################################    


