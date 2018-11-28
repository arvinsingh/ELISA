import spacy
from pyspark.sql import SparkSession
import gensim
import pickle
import time


def setup():
    t = time.time()
    
    # load spacy model for feature generation
    nlp = spacy.load('en_core_web_md')
    print("-----> Spacy model loaded!!!")

    model = gensim.models.KeyedVectors.load_word2vec_format('media/FND/word2vec.txt')
    print ("-----> GloVe model loaded!!!")
    
    print("\nTime required to load : ", "{:.2f}".format(time.time()-t), " seconds.")
    return nlp, model