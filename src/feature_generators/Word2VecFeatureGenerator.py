from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
import gensim
from time import time
from sklearn.preprocessing import normalize
from functools import reduce
from helpers import *


class Word2VecFeatureGenerator(FeatureGenerator):
    """
    This module utilizes the pre-trained word vectors from public 
    sources, add them up to build vector representations of the 
    headline and body. The word vectors were trained on a Google 
    News corpus with 100 billion words and a vocabulary size of 
    3 million. The resulting word vectors can be used to find 
    synonyms, predict the next word given the previous words, 
    or to manipulate semantics. For example, when you calculate 
    vector(Germany) - Vector(Berlin) + Vector(England) you will 
    obtain a vector that is very close to Vector(London). For the 
    current problem constructing the vector representation out of 
    word vectors could potentially overcome the ambiguities introduced 
    by the fact that headline and body may use synonyms instead 
    of exact words.
    """

    def __init__(self, name='word2vecFeatureGenerator'):
        super(Word2VecFeatureGenerator, self).__init__(name)


    def process(self, df):

        t0 = time()
        print("\n---Generating Word2Vector Features:---\n")

        df["Headline_unigram_vec"] = df["Headline"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        df["articleBody_unigram_vec"] = df["articleBody"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        
        # 1). document vector built by multiplying together all the word vectors
        # using Google's pre-trained word vectors
        # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        model = gensim.models.KeyedVectors.load_word2vec_format('../datasets/word2vec.txt')
        print ('model loaded')

        Headline_unigram_array = df['Headline_unigram_vec'].values
        
        # word vectors weighted by normalized tf-idf coefficient?
        #headlineVec = [0]
        headlineVec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*50), Headline_unigram_array))
        headlineVec = np.array(headlineVec)
        headlineVec = normalize(headlineVec)
        print ('headlineVec.shape', headlineVec.shape)

        outfilename_hvec = "headline.word2vec.pkl"
        with open("../saved_data/" + outfilename_hvec, "wb") as outfile:
            pickle.dump(headlineVec, outfile, -1)
        print ('headline word2vec features saved in %s' % outfilename_hvec)

        print ('headine done')

        Body_unigram_array = df['articleBody_unigram_vec'].values

        #bodyVec = [0]
        bodyVec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*50), Body_unigram_array))
        bodyVec = np.array(bodyVec)
        bodyVec = normalize(bodyVec)
        print ('bodyVec.shape: ', bodyVec.shape)

        outfilename_bvec = "body.word2vec.pkl"
        with open("../saved_data/" + outfilename_bvec, "wb") as outfile:
            pickle.dump(bodyVec, outfile, -1)
        print ('body word2vec features saved in %s' % outfilename_bvec)

        print ('body done')

        # compute cosine similarity between headline/body word2vec features
        simVec = np.asarray(list(map(cosine_sim, headlineVec, bodyVec)))[:, np.newaxis]
        print ('simVec.shape:', simVec.shape)


        outfilename_simvec = "sim.word2vec.pkl"
        with open("../saved_data/" + outfilename_simvec, "wb") as outfile:
            pickle.dump(simVec, outfile, -1)
        print ('word2vec similarities features set saved in %s' % outfilename_simvec)

        print("\n---Word2Vector Features is complete---")
        print("Time taken {} seconds\n".format(time() - t0))
        
        return 1

    def read(self):

        filename_hvec = "headline.word2vec.pkl"
        with open("../saved_data/" + filename_hvec, "rb") as infile:
            headlineVec = pickle.load(infile)

        filename_bvec = "body.word2vec.pkl"
        with open("../saved_data/" + filename_bvec, "rb") as infile:
            bodyVec = pickle.load(infile)

        filename_simvec = "sim.word2vec.pkl"
        with open("../saved_data/" + filename_simvec, "rb") as infile:
            simVec = pickle.load(infile)

        print ('headlineVec.shape: ', headlineVec.shape)
        print ('bodyVec.shape: ', bodyVec.shape)
        print ('simVec.shape: ', simVec.shape)

        return [headlineVec, bodyVec, simVec]
        #return [simVec.reshape(-1,1)]
