from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers import *


class TfidfFeatureGenerator(FeatureGenerator):
    """
    This module constructs sparse vector representations of 
    the headline and body by calculating the Term-Frequency 
    of each gram and normalize it by its Inverse-Document 
    Frequency. First off a TfidfVectorizer is fit to the 
    concatenations of headline and body text to obtain the 
    vocabulary. Then using the same vocabulary it separately 
    fits and transforms the headline grams and body grams into 
    sparse vectors. It also calculates the cosine similarity 
    between the headline vector and the body vector.

    Line 141: Raw TF-IDF vectors are needed by SvdFeatureGenerator.py
              during feature generation

    Line 142: But only the similarities are needed for training.
    """    
    
    def __init__(self, name='tfidfFeatureGenerator'):
        super(TfidfFeatureGenerator, self).__init__(name)

    
    def process(self, df):

        t0 = time()
        print("\n---Generating TFIDF Features:---\n")

        # 1). create strings based on ' '.join(Headline_unigram + articleBody_unigram) [ already stemmed ]
        def cat_text(x):
            res = '%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res

        df["all_text"] = list(df.apply(cat_text, axis=1))

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["all_text"]) # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        print("\nlength of vocabulary: " + str(len(vocabulary)))
        with open('../saved_data/vec.pkl', 'wb') as vo:
            pickle.dump(vec, vo)
        print("Vocabulary vector saved!\n")

        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        xHeadlineTfidf = vecH.fit_transform(df['Headline_unigram'].map(lambda x: ' '.join(x)))
        print ('xHeadlineTfidf.shape: ', xHeadlineTfidf.shape)


        vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xBodyTfidf = vecB.fit_transform(df['articleBody_unigram'].map(lambda x: ' '.join(x)))
        print ('xBodyTfidf.shape: ', xBodyTfidf.shape)
        

        # 4). compute cosine similarity between headline tfidf features and body tfidf features
        simTfidf = np.asarray(list(map(cosine_sim, xHeadlineTfidf, xBodyTfidf)))[:, np.newaxis]
        print ('simTfidf.shape: ', simTfidf.shape)

        outfilename_simtfidf = "sim.tfidf.pkl"
        with open("../saved_data/" + outfilename_simtfidf, "wb") as outfile:
            pickle.dump(simTfidf, outfile, -1)
        print ('tfidf similarities features saved in %s' % outfilename_simtfidf)
        print('\n---TFIDF Features is complete---')
        print("Time taken {} seconds\n".format(time() - t0))
        
        return 1


    def read(self):


        filename_simtfidf = "sim.tfidf.pkl"
        with open("../saved_data/" + filename_simtfidf, "rb") as infile:
            simTfidf = pickle.load(infile)

        print ('simTfidf.shape: ', simTfidf.shape)

        return [simTfidf.reshape(-1, 1)]
