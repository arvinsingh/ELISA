from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers import *


class TfidfFeatureGenerator(FeatureGenerator):
    
    
    def __init__(self, name='tfidfFeatureGenerator'):
        super(TfidfFeatureGenerator, self).__init__(name)

    
    def process(self, df):

        # 1). create strings based on ' '.join(Headline_unigram + articleBody_unigram) [ already stemmed ]
        def cat_text(x):
            res = '%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res
        df["all_text"] = list(df.apply(cat_text, axis=1))
        n_train = df[~df['target'].isnull()].shape[0]
        print ('tfidf, n_train:',n_train)
        n_test = df[df['target'].isnull()].shape[0]
        print ('tfidf, n_test:',n_test)

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["all_text"]) # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xHeadlineTfidf = vecH.fit_transform(df['Headline_unigram'].map(lambda x: ' '.join(x))) # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        print ('xHeadlineTfidf.shape:')
        print (xHeadlineTfidf.shape)
        
        # save train and test into separate files
        xHeadlineTfidfTrain = xHeadlineTfidf[:n_train, :]
        outfilename_htfidf_train = "train.headline.tfidf.pkl"
        with open("../saved_data/" + outfilename_htfidf_train, "wb") as outfile:
            pickle.dump(xHeadlineTfidfTrain, outfile, -1)
        print ('headline tfidf features of training set saved in %s' % outfilename_htfidf_train)
        
        if n_test > 0:
            # test set is available
            xHeadlineTfidfTest = xHeadlineTfidf[n_train:, :]
            outfilename_htfidf_test = "test.headline.tfidf.pkl"
            with open("../saved_data/" + outfilename_htfidf_test, "wb") as outfile:
                pickle.dump(xHeadlineTfidfTest, outfile, -1)
            print ('headline tfidf features of test set saved in %s' % outfilename_htfidf_test)


        vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xBodyTfidf = vecB.fit_transform(df['articleBody_unigram'].map(lambda x: ' '.join(x)))
        print ('xBodyTfidf.shape:')
        print (xBodyTfidf.shape)
        
        # save train and test into separate files
        xBodyTfidfTrain = xBodyTfidf[:n_train, :]
        outfilename_btfidf_train = "train.body.tfidf.pkl"
        with open("../saved_data/" + outfilename_btfidf_train, "wb") as outfile:
            pickle.dump(xBodyTfidfTrain, outfile, -1)
        print ('body tfidf features of training set saved in %s' % outfilename_btfidf_train)
        
        if n_test > 0:
            # test set is availble
            xBodyTfidfTest = xBodyTfidf[n_train:, :]
            outfilename_btfidf_test = "test.body.tfidf.pkl"
            with open("../saved_data/" + outfilename_btfidf_test, "wb") as outfile:
                pickle.dump(xBodyTfidfTest, outfile, -1)
            print ('body tfidf features of test set saved in %s' % outfilename_btfidf_test)
               

        # 4). compute cosine similarity between headline tfidf features and body tfidf features
        simTfidf = np.asarray(list(map(cosine_sim, xHeadlineTfidf, xBodyTfidf)))[:, np.newaxis]
        print ('simTfidf.shape:')
        print (simTfidf.shape)
        simTfidfTrain = simTfidf[:n_train]
        outfilename_simtfidf_train = "train.sim.tfidf.pkl"
        with open("../saved_data/" + outfilename_simtfidf_train, "wb") as outfile:
            pickle.dump(simTfidfTrain, outfile, -1)
        print ('tfidf sim. features of training set saved in %s' % outfilename_simtfidf_train)
        
        if n_test > 0:
            # test set is available
            simTfidfTest = simTfidf[n_train:]
            outfilename_simtfidf_test = "test.sim.tfidf.pkl"
            with open("../saved_data/" + outfilename_simtfidf_test, "wb") as outfile:
                pickle.dump(simTfidfTest, outfile, -1)
            print ('tfidf sim. features of test set saved in %s' % outfilename_simtfidf_test)

        return 1


    def read(self, header='train'):

        filename_htfidf = "%s.headline.tfidf.pkl" % header
        with open("../saved_data/" + filename_htfidf, "rb") as infile:
            xHeadlineTfidf = pickle.load(infile)

        filename_btfidf = "%s.body.tfidf.pkl" % header
        with open("../saved_data/" + filename_btfidf, "rb") as infile:
            xBodyTfidf = pickle.load(infile)

        filename_simtfidf = "%s.sim.tfidf.pkl" % header
        with open("../saved_data/" + filename_simtfidf, "rb") as infile:
            simTfidf = pickle.load(infile)

        print ('xHeadlineTfidf.shape:')
        print (xHeadlineTfidf.shape)
        #print (type(xHeadlineTfidf))
        print ('xBodyTfidf.shape:')
        print (xBodyTfidf.shape)
        #print (type(xBodyTfidf))
        print ('simTfidf.shape:')
        print (simTfidf.shape)
        #print (type(simTfidf))

        return [xHeadlineTfidf, xBodyTfidf, simTfidf.reshape(-1, 1)]
        #return [simTfidf.reshape(-1, 1)]
