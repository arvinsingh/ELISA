from . import ngram
import pandas as pd
import numpy as np
import pickle
from .helpers import *
from .CountFeatureGenerator import *
from .TfidfFeatureGenerator import *
from .SvdFeatureGenerator import *
from .Word2VecFeatureGenerator import *
from .SentimentFeatureGenerator import *
#from AlignmentFeatureGenerator import *

def process():

    read = False
    if not read:
    
        body_train = pd.read_csv("../datasets/train_bodies_processed.csv", encoding='utf-8')
        stances_train = pd.read_csv("../datasets/train_stances_processed.csv", encoding='utf-8')
        # training set
        train = pd.merge(stances_train, body_train, how='left', on='Body ID')
        targets = ['agree', 'disagree', 'discuss', 'unrelated']
        targets_dict = dict(zip(targets, range(len(targets))))
        train['target'] = list(map(lambda x: targets_dict[x], train['Stance']))
        print ('train.shape:')
        print (train.shape)
        n_train = train.shape[0]

        data = train
        # read test set, no 'Stance' column in test set -> target = NULL
        # concatenate training and test set
        test_flag = True
        if test_flag:
            body_test = pd.read_csv("../datasets/test_bodies_processed.csv", encoding='utf-8')
            headline_test = pd.read_csv("../datasets/test_stances_unlabeled.csv", encoding='utf-8')
            test = pd.merge(headline_test, body_test, how="left", on="Body ID")
            
            data = pd.concat((train, test), sort=False) # target = NaN for test set
            print (data)
            print ('data.shape:')
            print (data.shape)

            train = data[~data['target'].isnull()]
            print (train)
            print ('train.shape:')
            print (train.shape)
            
            test = data[data['target'].isnull()]
            print (test)
            print ('test.shape:')
            print (test.shape)

        #data = data.iloc[:100, :]
        
        #return 1
        
        print ("generate unigram")
        data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))
        data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))

        print ("generate bigram")
        join_str = "_"
        data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: ngram.getBigram(x, join_str))
        data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: ngram.getBigram(x, join_str))
        
        print ("generate trigram")
        join_str = "_"
        data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
        data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
        
        with open('saved_data/data.pkl', 'wb') as outfile:
            pickle.dump(data, outfile, -1)
            print('dataframe saved in data.pkl')

    else:
        with open('saved_data/data.pkl', 'rb') as infile:
            data = pickle.load(infile)
            print ('data loaded')
            print ('data.shape:')
            print (data.shape)
    #return 1

    # define feature generators
    countFG    = CountFeatureGenerator()
    tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG    = SentimentFeatureGenerator()
    #walignFG   = AlignmentFeatureGenerator()
    generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    #generators = [svdFG, word2vecFG, sentiFG]
    #generators = [tfidfFG]
    #generators = [countFG]
    #generators = [walignFG]
    
    for g in generators:
        g.process(data)
    
    for g in generators:
        g.read('train')
    
    #for g in generators:
    #    g.read('test')

    print ('done')


if __name__ == "__main__":

    process()