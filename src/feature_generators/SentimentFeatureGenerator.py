from FeatureGenerator import *
import pandas as pd
import numpy as np
from time import time
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from helpers import *


class SentimentFeatureGenerator(FeatureGenerator):
    """
    This modules uses the Sentiment Analyzer in the NLTK package 
    to assign a sentiment polarity score to the headline and body 
    separately. For example, negative score means the text shows 
    a negative opinion of something. This score can be informative 
    of whether the body is being positive about a subject while the 
    headline is being negative. But it does not indicate whether it's 
    the same subject that appears in the body and headline; however, 
    this piece of information should be preserved in other features.
    """


    def __init__(self, name='sentimentFeatureGenerator'):
        super(SentimentFeatureGenerator, self).__init__(name)


    def process(self, df):

        t0 = time()
        print("\n---Generating Sentiment Features:---\n")

        print ('for headline')
        # calculate the polarity score of each sentence then take the average
        sid = SentimentIntensityAnalyzer()
        def compute_sentiment(sentences):
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()
        
        #df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x.decode('utf-8')))
        df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)

        headlineSenti = df[['h_compound','h_neg','h_neu','h_pos']].values
        print ('headlineSenti.shape:', headlineSenti.shape)

        
        outfilename_hsenti = "headline.senti.pkl"
        with open("../saved_data/" + outfilename_hsenti, "wb") as outfile:
            pickle.dump(headlineSenti, outfile, -1)
        print ('headline sentiment features saved in %s' % outfilename_hsenti)
        
        print ('headine sentiment done')
        
        #return 1

        print ('for body')
        df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['body_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)
        bodySenti = df[['b_compound','b_neg','b_neu','b_pos']].values
        print ('bodySenti.shape:', bodySenti.shape)


        outfilename_bsenti = "body.senti.pkl"
        with open("../saved_data/" + outfilename_bsenti, "wb") as outfile:
            pickle.dump(bodySenti, outfile, -1)
        print ('body sentiment features saved in %s' % outfilename_bsenti)

        print ('body senti done')

        print("\n---Sentiment Features is complete---")
        print("Time taken {} seconds\n".format(time() - t0))

        return 1


    def read(self):

        filename_hsenti = "headline.senti.pkl"
        with open("../saved_data/" + filename_hsenti, "rb") as infile:
            headlineSenti = pickle.load(infile)

        filename_bsenti = "body.senti.pkl"
        with open("../saved_data/" + filename_bsenti, "rb") as infile:
            bodySenti = pickle.load(infile)

        print ('headlineSenti.shape:', headlineSenti.shape)
        print ('bodySenti.shape: ', bodySenti.shape)

        return [headlineSenti, bodySenti]

