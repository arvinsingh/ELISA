import pickle
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb

import ngram as ngram
from helpers import *
from prediction_features_generator import *

LABELS = ['reliable', 'unreliable']


# Load the saved booster model
with open('saved_data/xgb_model.pkl', 'rb') as mod:
    xgb_mod = pickle.load(mod)

print ('Booster model loaded!')


def process(data):
    """
    Preprocesses the data provided and generates
    unigrams, bigrams and trigrams.
    Saves the features in the separate columns in the dataframe.
    
    Input: Dataframe
    
    Returns Dataframe
    """
    
    print (data.iloc[0])
    print ('>>> Data shape: ', data.shape)
    
    t0 = time()
    print("---Generating n-grams Features!---")
    print ("Generating unigram")
    data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))
    data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))
    
    print ("Generating bigram")
    join_str = "_"
    data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: ngram.getBigram(x, join_str))
    data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: ngram.getBigram(x, join_str))

    print ("Generating trigram")
    join_str = "_"
    data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
    data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: ngram.getTrigram(x, join_str))

    print("---n-gram Features generated---")
    print("Time taken: {} seconds\n".format(time() - t0))
    
    return data


def build_test_data(data):
    """
    Generates various features needed to predict
    the class of the news.
    
    Input: DataFrame
    Returns Array of generated features.
    """

    data = process(data)
    
    generators = [
                  CountFeatureGenerator,
                  TfidfFeatureGenerator,
                  Word2VecFeatureGenerator,
                  SentimentFeatureGenerator,
                  ReadabilityFeatureGenerator
                 ]
    
    # Class generators one by one to generate features
    features = [feature for generator in generators for feature in generator(data)]
    print("Total number of raw features: {}".format(len(features)))
    
    # Stack and return the features
    return np.hstack(features)


def check(headline, body):
    """
    Predicts the probable class and corresponding probabilites
    of the news belonging to a certian clas
    
    Input: Headline and Article body string
    Returns list(predicted_class, reliability_score, unreliability_score)
    """

    news = pd.DataFrame({'Headline': headline, 'articleBody': body}, [0])
    test_x = build_test_data(news)

    # Build DMatrix for booster
    dtest = xgb.DMatrix(test_x)
    print("Total Feature count: ", len(dtest.feature_names))

    
    
    # Use Booster to predict class
    pred_prob_y = xgb_mod.predict(dtest).reshape(test_x.shape[0], 4) # predicted probabilities
    pred_y = np.argmax(pred_prob_y, axis=1)

    # print (predicted)
    print ('pred_y.shape: ', pred_y.shape)
    predicted = [LABELS[int(a)] for a in pred_y]

    news['preds'] = predicted
    news['Reliable'] = pred_prob_y[:, 0]
    news['Unreliable'] = pred_prob_y[:, 1]

    # news.to_csv('../results/tree_pred_prob.csv', index=False)
    return [news['preds'][0], news['Reliable'][0], news['Unreliable'][0]]
