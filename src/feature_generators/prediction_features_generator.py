import pickle
from functools import reduce
from time import time

import numpy as np
import pandas as pd
import gensim
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from helpers import *

model = gensim.models.KeyedVectors.load_word2vec_format('datasets/word2vec.txt')
print ('GloVe model loaded!')


def CountFeatureGenerator(df):
    """
    Counts occurences of predetermined words in body-content of news.
    
    Input: Datafram
    Returns: Count feature vector.
    """

    t0 = time()
    print("\n---Generating Counting Features:---")

    grams = ["unigram", "bigram", "trigram"]
    feat_names = ["Headline", "articleBody"]
    for feat_name in feat_names:
        for gram in grams:
            df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
            df["count_of_unique_%s_%s" % (feat_name, gram)] = \
                list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
            df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                list(map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)]))

    # overlapping n-grams count
    for gram in grams:
        df["count_of_Headline_%s_in_articleBody" % gram] = \
            list(df.apply(lambda x: sum([1. for w in x["Headline_" + gram] if w in set(x["articleBody_" + gram])]), axis=1))
        df["ratio_of_Headline_%s_in_articleBody" % gram] = \
            list(map(try_divide, df["count_of_Headline_%s_in_articleBody" % gram], df["count_of_Headline_%s" % gram]))
        
    # number of sentences in headline and body
    for feat_name in feat_names:
        df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))

    # dump the basic counting features into a file
    feat_names = [ n for n in df.columns \
                  if "count" in n \
                  or "ratio" in n \
                  or "len_sent" in n]
    
    # binary refuting features
    _refuting_words = [
        'fake', 'fraud', 'hoax', 'false', 'deny', 'denies', # 'refute',
        'not', 'despite', 'nope', 'doubt', 'doubts', 'bogus', 'debunk',
        'pranks', 'retract'
    ]

    _hedging_seed_words = [
        'alleged', 'allegedly', 'apparently', 'appear', 'appears', 'claim',
        'claims', 'could', 'evidently', 'largely', 'likely', 'mainly', 'may',
        'maybe', 'might', 'mostly', 'perhaps', 'presumably', 'probably',
        'purported', 'purportedly', 'reported', 'reportedly', 'rumor',
        'rumour', 'rumors', 'rumours', 'rumored', 'rumoured', 'says',
        'seem', 'somewhat', # 'supposedly',
        'unconfirmed'
    ]
        
    check_words = _refuting_words
    for rf in check_words:
        fname = '%s_exist' % rf
        feat_names.append(fname)
        df[fname] = df['Headline'].map(lambda x: 1 if rf in x else 0)   
    xBasicCounts = df[feat_names].values
    print ('xBasicCounts.shape: ', xBasicCounts.shape)

    print ('---Counting Features is complete---')
    print("Time taken {} seconds\n".format(time() - t0))
    return [xBasicCounts]


def TfidfFeatureGenerator(df):
    """
    Forms Term-Frequency Inverse Document-Fequency matrix for head and body
    to compute cosine similarity between them.
    
    Input: DataFrame
    Returns TFIDF cosine feature list.
    """

    t0 = time()
    print("\n---Generating TFIDF Features:---")

    # Fit a TfidfVectorizer on the concatenated strings
    with open('saved_data/vec.pkl', 'rb') as vocab:
        vec = pickle.load(vocab)
    
    # Tfidf for Headline
    xHeadlineTfidf = vec.transform(df['Headline_unigram'].map(lambda x: ' '.join(x)))
    print ('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)
    ## Only for SVD
    #outfilename_htfidf = "headline.tfidf.pkl"
    #with open("tmp/" + outfilename_htfidf, "wb") as outfile:
    #    pickle.dump(xHeadlineTfidf, outfile, -1)

    # Tfidf for articleBody
    xBodyTfidf = vec.transform(df['articleBody_unigram'].map(lambda x: ' '.join(x)))
    print ('xBodyTfidf.shape: ', xBodyTfidf.shape)
    ## Only for SVD
    #outfilename_btfidf = "body.tfidf.pkl"
    #with open("tmp/" + outfilename_btfidf, "wb") as outfile:
    #    pickle.dump(xBodyTfidf, outfile, -1)
    
    # Compute cosine similarity between headline tfidf features and body tfidf features
    simTfidf = np.asarray(list(map(cosine_sim, xHeadlineTfidf, xBodyTfidf)))[:, np.newaxis]
    print ('simTfidf.shape: ', simTfidf.shape)

    print('---TFIDF Features is complete---')
    print("Time taken {} seconds\n".format(time() - t0))
    return [simTfidf.reshape(-1, 1)]


def SvdFeatureGenerator(df):
    """
    Reduces dimensions of the TFIDF head and body matrix and computes
    cosine similarity between them.
    
    Input: DataFrame
    Returns list(headsvd_mat, bodysvd_mat, simsvd_mat)
    """

    t0 = time()
    print("\n---Generating SVD Features:---")

    with open("tmp/" + "headline.tfidf.pkl", "rb") as infile:
        xHeadlineTfidf = pickle.load(infile)
    with open("tmp/" + "body.tfidf.pkl", "rb") as infile:
        xBodyTfidf = pickle.load(infile)
    
    # compute the cosine similarity between truncated-svd features
    svd = TruncatedSVD(n_components=50, n_iter=15)
    xHBTfidf = vstack([xHeadlineTfidf, xBodyTfidf])
    svd.fit(xHBTfidf) # fit to the combined train-test set (or the full training set for cv process)
    
    # For Headline
    xHeadlineSvd = svd.transform(xHeadlineTfidf)
    print ('xHeadlineSvd.shape: ', xHeadlineSvd.shape)
    
    # For Body
    xBodySvd = svd.transform(xBodyTfidf)
    print ('xBodySvd.shape: ', xBodySvd.shape)
    
    # Compute cosine similarity
    simSvd = np.asarray(list(map(cosine_sim, xHeadlineSvd, xBodySvd)))[:, np.newaxis]
    print ('simSvd.shape:', simSvd.shape)

    print("---SVD Features is complete---")
    print("Time taken {} seconds\n".format(time() - t0))
    return [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]


def Word2VecFeatureGenerator(df):
    """
    Finds and returns word embedding for the head and body
    and computes cosine similarity.
    
    Input: DataFrame
    Returns list(headlineVec, bodyVec, simVec)"""

    t0 = time()
    print("\n---Generating Word2Vector Features:---")

    df["Headline_unigram_vec"] = df["Headline"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
    df["articleBody_unigram_vec"] = df["articleBody"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))

    
    # Document vector built by multiplying together all the word vectors
    # using Google's pre-trained word vectors
    Headline_unigram_array = df['Headline_unigram_vec'].values
    
    # word vectors weighted by normalized tf-idf coefficient?
    headlineVec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*50), Headline_unigram_array))
    headlineVec = np.array(headlineVec)

    #headlineVec = np.exp(headlineVec)
    headlineVec = normalize(headlineVec)
    print ('headlineVec.shape: ', headlineVec.shape)
    
    Body_unigram_array = df['articleBody_unigram_vec'].values
    bodyVec = list(map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*50), Body_unigram_array))
    bodyVec = np.array(bodyVec)
    bodyVec = normalize(bodyVec)
    print ('bodyVec.shape: ', bodyVec.shape)

    # compute cosine similarity between headline/body word2vec features
    simVec = np.asarray(list(map(cosine_sim, headlineVec, bodyVec)))[:, np.newaxis]
    print ('simVec.shape: ', simVec.shape)

    print("---Word2Vector Features is complete---")
    print("Time taken {} seconds\n".format(time() - t0))
    return [headlineVec, bodyVec, simVec]


def SentimentFeatureGenerator(df):
    """
    Generates Sentiment Intensity score for headline and body.
    
    Input: DataFrame
    Returns list(headlineSenti, bodySenti)"""

    t0 = time()
    print("\n---Generating Sentiment Features:---")

    # calculate the polarity score of each sentence then take the average
    sid = SentimentIntensityAnalyzer()
    def compute_sentiment(sentences):
        result = []
        for sentence in sentences:
            vs = sid.polarity_scores(sentence)
            result.append(vs)
        return pd.DataFrame(result).mean()
    
    df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x))
    df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
    df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)

    headlineSenti = df[['h_compound','h_neg','h_neu','h_pos']].values
    print ('headlineSenti.shape: ', headlineSenti.shape)

    df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
    df = pd.concat([df, df['body_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
    df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)

    bodySenti = df[['b_compound','b_neg','b_neu','b_pos']].values
    print ('bodySenti.shape: ', bodySenti.shape)

    print("---Sentiment Features is complete---")
    print("Time taken {} seconds\n".format(time() - t0))
    return [headlineSenti, bodySenti]


def ReadabilityFeatureGenerator(df):
    """
    Computes various readability features of news content.

    Input: DataFrame
    Returns list of readability features
    """

    t0 = time()
    print("\n---Generating Readability Features:---")

    def lexical_diversity(text):
        word_count = len(text)
        vocab_size = len(set(text))
        diversity_score = word_count / vocab_size
        return diversity_score

    def get_counts(text, word_list):
        words = nltk.tokenize.word_tokenize(text.lower())
        count = 0
        for word in words:
            if word in word_list:
                count += 1
        return count

    df['flesch_reading_ease'] = df['articleBody'].map(lambda x: textstat.flesch_reading_ease(x))
    df['smog_index'] = df['articleBody'].map(lambda x: textstat.smog_index(x))
    df['flesch_kincaid_grade'] = df['articleBody'].map(lambda x: textstat.flesch_kincaid_grade(x))
    df['coleman_liau_index'] = df['articleBody'].map(lambda x: textstat.coleman_liau_index(x))
    df['automated_readability_index'] = df['articleBody'].map(lambda x: textstat.automated_readability_index(x))
    df['dale_chall_readability_score'] = df['articleBody'].map(lambda x: textstat.dale_chall_readability_score(x))
    df['difficult_words'] = df['articleBody'].map(lambda x: textstat.difficult_words(x))
    df['linsear_write_formula'] = df['articleBody'].map(lambda x: textstat.linsear_write_formula(x))
    df['gunning_fog'] = df['articleBody'].map(lambda x: textstat.gunning_fog(x))
    df['i_me_myself'] = df['articleBody'].apply(get_counts,args = (['i', 'me', 'myself'],))
    df['punct'] = df['articleBody'].apply(get_counts,args = ([',','.', '!', '?'],))
    df['lexical_diversity'] = df['articleBody'].apply(lexical_diversity)

    feats = ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade',
    'coleman_liau_index', 'automated_readability_index', 
    'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula',
    'gunning_fog', 'i_me_myself', 'punct', 'lexical_diversity'
    ]

    xReadable = df[feats].values
    print ('xReadable.shape: ', xReadable.shape)

    print('---Readability Features is complete---')
    print("Time taken {} seconds\n".format(time() - t0))
    return [xReadable]
