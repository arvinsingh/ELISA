from FeatureGenerator import *
import ngram
import pickle
import pandas as pd
from time import time
from nltk.tokenize import sent_tokenize
from helpers import *
import hashlib
import nltk
import textstat


class ReadabilityFeatureGenerator(FeatureGenerator):
    """
    Readability is the ease with which a reader can understand
    a written text. In natural language, the readability of text
    depends on its content (the complexity of its vocabulary 
    and syntax) and its presentation
    """


    def __init__(self, name='readabilityFeatureGenerator'):
        super(ReadabilityFeatureGenerator, self).__init__(name)


    def process(self, df):

        t0 = time()
        print("\n---Generating Readability Features:---\n")

        def lexical_diversity(text):
            word_count = len(nltk.tokenize.word_tokenize(text))
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


        outfilename_xReadable = df[feats].values

        with open('../saved_data/read.pkl', 'wb') as outfile:
            pickle.dump(feats, outfile, -1)
            pickle.dump(outfilename_xReadable, outfile, -1)

        print ('readable features saved in read.pkl')
        
        print('\n---Readability Features is complete---')
        print("Time taken {} seconds\n".format(time() - t0))
        
        return 1


    def read(self):

        filename_rf = 'read.pkl'
        with open("../saved_data/" + filename_rf, "rb") as infile:
            _ = pickle.load(infile)
            xReadable = pickle.load(infile)
        print ('xReadable.shape: ', xReadable.shape)

        return [xReadable]

