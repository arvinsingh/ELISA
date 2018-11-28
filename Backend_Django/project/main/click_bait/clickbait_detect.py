from keras.models import load_model
import re
import string
from keras.preprocessing import sequence
import numpy as np
 

vocabulary = open("media/ClickBait/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))
print('-----> Click Bait vocabulary loaded!!!')
model_click_bait = load_model('media/ClickBait/fine.h5')
print('-----> Click Bait model loaded!!!')


def predict(headline):
    def clean(text):
        MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
        text = text.lower()
        for punctuation in string.punctuation:
            text = text.replace(punctuation, " " + punctuation + " ")
        for i in range(10):
            text = text.replace(str(i), " " + str(i) + " ")
        text = MATCH_MULTIPLE_SPACES.sub(" ", text)
        return "\n".join(line.strip() for line in text.split("\n"))


    def words_to_indices(inverse_vocabulary, words):
        UNK = "<UNK>"
        return [inverse_vocabulary.get(word, inverse_vocabulary[UNK]) for word in words]


    SEQUENCE_LENGTH = 20
    EMBEDDING_DIMENSION = 30
    PAD = "<PAD>"        
    inputs = sequence.pad_sequences([words_to_indices(inverse_vocabulary, clean(str(headline)).lower().split())], maxlen=SEQUENCE_LENGTH)
    clickbaitiness = model_click_bait.predict(inputs)[0, 0]
    return clickbaitiness

predict("test")
