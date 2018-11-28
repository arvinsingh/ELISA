# ELISA - For FYP

## Overview

The model takes two string inputs: Headline and article Body and feeds it to Boosted tree for prediction of article in (Reliable or Unreliable) class.

## Feature Engineering

**1. Preprocessing (`gen_features.ipynb`)**

The labels (`reliable`, `unreliable`) are encoded into numeric target values as (`0`, `1`). The text of headline and body are then tokenized and stemmed (by `preprocess_data()` in `helpers.py`). Finally Uni-grams, bi-grams and tri-grams are created out of the list of tokens. These grams and the original text are used by the following feature extractor modules.

**2. Basic Count Features (`CountFeatureGenerator.py`)**

This module takes the uni-grams, bi-grams and tri-grams and creates various counts and ratios which could potentially signify how a body text is related to a headline. Specifically, it counts how many times a gram appears in the headline, how many unique grams there are in the headline, and the ratio between the two. The same statistics are computed for the body text, too. It then calculates how many grams in the headline also appear in the body text, and a normalized version of this overlapping count by the number of grams in the headline. The results are saved in the pickle file which will be read back in by the classifier.

**3. [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) Features (`TfidfFeatureGenerator.py`)**

This module constructs sparse vector representations of the headline and body by calculating the Term-Frequency of each gram and normalize it by its Inverse-Document Frequency. First off a `TfidfVectorizer` is fit to the concatenations of headline and body text to obtain the vocabulary. Then using the same vocabulary it separately fits and transforms the headline grams and body grams into sparse vectors. It also calculates the cosine similarity between the headline vector and the body vector.

**4. Readability Features (`ReadabilityFeatureGenerator.py`)**

This module takes Textstat Python package to calculate statistics from text to determine readability, complexity and grade level of a particular corpus.

**5. Word2Vec Features (`Word2VecFeatureGenerator.py`)**

This module utilizes the pre-trained [word vectors](https://arxiv.org/abs/1301.3781) from public sources, add them up to build vector representations of the headline and body. The word vectors were trained on a Google News corpus with 100 billion words and a vocabulary size of 3 million. The resulting word vectors can be used to find synonyms, predict the next word given the previous words, or to manipulate semantics. For example, when you calculate `vector(Germany) - Vector(Berlin) + Vector(England)` you will obtain a vector that is very close to `Vector(London)`. For the current problem constructing the vector representation out of word vectors could potentially overcome the ambiguities introduced by the fact that headline and body may use synonyms instead of exact words.

**6. Sentiment Features (`SentimentFeatureGenerator.py`)**

This modules uses the Sentiment Analyzer in the `NLTK` package to assign a sentiment polarity score to the headline and body separately. For example, negative score means the text shows a negative opinion of something. This score can be informative of whether the body is being positive about a subject while the headline is being negative. But it does not indicate whether it's the same subject that appears in the body and headline; however, this piece of information should be preserved in other features.

## Library Dependencies
* Python <= 3.5
* Scipy Stack (`numpy`, `scipy` and `pandas`)
* [scikit-learn](http://scikit-learn.org/stable/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/)
* [gensim (for word2vec)](https://radimrehurek.com/gensim/)
* [NLTK (python NLP library)](http://www.nltk.org)

## Procedure
**1. Install all the dependencies**

**2.`clone the repo`**

**3. Download the `GloVe` [model](http://nlp.stanford.edu/data/glove.6B.zip) trained on Wikipedia 2014 + Gigaword 5. Convert the file to word2vec.txt using `convert_GloVe2Word2Vec.ipynb` and save under `datasets/`.**

**4. Use `prepare_data.ipynb` then `gen_features.ipynb` to generate all the required features.**

**All the pickled files are saved under `saved_data/`.**

**6. Run `xgb_train` to train and make predictions on the test set. Output file is `predictions_*.csv`**

**7. Use `Result_visualization.ipynb` and `test_xgb_model.ipynb` to study the output and use the model respectively.**

All the output files are also stored under `results/` and all parameters are hard-coded. 
