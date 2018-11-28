import time
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, train_test_split
from CountFeatureGenerator import CountFeatureGenerator
from TfidfFeatureGenerator import TfidfFeatureGenerator
from ReadabilityFeatureGenerator import ReadabilityFeatureGenerator
from Word2VecFeatureGenerator import Word2VecFeatureGenerator
from SentimentFeatureGenerator import SentimentFeatureGenerator
# from feature_generators import Features

from score import score_submission, report_score

# import warnings
# warnings.filterwarnings("ignore")

num_round = 1000
LABELS = ['reliable', 'unreliable']
params_xgb = {

    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    #'objective': 'multi:softmax',
    'objective': 'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class': 2
}

# Load data
def load_targets():
    # with open('../saved_data/targets.pkl', 'rb') as Ts:
    #    targets = pickle.load(Ts)
    targets = pd.read_csv('../datasets/targets.csv')
    return targets

# Load features
def load_features():

    generators = [
                  CountFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator(),
                  ReadabilityFeatureGenerator()
                  # Add more generators
                 ]

    features = [feature for generator in generators for feature in generator.read()]
    
    print("Total number of raw features: {}".format(len(features)))
    return np.hstack(features)


def fscore(pred_y, truth_y):
    
    # targets = ['reliable', 'unrelable']
    # y = [0, 1]
    score = 0
    if pred_y.shape != truth_y.shape:
        raise Exception('pred_y and truth have different shapes')
    for i in range(pred_y.shape[0]):
        if truth_y[i] == 3:
            if pred_y[i] == 3: score += 0.25
        else:
            if pred_y[i] != 3: score += 0.25
            if truth_y[i] == pred_y[i]: score += 0.75
    
    return score

def perfect_score(truth_y):
    
    score = 0
    for i in range(truth_y.shape[0]):
        if truth_y[i] == 3: score += 0.25
        else: score += 1

    return score

def eval_metric(yhat, dtrain):
    y = dtrain.get_label()
    yhat = np.argmax(yhat, axis=1)
    predicted = [LABELS[int(a)] for a in yhat]
    actual = [LABELS[int(a)] for a in y]
    s, _ = score_submission(actual, predicted)
    s_perf, _ = score_submission(actual, actual)
    score = float(s) / s_perf
    return 'score', score


# Train Model
def train():

    data = load_features().astype(float)
    targets = load_targets()
    X_train, X_test, y_train_, y_test_ = train_test_split(data, targets, test_size=0.20, random_state=42)

    y_train = y_train_['target'].values
    y_test = y_test_['target'].values

    
    n_iters = 500
    #n_iters = 50
    # perfect score on training set
    print ('perfect_score: ', perfect_score(y_train))
    print (Counter(y_train))

    dtrain = xgb.DMatrix(X_train, label=y_train)

    dtest = xgb.DMatrix(X_test)
    
    print("Total Feature count in train, test set: ", len(dtrain.feature_names))


    print("Shape of train is ", X_train.shape," and shape of test is ", X_test.shape)
    watchlist = [(dtrain, 'train')]
    print("----------Training XGBoost Model----------")
    t0 = time.time()
    bst = xgb.train(params_xgb, 
                    dtrain,
                    n_iters,
                    watchlist,
                    feval=eval_metric,
                    verbose_eval=10,
                    early_stopping_rounds=80)
    with open("../saved_data/xgb_model.pkl", 'wb') as mod:
        pickle.dump(bst, mod)
    print("----------Predicting labels----------")
    print("Model trained in: {} seconds".format(time.time()-t0))


    pred_prob_y = bst.predict(dtest).reshape(X_test.shape[0], 2) # predicted probabilities
    pred_y = np.argmax(pred_prob_y, axis=1)
    print ('pred_y.shape: ', pred_y.shape)
    predicted = [LABELS[int(a)] for a in pred_y]

    y_test_['preds'] = predicted
    y_test_['Reliable'] = pred_prob_y[:, 0]
    y_test_['Unreliable'] = pred_prob_y[:, 1]

    y_test_.to_csv('../results/predictions_early_80.csv', index=False)
    print("----Results saved in results/predictions_early_80.csv----")


# Validate Needs to be implimented correctly!
def cv():

    K_FOLD = 10
    data = load_features().astype(float)
    targets = load_targets()

    # K-Fold and Score Tracking
    scores = []
    wscores = []
    pscores = []
    n_folds = 10
    best_iters = [0] * n_folds
    # kf = GroupKFold(n_splits=K_FOLD)
    kf = StratifiedKFold(n_splits=K_FOLD)
    print('Training Model...')
    for fold, (train_idx, test_idx) in enumerate(kf.split(data, targets['target'])):
        print('\n[K = ' + str(fold+1) + ']')

        # Train Model
        dtrain = xgb.DMatrix(data[train_idx], label=targets[train_idx])
        dtest = xgb.DMatrix(data[test_idx])
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        bst = xgb.train(params_xgb, 
                        dtrain,
                        num_round,
                        watchlist,
                        verbose_eval=10,
                        #feval = eval_metric,
                        #maximize = True,
                        early_stopping_rounds=80)

        pred_prob_y = bst.predict(dtest).reshape(targets[test_idx].shape[0], 2) # predicted probabilities
        pred_y = bst.predict(dtest, ntree_limit=bst.best_ntree_limit).reshape(targets[test_idx].shape[0], 2)
        print ('predicted probabilities: ')
        print (pred_y)
        pred_y = np.argmax(pred_y, axis=1)
        print ('predicted label indices: ')
        print (pred_y)

        print ('best iterations: ', bst.best_ntree_limit)
        best_iters[fold] = bst.best_ntree_limit

        print (pred_y)
        print ('pred_y.shape')
        print (pred_y.shape)
        print ('y_valid.shape')
        print (targets[test_idx].shape)

        predicted = [LABELS[int(a)] for a in pred_y]
        actual = [LABELS[int(a)] for a in targets[test_idx]]
        s, _ = score_submission(actual, predicted)
        s_perf, _ = score_submission(actual, actual)
        r_score = report_score(actual, predicted)
        wscore = float(s) / s_perf
        print ('report_score', r_score)
        print ('fold %s, score = %f, perfect_score %f, weighted percentage %f' % (fold, s, s_perf, wscore))
        scores.append(s)
        pscores.append(s_perf)
        wscores.append(wscore)
    
    print ('scores:')
    print (scores)
    print ('mean score:')
    print (np.mean(scores))
    print ('perfect scores:')
    print (pscores)
    print ('mean perfect score:')
    print (np.mean(pscores))
    print ('w scores:')
    print (wscores)
    print ('mean w score:')
    print (np.mean(wscores))
    print ('best iters:')
    print (best_iters)
    print ('mean best_iter:')
    m_best = np.mean(best_iters)
    print (m_best)


if __name__ == '__main__':

    # cv()
    train()

