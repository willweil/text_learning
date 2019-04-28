
#### save as sentiment_module.py

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

            choice_votes = votes.count(mode(votes))
            conf = choice_votes / len(votes)
        return conf

documents_pickle = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\documents.pickle", "rb")
documents = pickle.load(documents_pickle)
documents_pickle.close()

word_features5k_pickle = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\word_features5k.pickle", "rb")
word_features = pick.load(word_features5k_pickle)
word_features5k_pickle.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


featuresets_pickle = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_pickle)
featuresets_pickle.close()


random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]


# load classifier from pickle
open_file = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\ONB_classifier.5k.pickle", "rb")
ONB_classifier = pickle.load(open_file)
open_file.close()
open_file = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\MNB_classifier.5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()
open_file = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\BernoulliNB_classifier.5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
open_file = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\LogisticRegression_classifier.5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()
open_file = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\SGDClassifier_classifier.5k.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()
open_file = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\NuSVC_classifier.5k.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()



voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


# end of sentiment_module.py
#######################################################################