import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

##########################################
# tokenizing
# word tokenizers -> seperate by words
# setence tokenizers -> seperate by sentences
# corpora - body of text. ex: medical journals
# lexicon - words and their means

example_text = 'Hello Mr. Smith, there, how are you doing today? ' \
               'The weather is great and Python is awesome. The sky is pinkish-blue.' \
               'You should not eat carboard.'

# print(sent_tokenize(example_text))
# print(word_tokenize(example_text))

stop_words = set(stopwords.words("english"))

# print(stop_words)

words = word_tokenize(example_text)

filtered_sentence = [w for w in words if w not in stop_words]

# print(filtered_sentence)

#############################################
# stemming
# I was taking a ride in the car.
# I was riding in the car.
ps = PorterStemmer()

example_words = ['python', 'pythoner', 'pythoning']

# for w in example_words:
#     print(ps.stem(w))

new_text = 'It is very important to be pythonly while you are pythoning with python. all pythoners have pythoners.'

words = word_tokenize(new_text)

train_text = state_union.raw('2005-GWBush.txt')

sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

###############################################
# chunking
def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunk_gram = r'''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''

            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)

            chunked.draw()

            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

    except Exception as e:
        print(str(e))

# chinking
def process_content_2():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunk_gram = r"""Chunk: {<.*>+}
                                        }<VB.?|IN|DT|TO>+{"""

            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)

            chunked.draw()

            # print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

    except Exception as e:
        print(str(e))

# process_content_2()



with open(r'C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\sample_text.txt') as sample_text:
    text = sample_text.read()


#######################################################
# named entity
def process_content_3():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

# process_content_3()


#######################################################
# lemmatizing
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# print(lemmatizer.lemmatize('cats'))
# print(lemmatizer.lemmatize('better', pos='a'))   # pos to specify the type for the word, default is noun.
# print(lemmatizer.lemmatize('best', pos='a'))
# print(lemmatizer.lemmatize('ran','v'))



#######################################################
# corpora
import nltk
from nltk.corpus import gutenberg

# print(nltk.__file__)

sample = gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(sample)
print(tok[5:15])


#######################################################
# wordNet
from nltk.corpus import wordnet

syns = wordnet.synsets('program')
print(syns)

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())

#examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        print('l:',l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

# print(set(synonyms))
# print(set(antonyms))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')

print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')

print(w1.wup_similarity(w2))


#####################################################
# text classifier
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words['awesome'])
# print(all_words['stupid'])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]


training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# # positive data
# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]
# # negative data
# training_set = featuresets[100:]
# testing_set = featuresets[:100]


# Bayes:
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algorithm Accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)



###############################################
# save model with pickle
import pickle

# save with pickle
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# load model with pickle
classifier_f = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print('Original Naive Bayes Algorithm Accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


################################################
# scikit-learn

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


#MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#GaussianNB
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy percent:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

#BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


#################################################
#  ensemble
from nltk.classify import ClassifierI
from statistics import mode

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



voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]),
      ", Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]),
      ", Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[2][0]),
#       ", Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]),
      ", Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]),
      ", Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[5][0]),
#       ", Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)


#####################################################
# handle with bias



#####################################################
# better training data
from nltk.tokenize import word_tokenize

short_pos = open(r'C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\positive.txt','r').read()
short_neg = open(r'C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\negative.txt','r').read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r, 'pos'))

for r in short_neg.split('\n'):
    documents.append((r, 'neg'))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# Bayes:
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Original Naive Bayes Algorithm Accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)




##############################################################
# build a module

#####################################################
# better training data

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
import pickle


short_pos = open(r'C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\positive.txt','r').read()
short_neg = open(r'C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\negative.txt','r').read()


# j is adject, r is adverb, v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

all_words = []
documents = []

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets= open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()


random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# Bayes:
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Original Naive Bayes Algorithm Accuracy:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# save classifier
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\originalnaivebayes.5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\MNB_classifier.5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

#GaussianNB
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy percent:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

#BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\BernoulliNB_classifier.5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\LogisticRegression_classifier.5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\SGDClassifier_classifier.5k.pickle", "wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\LinearSVC_classifier.5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
save_classifier = open(r"C:\Users\WEIL\Documents\GitHub\pyfun\Text_Mining\NuSVC_classifier.5k.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()




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



# load sentiment_module
import sentiment_module as s

print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons... so yea!"))
print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was."))







# Graphing Live Twitter Sentiment
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    pullData = open("twitter-out.txt", "r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines[-200:]:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

        xar.append(x)
        yar.append(y)

    ax1.clear()
    ax1.plot(xar, yar)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()




def func3():
    for i in range(5):
        yield i


for f in func3():
    print(f)

