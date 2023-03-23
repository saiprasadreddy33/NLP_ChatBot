import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from functools import lru_cache
import threading
import cProfile

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

with open('training_data.txt', 'r') as file:
    training_data = file.readlines()
training_data = [line.strip().split('\t') for line in training_data]

def train_classifier():
    documents = [(word_tokenize(text.lower()), category) for text, category in training_data]
    all_words = []
    for text, _ in documents:
        for word in text:
            if word not in stop_words:
                all_words.append(lemmatizer.lemmatize(word))
    all_words_freq = nltk.FreqDist(all_words)
    word_features = [word for word, freq in all_words_freq.most_common(1000)]
    featuresets = [(get_features(text), category) for (text, category) in documents]
    training_set = featuresets[:800]
    testing_set = featuresets[800:]
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set)
    print('Classifier accuracy:', accuracy)
    return classifier

@lru_cache(maxsize=1000)
def get_features(text):
    features = defaultdict(int)
    for word in word_tokenize(text.lower()):
        if word not in stop_words:
            features[lemmatizer.lemmatize(word)] = 1
    return features

classifier = train_classifier()

def process_input(input_text):
    features = get_features(input_text)
    category = classifier.classify(features)
    if category == 'greeting':
        return 'Hello! How can I assist you today?'
    elif category == 'farewell':
        return 'Goodbye!'
    elif category == 'question':
        return 'I am sorry, I cannot answer that question at this time.'
    else:
        return 'I am sorry, I do not understand your request.'

def profile_process_input():
    cProfile.runctx('process_input("What can you do?")', globals(), locals())

def run_multi_threading():
    threads = []
    for i in range(10):
        t = threading.Thread(target=profile_process_input)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

run_multi_threading()
