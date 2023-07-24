import numpy as np
from urllib.parse import urlparse
from gensim.models import Word2Vec
import pandas as pd

from chillml import Network
from chillml.layers import FullyConnected, Activation
from chillml.losses import BinaryCrossEntropy
from chillml.activations import Relu, Sigmoid

model = Word2Vec.load('examples/malicious_url_classifier/word2vec/word2vec_model.bin')

def tokenizer(url):
    if not urlparse(url).scheme:
        return urlparse(f'//{url}')
    else:
        return urlparse(url)

def vectorizer(url):
    tokens = tokenizer(url)
    vectors = [model.wv.get_vector(token) for token in tokens if token in model.wv]
    vector = np.mean(vectors, axis=0) if vectors else None
    return vector

data = pd.read_csv('examples/malicious_url_classifier/data.csv')
data = data.sample(frac=1, ignore_index=True)
data['url_vector'] = data['url'].apply(vectorizer).apply(lambda x: x.reshape(1, 100))
data['label_vector'] = data['label'].replace({'bad': 1, 'good': 0}).apply(lambda x: np.array(x).reshape(1, 1))

layers = [
    FullyConnected(100, 75), # input shape: [1, 100]
    Activation(Sigmoid),
    FullyConnected(75, 50),
    Activation(Sigmoid),
    FullyConnected(50, 1), # output shape: [1, 1]
    Activation(Sigmoid)
]

classifier = Network(layers, BinaryCrossEntropy)

while True:
    classifier.load('examples/malicious_url_classifier/classifer.pkl')
    classifier.train(data['url_vector'].values, data['label_vector'].values, 10, 0.05)
    classifier.save('examples/malicious_url_classifier/classifer.pkl')

    print('\nSaved Classifier... Examples:\n')

    forward = classifier.forward(data['url_vector'].values[:10])
    for i, value in enumerate(forward):
        print(value, data['label_vector'][i])