import numpy as np
from gensim.models import Word2Vec
import pandas as pd

from chillml import Network
from chillml.layers import FullyConnected, Activation
from chillml.losses import MeanSquaredError
from chillml.activations import Relu, Softmax

model = Word2Vec.load('examples/word2vec_model.bin')

def tokenizer(url):
    return url.split('/')[0].split('.')

def vectorizer(url):
    tokens = tokenizer(url)
    vectors = [model.wv.get_vector(token) for token in tokens if token in model.wv]
    vector = np.mean(vectors, axis=0) if vectors else None
    return vector

data = pd.read_csv('examples/data.csv')
data = data.sample(frac=1, ignore_index=True)
data['url'] = data['url'].apply(vectorizer).apply(lambda x: x.reshape(1, 100))
data['label'] = data['label'].replace({'bad': 1, 'good': 0}).apply(lambda x: np.array(x).reshape(1, 1))


layers = [
    FullyConnected(100, 75), # input shape: [1, 100]
    Activation(Relu),
    FullyConnected(75, 50),
    Activation(Relu),
    FullyConnected(50, 1) # output shape: [1, 1]
]

classifier = Network(layers, MeanSquaredError)

forward = classifier.forward(data['url'].values[:25])
for i, value in enumerate(forward):
    print(value, data['label'][i])

classifier.train(data['url'][:10000], data['label'][:10000], 5000, 0.005)

forward = classifier.forward(data['url'].values[:25])
for i, value in enumerate(forward):
    print(value, data['label'][i])

classifier.save('examples/weights.pkl')