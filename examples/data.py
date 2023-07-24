import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# Step 1: Load and preprocess your dataset
data = pd.read_csv('examples/data.csv')

# Step 2: Tokenization
def tokenizer(url):
    return url.split('/')[0].split('.')

tokens = data['url'].apply(tokenizer)

try:
    model = Word2Vec.load('examples/word2vec_model.bin')
except:
    # Step 3: Train your Word2Vec model
    model = Word2Vec(sentences=tokens, vector_size=100, window=5, min_count=1, workers=4)

    # Step 4: Save the trained model
    model.save("examples/word2vec_model.bin")
  
# Step 5: Convert URLs to vectors using the trained model
def vectorizer(url):
    tokens = tokenizer(url)
    vectors = [model.wv.get_vector(token) for token in tokens if token in model.wv]
    vector = np.mean(vectors, axis=0) if vectors else None
    return vector

# Step 6: Format data
data = data.sample(frac=1, ignore_index=True)
data['url'] = data['url'].apply(vectorizer).apply(lambda x: x.reshape(1, 100))
data['label'] = data['label'].replace({'bad': 1, 'good': 0}).apply(lambda x: np.array(x).reshape(1, 1))

data.to_csv('examples/urls.csv', index=False)