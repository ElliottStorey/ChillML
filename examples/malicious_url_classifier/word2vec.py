import pandas as pd
from urllib.parse import urlparse
from gensim.models import Word2Vec

# Step 1: Load and preprocess your dataset
data = pd.read_csv('examples/malicious_url_classifier/data.csv')

# Step 2: Tokenization
def tokenizer(url):
    if not urlparse(url).scheme:
        return urlparse(f'//{url}')
    else:
        return urlparse(url)

tokens = data['url'].apply(tokenizer)

# Step 3: Train your Word2Vec model
model = Word2Vec(sentences=tokens, min_count=1)

# Step 4: Save the trained model
model.save("examples/malicious_url_classifier/word2vec/word2vec_model.bin")
  
# Step 5: Convert URLs to vectors using the trained model
'''
def vectorizer(url):
    tokens = tokenizer(url)
    vectors = [model.wv.get_vector(token) for token in tokens if token in model.wv]
    vector = np.mean(vectors, axis=0) if vectors else None
    return vector

# Step 6: Format data
data = data.sample(frac=1, ignore_index=True)
data['url'] = data['url'].apply(vectorizer).apply(lambda x: x.reshape(1, 100))
data['label'] = data['label'].replace({'bad': 1, 'good': 0}).apply(lambda x: np.array(x).reshape(1, 1))
'''