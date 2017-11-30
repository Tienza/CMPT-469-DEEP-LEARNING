import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
import re

# Create the Model
'''sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size = 200)
model.save('text8.model')
model.wv.save_word2vec_format('text.model.bin', binary = True)'''

model = KeyedVectors.load_word2vec_format('text.model.bin', binary = True)

print(model['sentence'])
print(len(model['sentence']))
print(model['sentence'].shape)

x = pd.read_csv('formatted_dataset.csv', sep='^')

# Get the length of the largest headliine and find the average
headline_length_list = []
for headline in range(len(x['Headline'])):
    headline_length_list.append(len(x['Headline'][headline].split(" ")))

max_length = max(headline_length_list)
average = int(sum(headline_length_list) / len(headline_length_list))
# for row, idx in df.iter_rows:
# x['Headline']
# x['Close Price']
h1 = x['Headline'][0].lower()
print(h1)

wa = []
for w in h1.split(" "):
    filtered = re.findall('\w+', w)
    filtered = "".join(filtered)
    if len(filtered) > 0 :
        try:
            print(filtered)
            wa.append(model[filtered])
        except:
            pass

wa = np.array(wa).T
print(wa.shape)

# zero padding to max length of sentence (words)
mnwh = max_length
pwa = np.zeros((200, mnwh)) 
pwa[:wa.shape[0],:wa.shape[1]] = wa

print(pwa)
print(pwa.shape)

n = average # minimum or average size of 
pca_n = TruncatedSVD(n_components=n, algorithm='arpack')
pca_result_n = pca_n.fit_transform(pwa)
print('Explained variation per principal component (PCA): {}'.format(np.sum(pca_n.explained_variance_ratio_)))

print(pca_result_n)
print(pca_result_n.shape)