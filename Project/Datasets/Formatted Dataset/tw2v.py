import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
import re
import time
import matplotlib.pyplot as plt

# Create the Model
'''sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size = 200)
model.save('text8.model')
model.wv.save_word2vec_format('text.model.bin', binary = True)'''

model = KeyedVectors.load_word2vec_format('text.model.bin', binary = True)
MODEL_LEN = 200

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

text_file = open("targets.csv", "w")

for index, row in x.iterrows():
    h1 = row['Headline'].lower()
    print(h1)
    date_info_array = [0] * MODEL_LEN
    dateString = row['Date']
    date_info_array[0] = int(dateString[2] + dateString[3])
    date_info_array[1] = row['Month']
    date_info_array[2] = row['Weekday']
    
    
    wa = []
    for w in h1.split(" "):
        filtered = re.findall('\w+', w)
        filtered = "".join(filtered)
        if len(filtered) > 0 :
            try:
                wa.append(model[filtered])
            except:
                pass
    
    wa = np.array(wa).T
    #print(wa.shape)
    
    # zero padding to max length of sentence (words)
    mnwh = max_length
    pwa = np.zeros((MODEL_LEN, mnwh+1)) 
    pwa[:,1:wa.shape[1]+1] = wa
    pwa[:,0] = date_info_array

    #plt.imshow(pwa)
    #plt.show()
    #print(pwa)
    print(pwa.shape)
    np.savetxt("./ds/"+str(index)+".csv", pwa, delimiter=",")
    np.save("./ds/"+str(index)+".npy", pwa)
    text_file.write(str(index)+".csv,{:.9f}\n".format(row["ClosePriceUSD"]))

    n = average # minimum or average size of 
    pca_n = TruncatedSVD(n_components=n, algorithm='arpack')
    pca_result_n = pca_n.fit_transform(pwa)
    np.save("./ds/r-"+str(index)+".npy", pca_result_n)
    #plt.imshow(pca_result_n)
    #plt.show()
    pca_result_n = pca_result_n.flatten()
    print('Array Flattened')
    print('Explained variation per principal component (PCA): {}'.format(np.sum(pca_n.explained_variance_ratio_)))
    
    print(pca_result_n)
    print(pca_result_n.shape)
    #break

text_file.close()


