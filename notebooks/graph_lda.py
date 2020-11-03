#
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
import pickle
import matplotlib.pyplot as plt
path = '/Users/collinswestnedge/programming/Metis_Online/project_04/pickles'
with open(path, 'rb') as file:
    lda = pickle.load(file)
    word_vect = pickle.load(file)
    vectorizer = pickle.load(file)


visualizer = pyLDAvis.sklearn.prepare(lda, np.matrix(word_vect), vectorizer, mds='mmds')
pyLDAvis.show(visualizer)
