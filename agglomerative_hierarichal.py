
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import pandas as pd

X=pd.read_csv('CLUSTER_DATA.csv')
X.drop('NAME',axis=1,inplace=True)
model = AgglomerativeClustering(linkage='complete', n_clusters=2)
model.fit(X)
labels = model.labels_

