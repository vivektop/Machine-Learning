
import pandas as pd


stocks_data = pd.read_csv('company-stock-movements-2010-2015-incl.csv',index_col=0)

companies = list(stocks_data.index)
movements = stocks_data.values

from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

normalized_movements = normalize(movements)

mergings = linkage(normalized_movements, method='complete')

plt.figure(figsize=(10, 5))

dendrogram(
    mergings,
    labels=companies)

#dendrogram(
#    mergings,
#    labels=companies,
#    leaf_rotation=90.,
#    leaf_font_size=10
#)

plt.show()