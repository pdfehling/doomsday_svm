#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA


# In[111]:


df = pd.read_excel(r'C:\Users\pdfeh\Dropbox\Delver\Artificial.Intelligence\MTGO ML\Doomsday_ML.xlsx')


# In[112]:


df.head()


# In[113]:


df.drop('Game', axis=1, inplace=True)
df.head()


# In[114]:


df.dtypes


# In[115]:


X = df.drop('Win1Yes', axis=1).copy()
X.head()


# In[116]:


y = df['Win1Yes'].copy()
y.head()


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[118]:


clf_svm = SVC(random_state=42)
clf_svm.fit(X_train, y_train)


# In[119]:


plot_confusion_matrix(clf_svm, 
                      X_test, 
                      y_test, 
                      values_format='d', 
                      display_labels=["Win", "Loss"])


# In[120]:


param_grid = [
    {'C': [0.5, 1, 10, 100],
    'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy', 
        verbose=0
    )

optimal_params.fit(X_train, y_train)
print(optimal_params.best_params_)


# In[121]:


doom_svm = SVC(random_state=42, C=1, gamma=0.1)
doom_svm.fit(X_train, y_train)


# In[122]:


plot_confusion_matrix(doom_svm,
                     X_test,
                     y_test,
                     values_format='d',
                     display_labels=['Win', 'Loss'])


# In[123]:


len(df.columns)


# In[124]:


pca = PCA()
X_train = pca.fit_transform(X_train)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()


# In[ ]:




