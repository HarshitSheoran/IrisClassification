#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("iris_data.csv")
X = np.array(data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = np.array(data['species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[3]:


clf = SVC()


# In[4]:


X.shape[1]


# In[5]:


clf.fit(X_train, y_train)


# In[6]:


best_train_predictions = clf.predict(X_train)
best_test_predictions = clf.predict(X_test)

print("The training F1 score is", f1_score(best_train_predictions, y_train, average='macro'))
print("The testing F1 score is", f1_score(best_test_predictions, y_test, average='macro'))


# In[8]:


clf.predict([[6.5,3,7,4]])


# In[ ]:





# In[ ]:





# In[ ]:




