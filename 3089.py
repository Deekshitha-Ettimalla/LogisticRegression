#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


dataset=pd.read_csv('iris.csv')


# In[4]:


dataset.describe()


# In[6]:


dataset.info()


# In[7]:


X = dataset.iloc[:, [0,1,2, 3]].values  #input features
y = dataset.iloc[:, 4].values  #target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[8]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)


# In[9]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)


# In[11]:


test_accuracy = classifier.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)


# In[12]:


y_pred = classifier.predict(X_test)


# In[14]:


X_test


# In[15]:


probs_y = classifier.predict_proba(X_test)


# In[16]:


probs_y


# In[17]:


probs_y=np.round(probs_y,2)


# In[18]:


probs_y


# In[19]:


res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("y_test", "y_pred", "Setosa(%)", "versicolor(%)", "virginica(%)\n")
res += "-"*65+"\n"
res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) for x, y, a, b, c in 
                 zip(y_test, y_pred, probs_y[:,0], probs_y[:,1], probs_y[:,2]))
res += "\n"+"-"*65+"\n"
print(res)


# In[20]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[21]:


import seaborn as sns
import pandas as pd

ax = plt.axes()
df_cm = cm
sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = ax )
ax.set_title('Confusion Matrix')
plt.show()


# In[ ]:




