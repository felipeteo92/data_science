#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


# In[20]:


base_estudos = pd.read_csv(r"C:\Users\lipet\OneDrive\Área de Trabalho\horas_estudo.csv", sep = ";")


# In[10]:


print(base_estudos.dtypes)


# In[11]:


base_estudos.isna().sum()


# In[37]:


dados_duplicados = base_estudos[base_estudos.duplicated()]


# In[38]:


dados_duplicados


# In[35]:


base_estudos = base_estudos.drop_duplicates()


# In[18]:


base_estudos = base_estudos.apply(lambda x: x.str.replace(',', '.'))


# In[26]:


base_estudos = base_estudos.apply(lambda x: x.str.replace(',', '.'))


# In[33]:


print(base_estudos.dtypes)


# In[32]:


base_estudos['horas_estudo'] = base_estudos['horas_estudo'].astype(float)
base_estudos['nota_avaliacao'] = base_estudos['nota_avaliacao'].astype(float)


# In[36]:


base_estudos['horas_estudo'].mean()


# In[42]:


correlation = base_estudos.corr()


# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt

plot = sns.heatmap(correlation, annot = True, linewidths = .3)
plt.show()


# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.array(base_estudos['horas_estudo']).reshape(-1, 1)
y = np.array(base_estudos['nota_avaliacao']).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state=123)

model = LinearRegression()
model.fit(x_train, y_train)

#indica a qualidade do algoritmo (Quanto mais próximo de 1, melhor a qualidade do modelo)
model.score(x, y)


# In[55]:


y_predicted = model.predict(x)
plt.scatter(x, y)
plt.plot(x, y_predicted, color = 'red')

plt.show()


# In[57]:


model.intercept_


# In[58]:


model.coef_


# In[60]:


model.predict([[25]])


# In[59]:


model.predict([[18.06]])
base_estudos[base_estudos[horas_estudo]==18.06]


# In[69]:


nota_previsao = model.predict([[18.06]])
base_estudos['nota_avaliacao'][base_estudos['horas_estudo']==18.06] - nota_previsao[0][0]


# In[73]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(y, y_predicted)
MAE


# In[ ]:




