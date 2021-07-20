#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[ ]:





# In[3]:


books = pd.read_csv("https://query.data.world/s/prmljxdjiygq4krepunvwahz27wqu4", sep=';', encoding="latin-1", error_bad_lines=False)
users = pd.read_csv("https://query.data.world/s/ias5wyvlauegsm4l6kydu2itdpkt3s", sep=';', encoding="latin-1", error_bad_lines=False)
ratings = pd.read_csv("https://query.data.world/s/kdxvgbaw7bcmm4kdb3s3rxwqmlyb7g", sep=';', encoding="latin-1", error_bad_lines=False)


# In[4]:


books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)


# In[5]:


books.head()


# In[6]:


ratings['user_id'].value_counts()


# In[7]:


x = ratings['user_id'].value_counts() > 200
y = x[x].index  #user_ids
print(y.shape)
ratings = ratings[ratings['user_id'].isin(y)]


# In[8]:


rating_with_books = ratings.merge(books, on='ISBN')
rating_with_books.head()


# In[9]:


number_rating = rating_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
final_rating = rating_with_books.merge(number_rating, on='title')
final_rating.shape
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
final_rating.drop_duplicates(['user_id','title'], inplace=True)


# In[10]:


book_pivot = final_rating.pivot_table(columns='user_id', index='title', values="rating")
book_pivot.fillna(0, inplace=True)


# In[11]:


from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)


# In[12]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)


# In[13]:


distances, suggestions = model.kneighbors(book_pivot.iloc[237, :].values.reshape(1, -1))


# In[14]:


for i in range(len(suggestions)):
  print(book_pivot.index[suggestions[i]])


# In[ ]:




