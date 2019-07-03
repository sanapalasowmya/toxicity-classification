#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) python data analysis library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os#interface with the underlying operating system that Python is running on – be that Windows, Mac or Linux. 
print(os.listdir(r"C:\Users\ishak\OneDrive\Desktop\project bennett"))

# Any results you write to the current directory are saved as output.


# # 1. EDA

# In[1]:


import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#Matplotlib is a Python 2D plotting library.generate plots, histograms, power spectra, bar charts, errorcharts, scatterplots, etc., 
import seaborn as sns#Seaborn is a library for making statistical graphics in Python.


# In[2]:


from IPython.display import Markdown, display#Markdown is a lightweight and easy-to-use syntax for creating HTML. 
def printmd(string):
    display(Markdown(string))
#printmd('**bold**')


# In[3]:


data_path ="..input/train.csv"


# In[4]:


data_raw = pd.read_csv(data_path)
#data_raw = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
data_raw.shape


# In[5]:



print("Number of rows in data =",data_raw.shape[0])
print("Number of columns in data =",data_raw.shape[1])
print("\n")
printmd("**Sample data:**")
data_raw.head()


# ## Checking for missing values

# In[10]:


missing_values_check = data_raw.isnull().sum()
print(missing_values_check)


# ## Calculating number of comments under each label

# In[11]:


# Comments with no label are considered to be clean comments.
# Creating seperate column in dataframe to identify clean comments.

# We use axis=1 to count row-wise and axis=0 to count column wise

rowSums = data_raw.iloc[:,2:].sum(axis=1)
clean_comments_count = (rowSums==0).sum(axis=0)

print("Total number of comments = ",len(data_raw))
print("Number of clean comments = ",clean_comments_count)
print("Number of comments with labels =",(len(data_raw)-clean_comments_count))


# In[9]:


categories = list(data_raw.columns.values)
categories = categories[2:]
print(categories)


# In[ ]:


# Calculating number of comments in each category
'''
counts = []
for category in categories:
    counts.append((category, data_raw[category].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])
df_stats


# In[ ]:


sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax= sns.barplot(categories, data_raw.iloc[:,2:].sum().values)

plt.title("Comments in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Comment Type ', fontsize=18)

#adding the text labels
rects = ax.patches
labels = data_raw.iloc[:,2:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

plt.show()
'''

# In[ ]:


## Calculating number of comments having multiple labels


# In[ ]:
'''

rowSums = data_raw.iloc[:,2:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]

sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)

plt.title("Comments having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)

#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
'''

# In[ ]:





# # 2. Data Pre-Processing

# In[10]:


data = data_raw
data = data_raw.loc[np.random.choice(data_raw.index, size=2000)]
data.shape


# In[11]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ##  Cleaning Data

# In[12]:


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


# In[13]:


data['comment_text'] = data['comment_text'].str.lower()
data['comment_text'] = data['comment_text'].apply(cleanHtml)
data['comment_text'] = data['comment_text'].apply(cleanPunc)
data['comment_text'] = data['comment_text'].apply(keepAlpha)
data.head()


# ##  Removing Stop Words

# In[14]:


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data['comment_text'] = data['comment_text'].apply(removeStopWords)
data.head()


# ## Stemming

# In[27]:


stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data['comment_text'] = data['comment_text'].apply(stemming)
data.head()


# ## Train-Test Split

# In[18]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

print(train.shape)
print(test.shape)


# In[19]:


train_text = train['comment_text']
test_text = test['comment_text']


# ## TF-IDF

# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)


# In[21]:


x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['id','comment_text'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['id','comment_text'], axis=1)


#  # 3. Multi-Label Classification
# ##  Multiple Binary Classifications - (One Vs Rest Classifier)

# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier


# In[23]:


get_ipython().run_cell_magic('time', '', '\n
# Using pipeline for applying logistic regression and one vs rest classifier\n
LogReg_pipeline = Pipeline([(\'clf\', OneVsRestClassifier(LogisticRegression(solver=\'sag\'), n_jobs=-1))])

for category in categories:
    printmd(\'**Processing {} comments...**\'.format(category))
    # Training logistic regression model on train data\n   
    LogReg_pipeline.fit(x_train, train[category])
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print(\'Test accuracy is {}\'.format(accuracy_score(test[category], prediction)))
    print("\\n")')


# ## Multiple Binary Classifications - (Binary Relevance)

# In[24]:


get_ipython().run_cell_magic('time', '', '\n
           # using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(x_train, y_train)
# predict
predictions = classifier.predict(x_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\\n")')


# ## Classifier Chains

# In[25]:


# using classifier chains
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression


# In[26]:


get_ipython().run_cell_magic('time', '', '\n
  #initialize classifier chains multi-label classifier
classifier = ClassifierChain(LogisticRegression())
  # Training logistic regression model on train data
classifier.fit(x_train, y_train)
  # predict
predictions = classifier.predict(x_test)
  # accuracy
print("Accuracy = ",accuracy_score(y_test,predictions))
print("\\n")')


# In[ ]:





# In[ ]:




