#import all the necessary libraries for data analysis

import bamboolib as bam                        #easy to do data manipulation and highly recommend going to https://bamboolib.8080labs.com/ and that will help you follow along
import pandas as pd                            #data manipulation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker
import gensim 
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

df=pd.read_csv('C:/Users/deeti/OneDrive/Documents/Winter 2022/ALY6040- Data Mining Applications/Reviews.csv')
df['Time'] = pd.to_datetime(df['Time'], unit='s')
df.sort_values(by='Time')

df.head()

df[df.HelpfulnessNumerator == 0].shape
df.shape

df.describe()

print(df['Score'].value_counts(),'\n','  ***'*10)

plt.figure(figsize=(10,5))
ax = sns.countplot(x='Score',data=df,palette='Set3')
#sns.countplot()
plt.title('Ratings dist across whole dataset')
plt.xlabel('Reviews ratings')
plt.ylabel("No. of reviews corresponding to each of 5 ratings")
plt.show()

numUsers = len(df['UserId'].unique())
numProducts = len(df['ProductId'].unique())

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
axes.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
sns.barplot(data=pd.DataFrame({
    'data': ['reviews', 'users', 'products'],
    'count': [len(df), numUsers, numProducts],
}), x='count', y='data', ax=axes)

newUsersPerMonth = df[['UserId', 'Time']].sort_values(by='Time').drop_duplicates(subset=['UserId']).groupby(pd.Grouper(key='Time', freq='M')).count().reset_index()
newProductsPerMonth = df[['ProductId', 'Time']].sort_values(by='Time').drop_duplicates(subset=['ProductId']).groupby(pd.Grouper(key='Time', freq='M')).count().reset_index()

plt.figure(figsize=(12, 4))
plt.title('New Users / New Products Per Month, 1999 to 2012')
sns.lineplot(data=newUsersPerMonth, x='Time', y='UserId', label='New Users Per Month')
ax = sns.lineplot(data=newProductsPerMonth, x='Time', y='ProductId', label='New Products Per Month')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

reviewsPerMonth = df.groupby(pd.Grouper(key='Time',freq='M')).count().reset_index()

fig, axes = plt.subplots(1, 1, figsize=(12, 4))
axes.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

sns.lineplot(data=reviewsPerMonth, x='Time', y='Id', label='Reviews Per Month')

# Reviews per user over time
data = df[['Time', 'Id', 'UserId']].groupby(by=[
    pd.Grouper(key='Time', freq='M'),
    'UserId',
]).count()
data['UserId_1'] = 1
data = data.reset_index().groupby(by=pd.Grouper(key='Time', freq='M')).sum()
data['Rewiews_Per_User'] = data['Id'] / data['UserId_1']
data = data.reset_index().dropna()

plt.figure(figsize=(12, 4))
plt.title('#Reviews per user in each month, 1999 - 2012')
ax = sns.lineplot(data=data, x='Time' , y='Rewiews_Per_User')
ax.set(ylabel='Rewiews Per User')
plt.show()

# Reviews per product over time
data = df[['Time', 'Id', 'ProductId']].groupby(by=[
    pd.Grouper(key='Time', freq='M'),
    'ProductId',
]).count()
data['ProductId_1'] = 1
data = data.reset_index().groupby(by=pd.Grouper(key='Time', freq='M')).sum()
data['Rewiews_Per_Product'] = data['Id'] / data['ProductId_1']
data = data.reset_index().dropna()

plt.figure(figsize=(12, 4))
plt.title('#Reviews per product in each month, 1999 - 2012')
ax = sns.lineplot(data=data, x='Time' , y='Rewiews_Per_Product')
ax.set(ylabel='Rewiews Per Product')
plt.show()

data = df.groupby(pd.Grouper(key='Time',freq='M')).mean().reset_index()
fig, axes = plt.subplots(1, 1, figsize=(12, 4))
sns.lineplot(data=data, x='Time', y='Score')

#Topic Modelling
df = df.sample(n=20000)
df.shape

%%time
import nltk
nltk.download('wordnet')
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, preprocess_string

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

df['Text_tokenized'] = df['Text'].apply(lambda text: preprocess_string(text, [
    strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, lemmatizer.lemmatize, lambda x: x.lower()
]))

df['Text_tokenized'].sample(n=20)

#review lengths
df['Text_len'] = df['Text_tokenized'].apply(lambda text: len(text))


plt.figure(figsize=(8, 4))
sns.histplot(df[df['Text_len'] > 0]['Text_len'], log_scale=True)

#wordcount
from wordcloud import WordCloud

long_string = ','.join([' '.join(words) for words in df['Text_tokenized'].values])
wordcloud = WordCloud()
wordcloud.generate(long_string)
wordcloud.to_image()

#LDA Topic Modelling
import gensim

dictionary = gensim.corpora.Dictionary(df['Text_tokenized'].values)

dictionary.filter_extremes(no_below=20, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in df['Text_tokenized'].values]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

%%time 
import logging
model = gensim.models.ldamulticore.LdaMulticore(corpus, id2word=dictionary, passes=10)


#visalising result

%time
import gensim 
import pyLDAvis
import pyLDAvis.gensim_models

prep_display = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
pyLDAvis.display(prep_display)
