import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("emotion-labels-test.csv",encoding="latin1")
data.head()
data.tail()

sns.countplot(data.label)

data.label = [0 if each == "joy" 
              else 
              1 if each == "fear" 
              else  
              2 if each == "anger"
              else 3 
              for each in data.label]

import re
first_text = data.text[4]
text = re.sub("[^a-zA-Z]"," " ,first_text)

import nltk
nltk.download("stopwords")
nltk.download("punkt") 
nltk.download("wordnet")
from nltk.corpus import stopwords

text = nltk.word_tokenize(text)
text = [word for word in text if not word in set(stopwords.words("english"))]

import nltk as nlp
lemma = nlp.WordNetLemmatizer()
text = [ lemma.lemmatize(word) for word in text]

text = " ".join(text)


#%% yukarıda ki işlemi tüm veriye uygulamak için

text_list = []

for text in data.text:
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    text_list.append(text)
    
#%% bag of word

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(max_features=2500)   

sparce_matrix = count_vectorizer.fit_transform(text_list).toarray()

#%% model training 

y = data.iloc[:,1].values 
x = sparce_matrix

from sklearn.model_selection import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

#naive - bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("accuracy : " ,nb.score(x_test,y_test))

#decision tree

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)

print("accuracy : " ,dt.score(x_test,y_test))

#random - forest regression

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=5, random_state=42)

rf.fit(x_train,y_train)

print("accuracy : " , rf.score(x_test, y_test))







