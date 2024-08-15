import pandas as pd
data = pd.read_csv("Language_Detection.csv")
data.head(10)
language_counts=data["Language"].value_counts()
language_percentages = (language_counts / len(data)) * 100
print(language_percentages)

#separation variable dependante et indépendante

X = data["Text"]
y = data["Language"]
def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))
import string as st
data['removed_punc'] = data['Text'].apply(lambda x: remove_punct(x))
data.head()

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

import re
data['tokens'] = data['removed_punc'].apply(lambda msg : tokenize(msg))
data.head()

def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]


data['larger_tokens'] = data['tokens'].apply(lambda x :remove_small_words(x) )

data.head()
import nltk

def remove_stopwords(text):
    return[word for word in text if word not in nltk.corpus.stopwords.words('english')]
nltk.download('stopwords')