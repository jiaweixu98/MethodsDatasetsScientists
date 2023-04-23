from sklearn.feature_extraction.text import TfidfVectorizer
import nltknltk.download('stopwords')
nltk.download('wordnet')
# For cleaning the text

import spacy

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import regex as re

import string

import numpy as np

import nltk.data

import re

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize, sent_tokenize, pos_tag



def preprocess_text(text):
    # 1. Tokenise to alphabetic tokens
    text = remove_numbers(text)
    text = remove_http(text)
    text = remove_punctuation(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    text = remove_short_words(text)
    tokens = toknizing(text)

    # 2. POS tagging
    pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
    pos_tags_list = pos_tag(tokens)
    #print(pos_tags)

    # 3. Lowercase and lemmatise
    lemmatiser = WordNetLemmatizer()
    tokens = [lemmatiser.lemmatize(w.lower(), pos=pos_map.get(p[0], 'v')) for w, p in pos_tags_list]
    return tokens

#------------------------------------------------------------------------------

def convert_to_lower(text):
    return text.lower()

#------------------------------------------------------------------------------

def remove_numbers(text):
    text = re.sub(r'd+' , '', text)
    return text

def remove_http(text):
    text = re.sub("https?://t.co/[A-Za-z0-9]*", ' ', text)
    return text

def remove_short_words(text):
    text = re.sub(r'bw{1,2}b', '', text)
    return text

def remove_punctuation(text):
    punctuations = '''!()[]{};«№»:'",`./?@=#$-(%^)+&[*_]~'''
    no_punctuation = ""
    for char in text:
        if char not in punctuations:
            no_punctuation = no_punctuation + char
    return no_punctuation

def remove_white_space(text):
    text = text.strip()
    return text

def toknizing(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    ## Remove Stopwords from tokens
    result = [i for i in tokens if not i in stop_words]
    return result

vectorizer = TfidfVectorizer(use_idf=True, max_df=0.5, min_df=1, ngram_range=(1, 3))
vectors = vectorizer.fit_transform(dtf['cleaned_text'])

#clean text applying all the text preprocessing functions
dtf['cleaned_text'] = dtf.text.apply(lambda x: ' '.join(preprocess_text(x)))
dtf.head()

dict_of_tokens={i[1]:i[0] for i in vectorizer.vocabulary_.items()}

tfidf_vectors = []  # all deoc vectors by tfidf
for row in vectors:
    tfidf_vectors.append({dict_of_tokens[column]:value for (column,value) in zip(row.indices,row.data)})


doc_sorted_tfidfs =[]  # list of doc features each with tfidf weight
#sort each dict of a document
for dn in tfidf_vectors:
    newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
    newD = dict(newD)
    doc_sorted_tfidfs.append(newD)

tfidf_kw = [] # get the keyphrases as a list of names without tfidf values
for doc_tfidf in doc_sorted_tfidfs:
    ll = list(doc_tfidf.keys())
    tfidf_kw.append(ll)


# get the top N phrases
TopN = 30 

for i in range(len(tfidf_kw)):
    print(tfidf_kw[0][0:TopN])