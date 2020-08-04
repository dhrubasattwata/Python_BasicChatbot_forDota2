# Import Packages
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Import NLP Packages
# pip install nltk

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

# Reading the Dataset
f=open('C:/Users/Dhruba/Desktop/GitHub/Python/chatbox.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase


# Tokenization: Tokenization is just the term used to describe the process of converting the normal text strings into a list of tokens
# i.e words that we actually want.
# Sentence tokenizer can be used to find the list of sentences and Word tokenizer can be used to find the list of words in strings.
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing¶
# We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.

lemmer = nltk.stem.WordNetLemmatizer()

#WordNet is a semantically-oriented dictionary of English included in NLTK.

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword matching
# Greeting Response

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["I hope you will not bore me", "I have all the information you need", "*nods*", "You there"]
def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating Response

# Bag of Words
# After the initial preprocessing phase, we need to transform text into a meaningful vector (or array) of numbers.
# The bag-of-words is a representation of text that describes the occurrence of words within a document.
# It involves two things:
# 1. A vocabulary of known words.
# 2. A measure of the presence of known words.
# Why is it is called a “bag” of words? That is because any information about the order or structure of words
# in the document is discarded and the model is only concerned with whether the known words occur in the document,
# not where they occur in the document.

# TF-IDF Approach
# A problem with the Bag of Words approach is that highly frequent words start to dominate in the document
# (e.g. larger score), but may not contain as much “informational content”. Also, it will give more weight to longer
#  documents than shorter documents.

# One approach is to rescale the frequency of words by how often they appear in all documents so that the scores
# for frequent words like “the” that are also frequent across all documents are penalized. This approach to scoring
# is called Term Frequency-Inverse Document Frequency, or TF-IDF for short, where:

# Term Frequency: is a scoring of the frequency of the word in the current document.
# TF = (Number of times term t appears in a document)/(Number of terms in the document)

# Inverse Document Frequency: is a scoring of how rare the word is across documents.
# IDF = 1+log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.

# Cosine Similarity
# Tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used
#  to evaluate how important a word is to a document in a collection or corpus

# Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||
# where d1,d2 are two non zero vectors.

# To generate a response from our bot for input questions, the concept of document similarity will be used.
# We define a function response which searches the user’s utterance for one or more known keywords and returns
# one of several possible responses.
# If it doesn’t find the input matching any of the keywords, it returns a response:”You are in the wrong place ! This is for Dota 2”

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"You are in the wrong place ! This is for Dota 2"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# Generating the Dota 2 Chatbot

flag=True
print("Outworld: I will tell you about the heroes that fight the battle in the land of Dota 2. If you are scared, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Outlworld: You should be.")
        else:
            if(greeting(user_response)!=None):
                print("Outlworld: "+greeting(user_response))
            else:
                print("Outlworld: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Outlworld: Scared cat! Run....")
