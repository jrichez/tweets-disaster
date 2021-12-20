# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:06:20 2021

@author: riche
"""

import pandas as pd
import numpy as np
import gradio as gr
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import re

def clean_text(text):

    text = text.translate(punctuation)

    text = text.lower().split()

    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]

    text = " ".join(text)
    text = re.sub(r"[^\w\s]", " ",text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ",text)

    text = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in text]
    text = " ".join(lemmatized_words)


    return text

with open("tokenizer.json", "r") as read_file:
   tokenizer = json.load(read_file)

text = tokenizer.texts_to_sequences(text)

vocab_size = len(tokenizer.word_index) + 1

text = pad_sequences(text, padding='post', maxlen=50)

model = load_model('tweets_disaster_def.h5')

def tweets_predictions(text):
    text = clean_text(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, padding='post', maxlen=50)
    pred = model.predict(text.reshape(1,-1)).tolist()[0]
    dic = {}
    dic['No disaster'] = 1 - pred[0]
    dic['Disaster'] = pred[0]
    return dic
	
interface = gr.Interface(fn=tweets_predictions, inputs='textbox', outputs='label').launch(share=True)   