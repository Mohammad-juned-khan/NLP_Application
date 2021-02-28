# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import requests
import pickle
import numpy as np
import pandas as pd
import re
#import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
wordnet = WordNetLemmatizer()
#library for text summary

punctuation = punctuation + "\n" + " " + "  "


app = Flask(__name__)
modell = pickle.load(open('final_model', 'rb'))
tfidf = pickle.load(open('tfidf', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('Index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        text = (request.form['article'])
        def preprocess_text(text):
            text= text.lower()
            text = text.replace('\n', ' ')
            text = text.replace('\r', '')
            text = text.strip()
            text = re.sub(' +', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)

            # removing stop words
            word_tokens = word_tokenize(text)
            filtered_sentence = []
            for word in word_tokens:
                if word not in set(stopwords.words('english')):
                    filtered_sentence.append(word)

            text = " ".join(filtered_sentence)

            return text

        def input_predict(text):
            # preprocess the text
            text = preprocess_text(text)
            # convert text to a list
            yh = [text]
            # transform the input
            inputpredict = tfidf.transform(yh)
            # predict the user input text
            y_predict_userinput = modell.predict(inputpredict)

            return y_predict_userinput
        output = input_predict(text)
        cat = int(output)
        if cat == 0:
            category="Business"
        elif cat == 1:
            category ="Entertainment"
        elif cat == 2:
            category ="Politics"
        elif cat == 3:
            category ="Sports"
        elif cat == 4:
            category ="Technology"
        else:
            category = "Error"

        return render_template('Index.html', text="Your Articel belongs to {}".format(category))

@app.route("/Summary")
def second():

    return  render_template('summary.html')

@app.route("/textsummary", methods=['POST'])
def textsummary():
    if request.method == 'POST':
        # import libraries

        stopword = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_sm')



        def process(text):

            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)

            # create Tokens of words
            tokens = [token.text for token in doc]
            # remove punctuations

            word_frequencies = {}
            for word in doc:
                if word.text.lower() not in stopword:
                    if word.text.lower() not in punctuation:
                        if word.text not in word_frequencies.keys():
                            word_frequencies[word.text] = 1
                        else:
                            word_frequencies[word.text] += 1
            max_frequency = max(word_frequencies.values())
            # normalize the frequencies of words
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_frequency
            # sentence tokenization
            sentence_tokens = [sent for sent in doc.sents]
            sentence_score = {}
            for sent in sentence_tokens:
                for word in sent:
                    if word.text.lower() in word_frequencies.keys():
                        if sent not in sentence_score.keys():
                            sentence_score[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_score[sent] += word_frequencies[word.text.lower()]
            select_length = int(len(sentence_tokens) * 0.4)
            summary = nlargest(select_length, sentence_score, key=sentence_score.get)
            final_summary = [word.text for word in summary]
            summary = " ".join(final_summary)

            return summary

        text = (request.form['text'])
        summaryoutput= process(text)


        return render_template('summary.html', Summary="Summary of your text is: {}".format(summaryoutput))




if __name__=="__main__":
    app.run(debug=True)
