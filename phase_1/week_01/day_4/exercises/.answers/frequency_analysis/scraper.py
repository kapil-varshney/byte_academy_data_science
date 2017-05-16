#!/usr/bin/env python3

from collections import Counter
from random import randint
import operator
import os
import pickle
import re

from bs4 import BeautifulSoup
from flask import Flask, render_template, request
import nltk
import requests

from stop_words import stop_list


microframework = Flask(__name__)


@microframework.route('/')
def index():
    return render_template('index.html')

@microframework.route('/seek', methods=["GET", "POST"])
def seek():
    errors  = []
    pairs = {}
    if request.method == "POST":
        try:
            url = request.form["url"]
            if 'http' not in url[:4]:
                url = "http://" + url
            response = requests.get(url)
        except:
            errors.append(
                "Invalid URL."
            )
            return render_template(
                "index.html",
                errors=errors
            )
        if response:
            nltk.data.path.append("./nltk_data/")
            sans_punctuation   = re.compile(".*[A-Za-z].*")
            structured_text    = BeautifulSoup(response.text, "html.parser").get_text()
            unstructured_text  = nltk.word_tokenize(structured_text)
            restructured_text  = nltk.Text(unstructured_text)
            words              = [word for word in restructured_text if sans_punctuation.match(word)]
            word_frequency     = Counter(words)
            keywords           = [word for word in words if word.lower() not in stop_list]
            keyword_frequency  = Counter(keywords)
            pairs = sorted(
                keyword_frequency.items(),
                key     = operator.itemgetter(1),
                reverse = True
            )
            with open('payload.pkl', "wb") as f:
               pickle.dump(keyword_frequency, f)
        return render_template(
            'index.html',
            errors=errors,
            pairs=pairs
        )


if __name__ == '__main__':
    microframework.run(host='127.0.0.1', port=5000, debug=True)
#    with open('payload.pkl', "rb") as f:
#        print(pickle.load(f))
