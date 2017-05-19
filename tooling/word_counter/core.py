#!/usr/bin/env python3

from collections import Counter
import csv
import operator
import pickle
import re

from bs4 import BeautifulSoup
import nltk

from reference_data.stop_words import stop_list


with open('reference_data/corpora.csv', newline='') as f:
    identifiers = csv.reader(f)
    for identifier in identifiers:
        resource = identifier[0]
        nltk.data.path.append("./nltk_data/")
        sans_punctuation  = re.compile(".*[A-Za-z].*")
        structured_text   = BeautifulSoup(open(resource), "html.parser").get_text()
        unstructured_text = nltk.word_tokenize(structured_text)
        restructured_text = nltk.Text(unstructured_text)
        words             = [word for word in restructured_text if sans_punctuation.match(word)]
        word_frequency    = Counter(words)
        keywords          = [word for word in words if word.lower() not in stop_list]
        keyword_frequency = Counter(keywords)
        pairs = sorted( 
            keyword_frequency.items(),
            key     = operator.itemgetter(1),
            reverse = True
        )
        with open('results/keyword_frequency.pkl', "wb") as f:
            pickle.dump(keyword_frequency, f)

if __name__ == '__main__':
    with open('results/keyword_frequency.pkl', "rb") as f:
        print(pickle.load(f))
