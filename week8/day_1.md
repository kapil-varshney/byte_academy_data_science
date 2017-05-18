## 1.0 Background


### 1.1 What is NLP? 

Natural Language Processing, or NLP, is an area of computer science that focuses on developing techniques to produce machine-driven analyses of text.

### 1.2 Why is Natural Language Processing Important? 

NLP expands the sheer amount of data that can be used for insight. Since so much of the data we have available is in the form of text, this is extremely important to data science!

A specific common application of NLP is each time you use a language conversion tool. The techniques used to accurately convert text from one language to another very much falls under the umbrella of "natural language processing."

### 1.3 Why is NLP a "hard" problem? 

Language is inherently ambiguous. Once person's interpretation of a sentence may very well differ from another person's interpretation. Because of this inability to consistently be clear, it's hard to have an NLP technique that works perfectly. 

### 1.4 Glossary

Here is some common terminology that we'll encounter throughout the workshop:

<b>Corpus: </b> (Plural: Corpora) a collection of written texts that serve as our datasets.

<b>nltk: </b> (Natural Language Toolkit) the python module we'll be using repeatedly; it has a lot of useful built-in NLP techniques.

<b>Token: </b> a string of contiguous characters between two spaces, or between a space and punctuation marks. A token can also be an integer, real, or a number with a colon.

## 1.0 Background


### 1.1 Polarity Flippers

Polarity flippers are words that change positive expressions into negative ones or vice versa. 

#### 1.1.1 Negation 

Negations directly change an expression's sentiment by preceding the word before it. An example would be

```
The cat is not nice.
```

#### 1.1.2 Constructive Discourse Connectives

Constructive Discourse Connectives are words which indirectly change an expression's meaning with words like "but". An example would be 

``` 
I usually like cats, but this cat is evil.
```

### 1.2 Multiword Expressions

Multiword expressions are important because, depending on the context, can be considered positive or negative. For example, 

``` 
This song is shit.
```
is definitely considered negative. Whereas

``` 
This song is the shit.
```
is actually considered positive, simply because of the addition of 'the' before the word 'shit'.

### 1.3 WordNet

WordNet is an English lexical database with emphasis on synonymy - sort of like a thesaurus. Specifically, nouns, verbs, adjectives and adjectives are grouped into synonym sets. 

#### 1.3.1 Synsets

nltk has a built-in WordNet that we can use to find synonyms. We import it as such:
``` python
from nltk.corpus import wordnet as wn
```

If we feed a word to the synsets() method, the return value will be the class to which belongs. For example, if we call the method on motorcycle,  

``` python
print(wn.synsets('motorcar'))
```

we get:

```
[Synset('car.n.01')]
```

Awesome stuff! But if we want to take it a step further, we can. We've previously learned what lemmas are - if you want to obtain the lemmas for a given synonym set, you can use the following method:

``` python
print(wn.synset('car.n.01').lemma_names())
```

This will get you:
```
['car', 'auto', 'automobile', 'machine', 'motorcar']
```

Even more, you can do things like get the definition of a word: 
``` python
print(wn.synset('car.n.01').definition())
```

Again, pretty neat stuff. 
```
'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
```

#### 1.3.2 Negation

With WordNet, we can easily detect negations. This is great because it's not only fast, but it requires no training data and has a fairly good predictive accuracy. On the other hand, it's not able to handle context well or work with multiple word phrases. 


### 1.4 SentiWordNet

Based on WordNet synsets, SentiWordNet is a lexical resource for opinion mining, where each synset is assigned three sentiment scores: positivity, negativity, and objectivity.

``` python
from nltk.corpus import sentiwordnet as swn
cat = swn.senti_synset('cat.n.03')
```

``` python
cat.pos_score()
```

``` python
cat.neg_score()
```

``` python
cat.obj_score()
```

### 1.5 Stop Words

Stop words are extremely common words that would be of little value in our analysis are often excluded from the vocabulary entirely. Some common examples are determiners like the, a, an, another, but your list of stop words (or <b>stop list</b>) depends on the context of the problem you're working on. 

### 1.6 Testing

#### 1.6.1 Cross Validation

Cross validation is a model evaluation method that works by not using the entire data set when training the model, i.e. some of the data is removed before training begins. Once training is completed, the removed data is used to test the performance of the learned model on this data. This is important because it prevents your model from over learning (or overfitting) your data. 

#### 1.6.2 Precision

Precision is the percentage of retrieved instances that are relevant - it measures the exactness of a classifier. A higher precision means less false positives, while a lower precision means more false positives. 

#### 1.6.3 Recall

Recall is the percentage of relevant instances that are retrieved. Higher recall means less false negatives, while lower recall means more false negatives. Improving recall can often decrease precision because it gets increasingly harder to be precise as the sample space increases.

#### 1.6.4 F-measure 

The f1-score is a measure of a test's accuracy that considers both the precision and the recall. 


## 3.0 Regular Expressions

A regular expression is a sequence of characters that define a string.

### 3.1 Simplest Form

The simplest form of a regular expression is a sequence of characters contained within <b>two backslashes</b>. For example, <i>python</i> would be  

``` 
\python
```

### 3.2 Case Sensitivity

Regular Expressions are <b>case sensitive</b>, which means 

``` 
\p and \P
```
are distinguishable from eachother. This means <i>python</i> and <i>Python</i> would have to be represented differently, as follows: 

``` 
\python and \Python
```

We can check these are different by running:

``` python
import re
re1 = re.compile('python')
print(bool(re1.match('Python')))
```

### 3.3 Disjunctions

If you want a regular expression to represent both <i>python</i> and <i>Python</i>, however, you can use <b>brackets</b> or the <b>pipe</b> symbol as the disjunction of the two forms. For example, 
``` 
[Pp]ython or \Python|python
```
could represent either <i>python</i> or <i>Python</i>. Likewise, 

``` 
[0123456789]
```
would represent a single integer digit. The pipe symbols are typically used for interchangable strings, such as in the following example:

```
\dog|cat
```

### 3.4 Ranges

If we want a regular expression to express the disjunction of a range of characters, we can use a <b>dash</b>. For example, instead of the previous example, we can write 

``` 
[0-9]
```
Similarly, we can represent all characters of the alphabet with 

``` 
[a-z]
```

### 3.5 Exclusions

Brackets can also be used to represent what an expression <b>cannot</b> be if you combine it with the <b>caret</b> sign. For example, the expression 

``` 
[^p]
```
represents any character, special characters included, but p.

### 3.6 Question Marks 

Question marks can be used to represent the expressions containing zero or one instances of the previous character. For example, 

``` 
<i>\colou?r
```
represents either <i>color</i> or <i>colour</i>. Question marks are often used in cases of plurality. For example, 

``` 
<i>\computers?
```
can be either <i>computers</i> or <i>computer</i>. If you want to extend this to more than one character, you can put the simple sequence within parenthesis, like this:

```
\Feb(ruary)?
```
This would evaluate to either <i>February</i> or <i>Feb</i>.

### 3.7 Kleene Star

To represent the expressions containing zero or <b>more</b> instances of the previous character, we use an <b>asterisk</b> as the kleene star. To represent the set of strings containing <i>a, ab, abb, abbb, ...</i>, the following regular expression would be used:  
```
\ab*
```

### 3.8 Wildcards

Wildcards are used to represent the possibility of any character and symbolized with a <b>period</b>. For example, 

```
\beg.n
```
From this regular expression, the strings <i>begun, begin, began,</i> etc., can be generated. 

### 3.9 Kleene+

To represent the expressions containing at <b>least</b> one or more instances of the previous character, we use a <b>plus</b> sign. To represent the set of strings containing <i>ab, abb, abbb, ...</i>, the following regular expression would be used:  

```
\ab+
```

## 4.0 Word Tagging and Models

Given any sentence, you can classify each word as a noun, verb, conjunction, or any other class of words. When there are hundreds of thousands of sentences, even millions, this is obviously a large and tedious task. But it's not one that can't be solved computationally. 


### 4.1 NLTK Parts of Speech Tagger

NLTK is a package in python that provides libraries for different text processing techniques, such as classification, tokenization, stemming, parsing, but important to this example, tagging. 

``` python
import nltk 

text = nltk.word_tokenize("Python is an awesome language!")
nltk.pos_tag(text)
```

```python
[('Python', 'NNP'), ('is', 'VBZ'), ('an', 'DT'), ('awesome', 'JJ'), ('language', 'NN'), ('!', '.')]
```

Not sure what DT, JJ, or any other tag is? Just try this in your python shell: 

```python
nltk.help.upenn_tagset('JJ')
```
``` 
JJ: adjective or numeral, ordinal
third ill-mannered pre-war regrettable oiled calamitous first separable
ectoplasmic battery-powered participatory fourth still-to-be-named
multilingual multi-disciplinary ...
```


#### 4.1.1 Ambiguity

But what if a word can be tagged as more than one part of speech? For example, the word "sink." Depending on the content of the sentence, it could either be a noun or a verb.

Furthermore, what if a piece of text demonstrates a rhetorical device like sarcasm or irony? Clearly this can mislead the sentiment analyzer to misclassify a regular expression. 


### 4.2 Unigram Models

Remember our bag of words model from earlier? One of its characteristics was that it didn't take the ordering of the words into account - that's why we were able to use dictionaries to map each words to True values. 

With that said, unigram models are models where the order doesn't make a difference in our model. You might be wondering why we care about unigram models since they seem to be so simple, but don't let their simplicity fool you - they're a foundational block for a lot of more advanced techniques in NLP. 

```python
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])

```


```
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'QL'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
```

### 4.3 Bigram Models

Here, ordering does matter. 

``` python
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
bigram_tagger.tag(brown_sents[2007])
```

Notice the changes from the last time we tagged the words of this same sentence: 

```
[('Various', 'JJ'), ('of', 'IN'), ('the', 'AT'), ('apartments', 'NNS'), ('are', 'BER'), ('of', 'IN'), ('the', 'AT'), ('terrace', 'NN'), ('type', 'NN'), (',', ','), ('being', 'BEG'), ('on', 'IN'), ('the', 'AT'), ('ground', 'NN'), ('floor', 'NN'), ('so', 'CS'), ('that', 'CS'), ('entrance', 'NN'), ('is', 'BEZ'), ('direct', 'JJ'), ('.', '.')]
```



## 5.0 Normalizing Text

The best data is data that's consistent - textual data usually isn't. But we can make it that way by normalizing it. To do this, we can do a number of things. 

At the very least, we can make all the text so that it's all in lowercase. You may have already done this before: 

Given a piece of text, 

``` python
raw = "OMG, Natural Language Processing is SO cool and I'm really enjoying this workshop!"
tokens = nltk.word_tokenize(raw)
tokens = [i.lower() for i in tokens]
```

```
['omg', ',', 'natural', 'language', 'processing', 'is', 'so', 'cool', 'and', 'i', "'m", 'really', 'enjoying', 'this', 'workshop', '!']
```


### 5.1 Stemming

But we can do more! 

#### 5.1.1 What is Stemming?

Stemming is the process of converting the words of a sentence to its non-changing portions. In the example of amusing, amusement, and amused above, the stem would be amus.

#### 5.1.2 Types of Stemmers

You're probably wondering how do I convert a series of words to its stems. Luckily, NLTK has a few built-in and established stemmers available for you to use! They work slightly differently since they follow different rules - which you use depends on whatever you happen to be working on. 

First, let's try the Lancaster Stemmer: 

``` python
lancaster = nltk.LancasterStemmer()
stems = [lancaster.stem(i) for i in tokens]
```

This should have the output: 
```
['omg', ',', 'nat', 'langu', 'process', 'is', 'so', 'cool', 'and', 'i', "'m", 'real', 'enjoy', 'thi', 'workshop', '!']
```

Secondly, we try the Porter Stemmer:

``` python
porter = nltk.PorterStemmer()
stem = [porter.stem(i) for i in tokens]
```

Notice how "natural" maps to "natur" instead of "nat" and "really" maps to "realli" instead of "real" in the last stemmer. 
```
['omg', ',', 'natur', 'languag', 'process', 'is', 'so', 'cool', 'and', 'i', "'m", 'realli', 'enjoy', 'thi', 'workshop', '!']
```


### 5.2 Lemmatization

#### 5.2.1 What is Lemmatization?

Lemmatization is the process of converting the words of a sentence to its dictionary form. For example, given the words amusement, amusing, and amused, the lemma for each and all would be amuse.

#### 5.2.2 WordNetLemmatizer

Once again, NLTK is awesome and has a built in lemmatizer for us to use: 

``` python
from nltk import WordNetLemmatizer

lemma = nltk.WordNetLemmatizer()
text = "Women in technology are amazing at coding"
ex = [i.lower() for i in text.split()]

lemmas = [lemma.lemmatize(i) for i in ex]
```

``` 
['woman', 'in', 'technology', 'are', 'amazing', 'at', 'coding']
```


Notice that women is changed to "woman"! 

