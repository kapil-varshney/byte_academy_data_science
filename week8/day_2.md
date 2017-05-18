## 2.0 Sentiment Analysis  


So you might be asking, what exactly is "sentiment analysis"? 

Well, sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that a machine can read, and using statistics to determine the actual sentiment.

### 2.1 Preparing the Data 

To accomplish sentiment analysis computationally, we have to use techniques that will allow us to learn from data that's already been labeled. 

So what's the first step? Formatting the data so that we can actually apply NLP techniques. 

``` python
import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})
```
Here, format_sentence changes a piece of text, in this case a tweet, into a dictionary of words mapped to True booleans. Though not obvious from this function alone, this will eventually allow us to train  our prediction model by splitting the text into its tokens, i.e. <i>tokenizing</i> the text.

``` 
{'!': True, 'animals': True, 'are': True, 'the': True, 'ever': True, 'Dogs': True, 'best': True}
```
You'll learn about why this format is important is section 2.2.

Using the data on the github repo, we'll actually format the positively and negatively labeled data.

``` python
pos = []
with open("./pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])
```

``` python
neg = []
with open("./neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])
```


#### 2.1.1 Training Data

Next, we'll split the labeled data we have into two pieces, one that can "train" data and the other to give us insight on how well our model is performing. The training data will inform our model on which features are most important.

``` python
training = pos[:int((.9)*len(pos))] + neg[:int((.9)*len(neg))]
```

#### 2.1.2 Test Data

We won't use the test data until the very end of this section, but nevertheless, we save the last 10% of the data to check the accuracy of our model. 
``` python
test = pos[int((.1)*len(pos)):] + neg[int((.1)*len(neg)):]
```

### 2.2 Building a Classifier


``` python
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)
```

All NLTK classifiers work with feature structures, which can be simple dictionaries mapping a feature name to a feature value. In this example, we’ve used a simple bag of words model where every word is a feature name with a value of True.
 
To see which features informed our model the most, we can run this line of code:

```python
classifier.show_most_informative_features()
```

```
Most Informative Features
        no = True                neg : pos    =     20.6 : 1.0
    awesome = True               pos : neg    =     18.7 : 1.0
    headache = True              neg : pos    =     18.0 : 1.0
   beautiful = True              pos : neg    =     14.2 : 1.0
        love = True              pos : neg    =     14.2 : 1.0
          Hi = True              pos : neg    =     12.7 : 1.0
        glad = True              pos : neg    =      9.7 : 1.0
       Thank = True              pos : neg    =      9.7 : 1.0
         fan = True              pos : neg    =      9.7 : 1.0
        lost = True              neg : pos    =      9.3 : 1.0
```

### 2.3 Classification

Just to see that our model works, let's try the classifier out with a positive example: 

```python
example1 = "this workshop is awesome."

print(classifier.classify(format_sentence(example1)))
```

```
'pos'
```

Now for a negative example:

``` python
example2 = "this workshop is awful."

print(classifier.classify(format_sentence(example2)))
```

```
'neg'
```
### 2.4 Accuracy

Now, there's no point in building a model if it doesn't work well. Luckily, once again, nltk comes to the rescue with a built in feature that allows us find the accuracy of our model.

``` python
from nltk.classify.util import accuracy
print(accuracy(classifier, test))
```

``` 
0.9562326869806094
```

Turns out it works decently well!

But it could be better! I think we can agree that the data is kind of messy - there are typos, abbreviations, grammatical errors of all sorts... So how do we handle that? Can we handle that? 

## 2.0  Information Extraction

Information Extraction is the process of acquiring meaning from text in a computational manner. 

### 2.1 Data Forms

#### 2.1.1 Structured Data

Structured Data is when there is a regular and predictable organization of entities and relationships.

#### 2.1.2 Unstructured Data

Unstructured data, as the name suggests, assumes no organization. This is the case with most written textual data. 

### 2.2 What is Information Extraction?

With that said, information extraction is the means by which you acquire structured data from a given unstructured dataset. There are a number of ways in which this can be done, but generally, information extraction consists of searching for specific types of entities and relationships between those entities. 

An example is being given the following text, 

```
Martin received a 98% on his math exam, whereas Jacob received a 84%. Eli, who also took the same test, received an 89%. Lastly, Ojas received a 72%.
```
This is clearly unstructured. It requires reading for any logical relationships to be extracted. Through the use of information extraction techniques, however, we could output structured data such as the following: 

```
Name     Grade
Martin   98
Jacob    84
Eli      89
Ojas     72
```

## 3.0 Chunking

Chunking is used for entity recognition and segments and labels multitoken sequences. This typically involves segmenting multi-token sequences and labeling them with entity types, such as 'person', 'organization', or 'time'. 

### 3.1 Noun Phrase Chunking

Noun Phrase Chunking, or NP-Chunking, is where we search for chunks corresponding to individual noun phrases.

We can use nltk, as is the case most of the time, to create a chunk parser. We begin with importing nltk and defining a sentence with its parts-of-speeches tagged (which we covered in the previous tutorial). 

``` python
import nltk 
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]
```

Next, we define the tag pattern of an NP chunk. A tag pattern is a sequence of part-of-speech tags delimited using angle brackets, e.g. `<DT>?<JJ>*<NN>`. This is how the parse tree for a given sentence is acquired.  
``` python
pattern = "NP: {<DT>?<JJ>*<NN>}" 
```

Finally we create the chunk parser with the nltk RegexpParser() class. 
``` python
NPChunker = nltk.RegexpParser(pattern) 
```

And lastly, we actually parse the example sentence and display its parse tree. 
``` python
result = NPChunker.parse(sentence) 
result.draw()
```

## 4.0 Named Entity Extraction

Named entities are noun phrases that refer to specific types of individuals, such as organizations, people, dates, etc. Therefore, the purpose of a named entity recognition (NER) system is to identify all textual mentions of the named entities.

### 4.1 spaCy

In the following exercise, we'll build our own named entity recognition system with the Python module `spaCy`, a Python module commonly used for Natural Language Processing in industry. 

``` python
import spacy
import pandas as pd
```

Using spaCy, we'll load the built-in English tokenizer, tagger, parser, NER and word vectors. We indicate this with the parameter `'en'`:

``` python
nlp = spacy.load('en')
```

We need an example to actually process, so below is some text from Columbia's website:  

``` python
review = "Columbia University was founded in 1754 as King's College by royal charter of King George II of England. It is the oldest institution of higher learning in the state of New York and the fifth oldest in the United States. Controversy preceded the founding of the College, with various groups competing to determine its location and religious affiliation. Advocates of New York City met with success on the first point, while the Anglicans prevailed on the latter. However, all constituencies agreed to commit themselves to principles of religious liberty in establishing the policies of the College. In July 1754, Samuel Johnson held the first classes in a new schoolhouse adjoining Trinity Church, located on what is now lower Broadway in Manhattan. There were eight students in the class. At King's College, the future leaders of colonial society could receive an education designed to 'enlarge the Mind, improve the Understanding, polish the whole Man, and qualify them to support the brightest Characters in all the elevated stations in life.'' One early manifestation of the institution's lofty goals was the establishment in 1767 of the first American medical school to grant the M.D. degree."
```

With this example in mind, we feed it into the tokenizer.

``` python
doc = nlp(review)
```

Going along the process of named entity extraction, we begin by segmenting the text, i.e. splitting it into a list of sentences. 

``` python
sentences = [sentence.orth_ for sentence in doc.sents] # list of sentences
print("There were {} sentences found.".format(len(sentences)))
```

And we get: 
```
There were 9 sentences found.
```

Now, we go a step further, and count the number of nounphrases by taking advantage of chunk properties.

``` python
nounphrases = [[np.orth_, np.root.head.orth_] for np in doc.noun_chunks]
print("There were {} noun phrases found.".format(len(nounphrases)))
```

And we get:

```
There were 54 noun phrases found.
```

Lastly, we achieve our final goal: entity extraction. 

``` python
entities = list(doc.ents) # converts entities into a list
print("There were {} entities found".format(len(entities)))
```
And we get: 

```
There were 22 entities found
```

So now, we can turn this into a DataFrame for better visualization: 

``` python
orgs_and_people = [entity.orth_ for entity in entities if entity.label_ in ['ORG','PERSON']]
pd.DataFrame(orgs_and_people)
```

Unsurprisingly, Columbia University is an entity, along with other names like King's College and Samuel Johnson. 
```
0  Columbia University      
1  King's College           
2  King George II of England
3  Samuel Johnson           
4  Trinity Church           
5  King's College 
```

In summary, named entity extraction typically follows the process of sentence segmentation, noun phrase chunking, and, finally, entity extraction. 

### 4.2 nltk

Next, we'll work through a similar example as before, this time using the nltk module to extract the named entities through the use of chunk parsing. As always, we begin by importing our needed modules and example:  

``` python
import nltk
import re
content = "Starbucks has not been doing well lately"
```

Then, as always, we tokenize the sentence and follow up with parts-of-speech tagging. 
``` python
tokenized = nltk.word_tokenize(content)
tagged = nltk.pos_tag(tokenized)
print(tagged)
```

Great, now we've got something to work with! 
``` 
[('Starbucks', 'NNP'), ('has', 'VBZ'), ('not', 'RB'), ('been', 'VBN'), ('doing', 'VBG'), ('well', 'RB'), ('lately', 'RB')]
```

So we take this POS tagged sentence and feed it to the `nltk.ne_chunk()` method. This method returns a nested Tree object, so we display the content with namedEnt.draw(). 

``` python
namedEnt = nltk.ne_chunk(tagged)
namedEnt.draw()
```

Now, if you wanted to simply get the named entities from the namedEnt object we created, how do you think you would go about doing so?

## 5.0 Relation Extraction 

Once we have identified named entities in a text, we then want to analyze for the relations that exist between them. This can be performed using either rule-based systems, which typically look for specific patterns in the text that connect entities and the intervening words, or using machine learning systems that typically attempt to learn such patterns automatically from a training corpus.


### 5.1 Rule-Based Systems

In the rule-based systems approach, we look for all triples of the form (X, a, Y), where X and Y are named entities and a is the string of words that indicates the relationship between X and Y. Using regular expressions, we can pull out those instances of a that express the relation that we are looking for. 

In the following code, we search for strings that contain the word "in". The special regular expression `(?!\b.+ing\b)` allows us to disregard strings such as `success in supervising the transition of`, where "in" is followed by a gerund. 

``` python
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.relextract.extract_rels('ORG', 'LOC', doc,corpus='ieer', pattern = IN):
         print (nltk.sem.relextract.rtuple(rel))
```

And so we get: 

```
[ORG: 'WHYY'] 'in' [LOC: 'Philadelphia']
[ORG: 'McGlashan &AMP; Sarrail'] 'firm in' [LOC: 'San Mateo']
[ORG: 'Freedom Forum'] 'in' [LOC: 'Arlington']
[ORG: 'Brookings Institution'] ', the research group in' [LOC: 'Washington']
[ORG: 'Idealab'] ', a self-described business incubator based in' [LOC: 'Los Angeles']
[ORG: 'Open Text'] ', based in' [LOC: 'Waterloo']
[ORG: 'WGBH'] 'in' [LOC: 'Boston']
[ORG: 'Bastille Opera'] 'in' [LOC: 'Paris']
[ORG: 'Omnicom'] 'in' [LOC: 'New York']
[ORG: 'DDB Needham'] 'in' [LOC: 'New York']
[ORG: 'Kaplan Thaler Group'] 'in' [LOC: 'New York']
[ORG: 'BBDO South'] 'in' [LOC: 'Atlanta']
[ORG: 'Georgia-Pacific'] 'in' [LOC: 'Atlanta']
```

Note that the X and Y named entitities types all match with one another! Object type matching is an important and required part of this process. 

### 5.2 Machine Learning

We won't be going through an example of a machine learning based entity extraction algorithm, but it's important to note the different machine learning algorithms that can be implemented to accomplish this task of relation extraction. 

Most simply, Logistic Regression can be used to classify the objects that relate to one another. But additionally, algorithms like Suport Vector Machines and Random Forest could also accomplish the job. Which algorithm you ultimately choose depends on which outperforms in terms of speed and accuracy.

In summary, it's important to note that while these algorithms will likely have high accurate rates, labeling thousands of relations (and entities!) is incredibly expensive.   
 

## 6.0 Sentiment Analysis

As we saw in the previous tutorial, sentiment analysis refers to the use of text analysis and statistical learning to identify and extract subjective information in textual data. For our last exercise in this tutorial, we'll introduce and use linear models in the context of a sentiment analysis problem.

### 6.1 Loading the Data

First, we begin by loading the data. Since we'll be using data available online, we'll use the urllib module to avoid having to manually download any data.

``` python
import urllib.request
```

So then we'll define the test and training data URLs to variables, as well as filenames for each of those datasets.

``` python
test_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
train_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

test_file = 'test_data.csv'
train_file = 'train_data.csv'
```

Using the links and filenames from above, we'll officially download the data using the urlib.request.urlretrieve method. 

```
test_data_f = urllib.request.urlretrieve(test_url, test_file)
train_data_f = urllib.request.urlretrieve(train_url, train_file)
```

Now that we've downloaded our datasets, we can load them into pandas dataframes. First for the test data:

``` python
import pandas as pd

test_data_df = pd.read_csv(test_file, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
```

Next for training data: 
``` python
train_data_df = pd.read_csv(train_file, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]
```

Just to see how the dataframe looks, let's call the .head() method on both dataframes. 

``` python
test_data_df.head()
```

And we get: 

```
                                                Text
0  " I don't care what anyone says, I like Hillar...
1                  have an awesome time at purdue!..
2  Yep, I'm still in London, which is pretty awes...
3  Have to say, I hate Paris Hilton's behavior bu...
4                            i will love the lakers.
```


``` python
 train_data_df.head()
```

And we get:

``` 
   Sentiment                                               Text
0          1            The Da Vinci Code book is just awesome.
1          1  this was the first clive cussler i've ever rea...
2          1                   i liked the Da Vinci Code a lot.
3          1                   i liked the Da Vinci Code a lot.
4          1  I liked the Da Vinci Code but it ultimatly did...
```

### 6.2 Preparing the Data

To implement our bag-of-words linear classifier, we need our data in a format that allows us to feed it in to the classifer. Using sklearn.feature_extraction.text.CountVectorizer in the Python scikit learn module, we can convert the text documents to a matrix of token counts. So first, we import all the needed modules: 

``` python
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer
```

We need to remove punctuations, lowercase, remove stop words, and stem words. All these steps can be directly performed by CountVectorizer if we pass the right parameter values. We can do this as follows. 


We first create a stemmer, using the Porter Stemmer implementation.

``` python
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = [stemmer.stem(item) for item in tokens]
    return(stemmed)
```

Here, we have our tokenizer, which removes non-letters and stems:

``` python
def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return(stems)
```

Here we init the vectoriser with the CountVectorizer class, making sure to pass our tokenizer and stemmers as parameters, remove stop words, and lowercase all characters.

``` python
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)
```

Next, we use the fit_transform() method to transform our corpus data into feature vectors. Since the input needed is a list of strings, we concatenate all of our training and test data. 

``` python
features = vectorizer.fit_transform(
    train_data_df.Text.tolist() + test_data_df.Text.tolist())
```

Here, we're simply converting the features to an array for easier use.  
``` python
features_nd = features.toarray()
```

### 6.3 Linear Classifier

Finally, we begin building our classifier. Earlier we learned what a bag-of-words model. Here, we'll be using a similar model, but with some modifications. To refresh your mind, this kind of model simplifies text to a multi-set of terms frequencies. 

So first we'll split our training data to get an evaluation set. As we mentioned before, we'll use cross validation to split the data. sklearn has a built-in method that will do this for us. All we need to do is provide the data and assign a training percentage (in this case, 75%).

``` python
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd[0:len(train_data_df)], 
        train_data_df.Sentiment,
        train_size=0.85, 
        random_state=1234)
```

Now we're ready to train our classifier. We'll be using Logistic Regression to model this data. Once again, sklearn has a built-in model for you to use, so we begin by importing the needed modules and calling the class.  

``` python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
```

And as always, we need actually do the training, so we call the `.fit()` method on our data. 
``` python
log_model = log_model.fit(X=X_train, y=y_train)
```

Now we use the classifier to label the evaluation set we created earlier:

``` python
y_pred = log_model.predict(X_test)
```

You can see that this array of labels looks like: 

```
array([0, 1, 0, ..., 0, 1, 0])
```

### 6.4 Accuracy

In sklearn, there is a function called sklearn.metrics.classification_report which calculates several types of predictive scores on a classification model. So here we check out how exactly our model is performing:

``` python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

And we get: 

```
                 precision    recall  f1-score   support

              0       0.98      0.99      0.98       467
              1       0.99      0.98      0.99       596

    avg / total       0.98      0.98      0.98      1063
```
where precision, recall, and f1-score are the accuracy values discussed in the section 1.6. Support is the number of occurrences of each class in y_true and x_true.


### 6.5 Retraining 

Finally, we can re-train our model with all the training data and use it for sentiment classification with the original unlabeled test set. 

So we repeat the process from earlier, this time with different data:
``` python
log_model = LogisticRegression()
log_model = log_model.fit(X=features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
test_pred = log_model.predict(features_nd[len(train_data_df):])
```

So again, we can see what the predictions look: 

```
array([1, 1, 1, ..., 1, 1, 0])
```

And lastly, let's actually look at our predictions! Using the random module to select a random sliver of the data we predicted on, we'll print the results.  
``` python
import random
spl = random.sample(range(len(test_pred)), 10)
for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    print (sentiment, text)
```

Recall that 0 indicates a negative sentence and 1 indicates a positive:

```
0 harvard is dumb, i mean they really have to be stupid to have not wanted her to be at their school.
0 I've been working on an article, and Antid Oto has been, er, so upset about the shitty Harvard plagiarizer that he hasn't been able to even look at keyboards.
0 I hate the Lakers...
0 Boston SUCKS.
0 stupid kids and their need for Honda emblems):
1 London-Museums I really love the museums in London because there are a lot for me to see and they are free!
0 Stupid UCLA.
1 as title, tho i hate london, i did love alittle bit about london..
1 I love the lakers even tho Trav makes fun of me.
1 that i love you aaa lllooootttttt...
```
