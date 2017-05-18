## 1.0 Introduction

Naive Bayes works on Bayes Theorem of probability to predict the class of a given data point. Naive Bayes is extremely fast compared to other classification algorithms and works with an assumption of independence among predictors. 

The Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

### 1.1 Bayes Theorem

Recall Bayes Theorem, which provides a way of calculating the posterior probability. Its formula is as follows:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/bayes.png?raw=true "Logo Title Text 1")


### 1.2 Overview

Let's go through an example of how the Naive Bayes Algorithm works. We'll go through a classification problem that determines whether a sports team will play or not based on the weather. 

Let's load the module data:

``` python
import pandas as pd
f1 = pd.read_csv("./weather.csv")
```

#### 1.2.1 Frequency Table

The first actual step of this process is converting the dataset into a frequency table. Using the `groupby()` function, we get the frequencies:

``` python
df = f1.groupby(['Weather','Play']).size()
```

Now let's split the frequencies by weather and yes/no. Let's start with the three weather frequencies:

``` python
df2 = f1.groupby('Weather').count()
```

Now let's get the frequencies of yes and no:

``` python
df1 = f1.groupby('Play').count()
```

#### 1.2.2 Likelihood Table


Next, you would create a likelihood table by finding the probabilites of each weather condition and yes/no. This will require that we add a new column that takes the play frequency and divides it by the total data occurances. 


``` python
df1['Likelihood'] = df1['Weather']/len(f1)
df2['Likelihood'] = df2['Play']/len(f1)
```

This gets us a dataframe that looks like:

```
          Play  Likelihood
Weather                   
Overcast     4    0.285714
Rainy        5    0.357143
Sunny        5    0.357143
```

Now, we're able to use the Naive Bayesian equation to calculate the posterior probability for each class. The highest posterior probability is the outcome of prediction.

#### 1.2.1 Calculation

So now we need a question. Let's propose the following: "Players will play if the weather is sunny. Is this true?"

From this question, we can construct Bayes Theorem. So what's our P(A|B)? P(Yes|Sunny), which gives us:

P(Yes|Sunny) = (P(Sunny|Yes)*P(Yes))/P(Sunny)

Based off the likelihood tables we created, we just grab P(Sunny) and P(Yes). 

``` python
ps = df2['Likelihood']['Sunny']
py = df1['Likelihood']['Yes']
```

That leaves us with P(Sunny|Yes). This is the probability that the weather is sunny given that the players played that day. In `df`, we see that the total number of `yes` days under `sunny` is 3. We take this number and divide it by the total number of `yes` days, which we can get from `df`. 

``` python
psy = df['Sunny']['Yes']/df1['Weather']['Yes']
```

Now, we just have to plug these variables into bayes theorem: 

``` python
p = (psy*py)/ps
```

And we get:

```
0.59999999999999998
```

That means the answer to our original question is yes!

### 1.3 Naive Bayes Evaluation

Every classifier has pros and cons, whether that be in terms of computational power, accuracy, etc. In this section, we'll review the pros and cons of Naive Bayes.

#### 1.3.1 Pros

Naive Bayes is incredibly easy and fast in predicting the class of test data. It also performs well in multi-class prediction.

When the assumption of independence is true, the Naive Bayes classifier performs better thanother models like logistic regression. It does this, and with less need of a lot of data.

Naive Bayes also performs well with categorical input variables compared to numerical variable(s), which is why we're able to use it for text classification. For numerical variables, normal distribution must be assumed.


#### 1.3.2 Cons

If a categorical variable has a category not observed in the training data set, then model will assign a 0 probability and will be unable to make a prediction. This is referred to as “Zero Frequency”. To solve this, we can use the smoothing technique, such as Laplace estimation.

Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.


## 2.0 Naive Bayes Types

With `scikit-learn`, we can implement Naive Bayes models in Python. There are three types of Naive Bayes models, all of which we'll review in the following sections.


### 2.1 Gaussian

The Gaussian Naive Bayes Model is used in classification and assumes that features will follow a normal distribution. 

We begin an example by importing the needed modules:

``` python
from sklearn.naive_bayes import GaussianNB
import numpy as np
```

As always, we need predictor and target variables, so we assign those:

``` python
x = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])

y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
```

Now we can initialize the Gaussian Classifier:

``` python
model = GaussianNB()
```

Now we can train the model using the training sets:
``` python
model.fit(x, y)
```

Now let's try out an example:
``` python
predicted = model.predict([[1,2],[3,4]])
```

We get:
```
([3,4])
```

### 2.2 Multinomial

MultinomialNB implements the multinomial Naive Bayes algorithm and is one of the two classic Naive Bayes variants used in text classification. This classifier is suitable for classification with discrete features (such as word counts for text classification). 

The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts may also work.

First, we need some data, so we import numpy and 

``` python
import numpy as np
x = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
```

Now let's build the Multinomial Naive Bayes model: 

``` python3
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x, y)
```

Let's try an example:

``` python
print(clf.predict(x[2:3]))
```

And we get:

```
[3]
```

### 2.3 Bernoulli

Like MultinomialNB, this classifier is suitable for discrete data. BernoulliNB implements the Naive Bayes training and classification algorithms for data that is distributed according to multivariate Bernoulli distributions, meaning there may be multiple features but each one is assumed to be a binary value. 

The decision rule for Bernoulli Naive Bayes is based on

![alt text](https://github.com/lesley2958/ml-bayes/blob/master/bernoulli.png?raw=true "Logo Title Text 1")


``` python
import numpy as np
x = np.random.randint(2, size=(6, 100))
y = np.array([1, 2, 3, 4, 4, 5])
```

``` python
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(x, y)
print(clf.predict(x[2:3]))
```

And we get: 
```
[3]
```

### 2.4 Tips for Improvement

If continuous features don't have a normal distribution (recall the assumption of normal distribution), you can use different methods to convert it to a normal distribution.

As we mentioned before, if the test data set has a zero frequency issue, you can apply smoothing techniques “Laplace Correction” to predict the class.

As usual, you can remove correlated features since the correlated features would be voted twice in the model and it can lead to over inflating importance.


## 3.0 Sentiment Analysis  

So you might be asking, what exactly is "sentiment analysis"? 

Well, sentiment analysis involves building a system to collect and determine the emotional tone behind words. This is important because it allows you to gain an understanding of the attitudes, opinions and emotions of the people in your data. 

At a high level, sentiment analysis involves Natural language processing and artificial intelligence by taking the actual text element, transforming it into a format that a machine can read, and using statistics to determine the actual sentiment.

### 3.1 Preparing the Data 

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


#### 3.1.1 Training Data

Next, we'll split the labeled data we have into two pieces, one that can "train" data and the other to give us insight on how well our model is performing. The training data will inform our model on which features are most important.

``` python
training = pos[:int((.9)*len(pos))] + neg[:int((.9)*len(neg))]
```

#### 3.1.2 Test Data

We won't use the test data until the very end of this section, but nevertheless, we save the last 10% of the data to check the accuracy of our model. 
``` python
test = pos[int((.1)*len(pos)):] + neg[int((.1)*len(neg)):]
```

### 3.2 Building a Classifier


``` python
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)
```

All NLTK classifiers work with feature structures, which can be simple dictionaries mapping a feature name to a feature value. In this example, we’ve used a simple bag of words model where every word is a feature name with a value of True. This is an implementation of the Bernoulli Naive Bayes Model.
 
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

### 3.3 Classification

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
### 3.4 Accuracy

Now, there's no point in building a model if it doesn't work well. Luckily, once again, nltk comes to the rescue with a built in feature that allows us find the accuracy of our model.

``` python
from nltk.classify.util import accuracy
print(accuracy(classifier, test))
```

``` 
0.9562326869806094
```

Turns out it works decently well!

## 4.0 Joint Models

If you have input data x and want to classify the data into labels y. A <b>generative model</b> learns the joint probability distribution `p(x,y)` and a discriminative model learns the <b>conditional probability distribution</b> `p(y|x)`.

Here's an simple example of this form:
```
(1,0), (1,0), (2,0), (2, 1)
```

`p(x,y)` is
```
      y=0   y=1
     -----------
x=1 | 1/2   0
x=2 | 1/4   1/4
```

Meanwhile,
```
p(y|x) is
```
```
      y=0   y=1
     -----------
x=1 | 1     0
x=2 | 1/2   1/2
```

Notice that if you add all 4 probabilities in the first chart, they add up to 1, but if you do the same for the second chart, they add up to 2. This is because the probabilities in chart 2 are read row by row. Hence, `1+0=1` in the first row and `1/2+1/2=1` in the second.

The distribution `p(y|x)` is the natural distribution for classifying a given example `x` into a class `y`, which is why algorithms that model this directly are called discriminative algorithms. 

Generative algorithms model `p(x, y)`, which can be tranformed into `p(y|x)` by applying Bayes rule and then used for classification. However, the distribution `p(x, y)` can also be used for other purposes. For example you could use `p(x,y)` to generate likely `(x, y)` pairs.
