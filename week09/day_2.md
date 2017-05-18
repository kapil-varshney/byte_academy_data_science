## 1.0 Background

Working off of the knowledge we gained from the [Intro NLP](learn.adicu.com/nlp), [Intermediate NLP](learn.adicu.com/intermediate-nlp), and [Deep Learning](learn.adicu.com/dl-python) workshops, we'll spend this tutorial covering the basics of how these fields intersect. 

With that said, we'll begin by overviewing some of the fundamental concepts of this intersection.  

### 1.1 Vector Space Models

A vector space search involves converting documents into vectors, where each dimension within the vectors represents a term. VSMs represent embedded words in a continuous vector space where semantically similar words are mapped to nearby points. 


#### 1.1.1 Distributional Hypothesis

Vector Space Model methods depend highly on the Distributional Hypothesis, which states that words that appear in the same contexts share semantic meaning. The different approaches that leverage this principle can be divided into two categories: count-based methods and predictive methods. 


#### 1.1.2 Stemming & Stop Words

To begin implementing a Vector Space Model, first you must get all terms within documents and clean them. A tool you will likely utilize for this process is a stemmer, takes words and reduces them to the unchanging portion. Words which have a common stem often have similar meanings, which is why you'd likely want to utilize a stemmer.

We will also want to remove any stopwords, such as a, am, an, also ,any, and, etc. Stop words have little value in this search so it's better that we just get rid of them. 

Just as we've seen before, this is what that code might look like:

``` python
from nltk import PorterStemmer
stemmer = PorterStemmer()

def removeStopWords(stopwords, list):
    return([word for word in list if word not in stopwords])


def tokenise(words, string):
    string = self.clean(string)
    words = string.split(" ")
    return ([self.stemmer.stem(word,0,len(word)-1) for word in words])
```

#### 1.1.3 Map Keywords to Vector Dimensions

Next, we'll map the vector dimensions to keywords using a dictionary.

``` python
def getVectorKeywordIndex(documentList):
        """create the keyword associated to the position of the elements within the document vectors"""

        # Maps documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)

        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
                vectorIndex[word]=offset
                offset+=1
        return(vectorIndex)  
```

#### 1.1.4 Map Document Strings to Vectors

We use the Simple Term Count Model to map the documents 

``` python
def makeVector(wordString):

        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
                vector[self.vectorKeywordIndex[word]] += 1; # Use Simple Term Count Model
        return(vector)
```

#### 1.1.5 Find Related Documents

We now have the ability to find related documents. We can test if two documents are in the same concept space by looking at the the cosine of the angle between the document vectors. We use the cosine of the angle as a metric for comparison. If the cosine is 1 then the angle is 0° and hence the vectors are parallel (and the document terms are related). If the cosine is 0 then the angle is 90° and the vectors are perpendicular (and the document terms are not related).

We calculate the cosine between the document vectors in python using scipy.

``` python
def cosine(vector1, vector2):
        """related documents j and q are in the concept space by comparing the vectors:
                cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
        return(float(dot(vector1,vector2) / (norm(vector1) * norm(vector2))))
```

#### 1.1.6 Search Vector Space

In order to perform a search across keywords we need to map the keywords to the vector space. We create a temporary document which represents the search terms and then we compare it against the document vectors using the same cosine measurement mentioned for relatedness.

``` python
def search(searchList, documentVectors):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in documentVectors]
        ratings.sort(reverse=True)
        return(ratings)
```

### 1.2 Autoencoders

Autoencoders are a type of neural network designed for dimensionality reduction; in other words, representing the same information with fewer numbers. A wide range of autoencoder architectures exist, including Denoising Autoencoders, Variational Autoencoders, or Sequence Autoencoders. 

The basic premise is simple — we take a neural network and train it to spit out the same information it’s given. By doing so, we ensure that the activations of each layer must, by definition, be able to represent the entirety of the input data. 

#### 1.2.1 How do Autencoders work? 

If each layer is the same size as the input, this becomes trivial, and the data can simply be copied over from layer to layer to the output. But if we start changing the size of the layers, the network inherently learns a new way to represent the data. If the size of one of the hidden layers is smaller than the input data, it has no choice but to find some way to compress the data.

And that’s exactly what an autoencoder does. The network starts out by “compressing” the data into a lower-dimensional representation, and then converts it back to a reconstruction of the original input. If the network converges properly, it will be a more compressed version of the data that encodes the same information. 

It’s often helpful to think about the network as an “encoder” and a “decoder”. The first half of the network, the encoder, takes the input and transforms it into the lower-dimensional representation. The decoder then takes that lower-dimensional representation and converts it back to the original image (or as close to it as it can get). The encoder and decoder are still trained together, but once we have the weights, we can use the two separately — maybe we use the encoder to generate a more meaningful representation of some data we’re feeding into another neural network, or the decoder to let the network generate new data we haven’t shown it before.

#### 1.2.2 Why are Autoencoders Important?

Because some of your features may be redundant or correlated, you can end up with wasted processing time and an overfitted model - Autoencoders help us avoid that.

### 1.3 Word Embeddings

Like an autoencoder, this type of model learns a vector space embedding for some data. For tasks like speech recognition we know that all the information required to successfully perform the task is encoded in the data. However, natural language processing systems traditionally treat words as discrete atomic symbols, and therefore 'cat' may be represented as Id537 and 'dog' as Id143. These encodings are arbitrary, and provide no useful information to the system regarding the relationships that may exist between the individual symbols. 

This means that the model can leverage very little of what it has learned about 'cats' when it is processing data about 'dogs' (such that they are both animals, four-legged, pets, etc.). Representing words as unique, discrete IDs furthermore leads to data sparsity, and usually means that we may need more data in order to successfully train statistical models. However, using vector representations can overcome some of these obstacles.


### 1.4 Word2Vec

Word2vec is an efficient predictive model for learning word embeddings from raw text. It comes in two models: the <b>Continuous Bag-of-Words model (CBOW)</b> and the <b>Skip-Gram model</b>. Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words. 

The easiest way to think about word2vec is that it figures out how to place words on a graph in such a way that their location is determined by their meaning. In other words, words with similar meanings will be clustered together. More interestingly, though, is that the gaps and distances on the graph have meaning as well. So if you go to where “king” appears, and move the same distance and direction between “man” and “woman”, you end up where “queen” appears. And this is true of all kinds of semantic relationships! You can look at this visualization [here](https://www.tensorflow.org/versions/master/images/linear-relationships.png).

Put more mathematically, you can think of this relationship as: 

```
[king] - [man] + [woman] ~= [queen]
```

### 1.5 Skip-gram Model


#### 1.5.1 Inputs

The input of the skip-gram model is a single word, w<sub>n</sub>. For example, in the following sentence:

``` 
I drove my car to school.
```

"car" could be the input, or "school". 


#### 1.5.2 Outputs

The output of a skip-gram model is the words in w<sub>n</sub>'s context. Going along with the example from before, the output would be:

``` 
{"I","drove","my","to","school"}
```

This output is defined as {w<sub>O</sub>,1 , ... , w<sub>O</sub>,C }, where C is the word window size that you define. 


### 1.6 Continuous Bag-of-Words model 

#### 1.6.1 Inputs

This input is defined as {w<sub>O</sub>,1 , ... , w<sub>O</sub>,C }, where C is the word window size that you define. For example, the input could be:

``` 
{"I","drove","my","to","school"}
```

#### 1.6.2 Outputs 

The output of the neural network will be w<sub>i</sub>. Hence you can think of the task as "predicting the word given its context". Note that the number of words we use depends on your setting for the window size.

``` 
I drove my car to school.
```

"car" could be the output.

### 1.7 Fine-Tuning

Fine-Tuning refers to the technique of initializing a network with parameters from another task (such as an unsupervised training task), and then updating these parameters based on the task at hand. For example, NLP architecture often use pre-trained word embeddings like word2vec, and these word embeddings are then updated during training based for a specific task like Sentiment Analysis.

### 1.8 Glove

GloVe is an unsupervised learning algorithm for obtaining vector representations (embeddings) for words. GloVe vectors serve the same purpose as word2vec but have different vector representations due to being trained on co-occurrence statistics.

### 1.9 Tensorflow

TensorFlow is an open source library for numerical computation using data flow graphs. We'll be using Tensorflow to implement a CNN for Text Classification later.

#### 1.9.1 Tensors

Tensors are geometric objects that describe linear relations between geometric vectors, scalars, and other tensors. Elementary examples of such relations include the dot product, the cross product, and linear maps. Geometric vectors, often used in physics and engineering applications, and scalars themselves are also tensors.

#### 1.9.2 Sessions

In TensorFlow, a Session is the environment you execute graph operations in, which contains the state of Variables and queues. Each session operates on a single graph. 

#### 1.9.3 Graphs

A Graph contains operations and tensors. You can use multiple graphs in your program, but most programs only need a single graph. You can use the same graph in multiple sessions, but not multiple graphs in one session. TensorFlow always creates a default graph, but you may also create a graph manually and set it as the new default. Explicitly creating sessions and graphs ensures that resources are released properly when you no longer need them.

#### 1.9.4 Summaries

TensorFlow has summaries, which allow you to keep track of and visualize various quantities during training and evaluation. For example, you probably want to keep track of how your loss and accuracy evolve over time. You can also keep track of more complex quantities, such as histograms of layer activations.


#### 1.9.5 Checkpoints

Another TensorFlow feature you might use is checkpointing, which is when you save the parameters of your model to restore them later on. Checkpoints can be used to continue training at a later point, or to pick the best parameters setting using early stopping. 

## 3.0 gensim and nltk

`gensim` provides a Python implementation of Word2Vec that works great in conjunction with NLTK corpora. The model takes a list of sentences as input, where each sentence is expected to be a list of words. 

Let's begin exploring the capabilities of these two modules and import them as needed:

``` python
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
```

Next, we'll load the Word2Vec model on three different NLTK corpora for the sake of comparison. 

``` python
b = Word2Vec(brown.sents())
mr = Word2Vec(movie_reviews.sents())
t = Word2Vec(treebank.sents())
```

Here we'll use the word2vec method `most_similar` to find words associated with a given word. We'll call this method on all three corpora and see how they differ. Note that we’re comparing a wide selection of text from the brown corpus with movie reviews and financial news from the treebank corpus.

``` python
print(b.most_similar('money', topn=5))
print(mr.most_similar('money', topn=5))
print(t.most_similar('money', topn=5))
```

And we get:

``` 
[(u'care', 0.9144914150238037), (u'chance', 0.9035866856575012), (u'job', 0.8981726765632629), (u'trouble', 0.8752247095108032), (u'everything', 0.8740494251251221)]

[(u'him', 0.783672571182251), (u'chance', 0.7683249711990356), (u'someone', 0.76824951171875), (u'sleep', 0.7554738521575928), (u'attention', 0.737582802772522)]

[(u'federal', 0.9998856782913208), (u'all', 0.9998772144317627), (u'new', 0.9998764991760254), (u'only', 0.9998745918273926), (u'companies', 0.9998691082000732)]
```

The results are vastly different! Let's try a different example: 
``` python
print(b.most_similar('great', topn=5))
print(mr.most_similar('great', topn=5))
print(t.most_similar('great', topn=5))
```

Again, still pretty different!
```
[(u'common', 0.8809183835983276), (u'experience', 0.8797307014465332), (u'limited', 0.841762900352478), (u'part', 0.8300108909606934), (u'history', 0.8120908737182617)]

[(u'nice', 0.8149471282958984), (u'decent', 0.8089801073074341), (u'wonderful', 0.8058308362960815), (u'good', 0.7910960912704468), (u'fine', 0.7873961925506592)]

[(u'out', 0.9992268681526184), (u'what', 0.9992024302482605), (u'if', 0.9991806745529175), (u'we', 0.9991364479064941), (u'not', 0.999133825302124)]
```

And lastly,

``` python
print(b.most_similar('company', topn=5))
print(mr.most_similar('company', topn=5))
print(t.most_similar('company', topn=5))
```

It’s pretty clear from these examples that the semantic similarity of words can vary greatly depending on the textual context. 


