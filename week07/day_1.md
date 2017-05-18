
## 1.0 Background

Recall in data structures learning about the different types of tree structures - binary, red black, and splay trees. In tree based modeling, we work off these structures for classification prediction. 

Tree based machine learning is great because it's incredibly accurate and stable, as well as easy to interpret. Despite being a linear model, tree based models map non-linear relationships well. The general structure is as follows: 


## 2.0 Decision Trees

Decision trees are a type of supervised learning algorithm used in classification that works for both categorical and continuous input/output variables. This typle of model includes structures with nodes which represent tests on attributes and the end nodes (leaves) of each branch represent class labels. Between these nodes are what we call edges, which represent a 'decision' that separates the data from the previous node based on some criteria. 

![alt text](https://www.analyticsvidhya.com/wp-content/uploads/2016/04/dt.png "Logo Title Text 1")

Looks familiar, right? 

### 2.1 Nodes

As mentioned above, nodes are an important part of the structure of Decision Trees. In this section, we'll review different types of nodes.

#### 2.1.1 Root Node

The root node is the node at the very top. It represents an entire population or sample because it has yet to be divided by any edges. 

#### 2.1.2 Decision Node

Decision Nodes are the nodes that occur between the root node and leaves of your decision tree. It's considered a decision node because it's a resulting node of an edge that then splits once again into either more decision nodes, or the leaves.

#### 2.1.3 Leaves/Terminal Nodes

As mentioned before, leaves are the final nodes at the bottom of the decision tree that represent a class label in classification. They're also called <i>terminal nodes</i> because more nodes do not split off of them. 

#### 2.1.4 Parent and Child Nodes

A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.

### 2.2 Pros & Cons

#### 2.2.1 Pros

1. Easy to Understand: Decision tree output is fairly easy to understand since it doesn't require any statistical knowledge to read and interpret them. Its graphical representation is very intuitive and users can easily relate their hypothesis.

2. Useful in Data exploration: Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. With the help of decision trees, we can create new variables / features that has better power to predict target variable. You can refer article (Trick to enhance power of regression model) for one such trick.  It can also be used in data exploration stage. For example, we are working on a problem where we have information available in hundreds of variables, there decision tree will help to identify most significant variable.

3. Less data cleaning required: It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree.

4. Data type is not a constraint: It can handle both numerical and categorical variables.

5. Non Parametric Method: Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.

#### 2.2.2 Cons

1. Over fitting: Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning (discussed in detailed below).

2. Not fit for continuous variables: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.

### 2.1 Decision Trees in R

We begin by loading the required libraries:

``` R
library(rpart)
library(party)
library(partykit)
```

In this example, we'll be working with credit scores, so we load the needed data:

``` R
library(caret)
data(GermanCredit)
```

As always, we split the data into training and test data:

``` R
inTrain <- runif(nrow(GermanCredit)) < 0.2
```

Now, we can build the actual decision tree with `rpart()`. 

``` R
dt <- rpart(Class ~ Duration + Amount + Age, method="class",data=GermanCredit[inTrain,])
```

But let's check out the internals of this decision tree. Let's plot it using the `plot()` function: 

``` R
plot(dt)
text(dt)
```

And to prettify it, we type:

``` R
plot(as.party(dt))
```

We obviously want to have the best machine learning models for our data. To make sure we have optimal models, we can calculate the complexity parameter:

``` R
printcp(dt)
```

``` 
Classification tree:
rpart(formula = Class ~ Duration + Amount + Age, data = GermanCredit[inTrain, 
    ], method = "class")

Variables actually used in tree construction:
[1] Age      Amount   Duration

Root node error: 61/199 = 0.30653

n= 199 

        CP nsplit rel error xerror    xstd
1 0.065574      0   1.00000 1.0000 0.10662
2 0.057377      1   0.93443 1.0984 0.10929
3 0.032787      3   0.81967 1.0656 0.10846
4 0.016393      5   0.75410 1.0656 0.10846
5 0.010000      6   0.73770 1.1967 0.11145
```

In this output, the rows show result for trees with different numbers of nodes. The column `xerror` represents the cross-validation error and the `CP` represents the complexity parameter. 

### 2.2 Pruning Decision Trees

Decision Tree pruning is a technique that reduces the size of decision trees by removing sections (nodes) of the tree that provide little power to classify instances. This is great because it reduces the complexity of the final classifier, which results in increased predictive accuracy by reducing overfitting. 

Ultimately, our aim is to reduce the cross-validation error. First, we index with the smallest complexity parameter:

``` R
m <- which.min(dt$cptable[, "xerror"])
```

To get the optimal number of splits, we:

``` R
dt$cptable[m, "nsplit"]
```

Now, we choose the corresponding complexity parameter:

``` R
dt$cptable[m, "CP"]
```

Now, we're ready for pruning:

``` R
p <- prune(dt, cp = dt$cptable[which.min(dt$cptable[, "xerror"]), "CP"])
plot(as.party(p))
```

### 2.3 Prediction

Now we can get into the prediction portion. We can use the `predict()` function in R:

``` R
pred <- predict(p, GermanCredit[-inTrain,], type="class")
```

Now let's get the confusion matrix:

``` R
table(pred=pred, true=GermanCredit[-inTrain,]$Class)
```

## 3.0 Random Forests

Recall the ensemble learning method from the Optimization lecture. Random Forests are an ensemble learning method for classification and regression. It works by combining individual decision trees through bagging. This allows us to overcome overfitting. 

### 3.1 Algorithm

First, we create many decision trees through bagging. Once completed, we inject randomness into the decision trees by allowing the trees to grow to their maximum sizes, leaving them unpruned. 

We make sure that each split is based on randomly selected subset of attributes, which reduces the correlation between different trees. 

Now we get into the random forest by voting on categories by majority. We begin by splitting the training data into K bootstrap samples by drawing samples from training data with replacement. 

Next, we estimate individual trees t<sub>i</sub> to the samples and have every regression tree predict a value for the unseen data. Lastly, we estimate those predictions with the formula:

![alt text](https://github.com/lesley2958/ml-tree-modeling/blob/master/rf-pred.png?raw=true "Logo Title Text 1")

where y&#770; is the response vector and x = [x<sub>1</sub>,...,x<sub>N</sub>]<sup>T</sup> &isin; X as the input parameters. 


### 3.2 Advantages

Random Forests allow us to learn non-linearity with a simple algorithm and good performance. It's also a fast training algorithm and resistant to overfitting.

What's also phenomenal about Random Forests is that increasing the number of trees decreases the variance without increasing the bias, so the worry of the variance-bias tradeoff isn't as present. 

The averaging portion of the algorithm also allows the real structure of the data to reveal. Lastly, the noisy signals of individual trees cancel out. 

### 3.3 Limitations 

Unfortunately, random forests have high memory consumption because of the many tree constructions. There's also little performance gain from larger training datasets. 

### 3.4 Random Forests in R

We begin by loading the required libraries and data: 

``` R
library(randomForest)
library(caret)
data(GermanCredit)
```

Once again, we split our data into training and test data.
``` R
inTrain <- runif(nrow(GermanCredit)) < 0.2
```

Now we can run the random forest algorithm on this data:

``` R
rf <- randomForest(Class ~ .,
	data=GermanCredit[inTrain,],
	ntree=100)
```

Let's take a look at the estimated error across the number of decision trees. The dotted lines represent the individual errors and the solid black line represents the overall error. 

``` R
library(rf)
```


Let's check out the confusion matrix:

``` R
rf$confusion
```

And finally, let's get back to predicting: 

``` R
pred <- predict(rf, newdata=GermanCredit[-inTrain,])
table(pred=pred, true=GermanCredit$Class[-inTrain])
```

### 3.5 Variable Importance


![alt text](https://github.com/lesley2958/ml-tree-modeling/blob/master/var%20importance.png?raw=true "Logo Title Text 1")

``` R
rf2 <- randomForest(Class ~ .,
	data=GermanCredit, #with full dataset
	ntree=100,
	importance=TRUE)
```

Now, let's plot it. `type` chooses the importance metirx (1 denotes the mean decrease in accuracy if the variable were randomly permuted). `n.var` denotes the number of variables:

``` R
varImpPlot(rf2, type=1, n.var=5)
```
