
# What is Machine Learning?

Machine Learning is the field where statistics and computer science overlap for prediction insights on data. What do we mean by <i>prediction</i>? Given a dataset, we generate an algorithm to <i>learn</i> what features or attributes are indicators of a certain label or prediction. These attributes or features can be anything that describes a data point, whether that's height, frequency, class, etc. 

This algorithm is chosen based on the original dataset, which you can think of as prior or historical data. When we refer to machine learning algorithms, we're referring to a function that best maps a data point's features to its label. A large part of the machine learning is spent improving this function as much as possible. 

## Hypothesis

You've likely heard of hypotheses in the context of science before; typically, it's an educated guess on what and outcome will be. In the context of machine learning, a hypothesis is the function we believe is similar to the <b>true</b> function of learning - the target function that we want to model and use as our machine learning algorithm. 

## Assumptions

In this course, we'll review different machine learning algorithms, from decision trees to support vector machines. Each is different in its methodology and results. A critical part of the process of choosing the best algorithm is considering the assumptions you can make about the particular dataset you're working with. These assumptions can include the linearity or lack of linearity, the distribution of the dataset, and much more. 

### Notation

In this course and future courses, you'll see lots of notation. I've collected a list of all the general notation you'll see:

<i>a</i>: scalar <br>
<i><b>a</b></i>: vector <br>
<i>A</i>: matrix <br>
<i>a<sub>i</sub></i>: the ith entry of <i><b>a</b></i> <br>
<i>a<sub>ij</sub></i>: the entry (i,j) of <i>A</i> <br>
<i><b>a<sup>(n)</sup></b></i>: the nth vector <i><b>a</b></i> in a dataset <br>
<i>A<sup>(n)</sup></i>: the nth matrix <i>A</i> in a dataset <br>
<i>H</i>: Hypothesis Space, the set of all possible hypotheses <br>

## Data 

As a data scientist, knowing the different forms data takes is highly important. When working on a prediction problem, the collection of data you're working with is called a <i>dataset</i>.

### Labeled vs Unlabeled Data

Generally speaking, there are two forms of data: unlabeled and labeled. Labeled data refers to data that has inputs and attached, known outputs. Unlabeled means all you have is inputs. They're denoted as follows:

- Labeled Dataset: X = {x<sup>(n)</sup> &isin; R<sup>d</sup>}<sup>N</sup><sub>n=1</sub>, Y = {y<sup>n</sup> &isin; R}<sup>N</sup><sub>n=1</sub>

- Unlabed Dataset: X = {x<sup>(n)</sup> &isin; R<sup>d</sup>}<sup>N</sup><sub>n=1</sub>

Here, X denotes a feature set containing N samples. Each of these samples is a d-dimension vector, <b>x<sup>(n)</sup></b>. Each of these dimensions is an attribute, feature, variable, or element. Meanwhile, Y is the label set. 


### Training vs Test Data

When it comes time to train your classifier or model, you're going to need to split your data into <b>testing</b> and <b>training</b> data. 

Typically, the majority of your data will go towards your training data, while only 10-25% of your data will go towards testing. 

It's important to note there is no overlap between the two. Should you have overlap or use all your training data for testing, your accuracy results will be wrong. Any classifier that's tested on the data it's training is obviously going to do very well since it will have observed those results before, so the accuracy will be high, but wrongly so. 

Furthermore, both of these sets of data must originate from the same source. If they don't, we can't expect that a model built for one will work for the other. We handle this by requiring the training and testing data to be <b>identically and independently distributed (iid)</b>. This means that the testing data show the same distribution as the training data, but again, must not overlap.

Together these two aspects of the data are known as <i>IID assumption</i>.


### Overfitting vs Underfitting

The concept of overfitting refers to creating a model that doesn't generaliz e to your model. In other words, if your model overfits your data, that means it's learned your data <i>too</i> much - it's essentially memorized it. 

This might not seem like it would be a problem at first, but a model that's just "memorized" your data is one that's going to perform poorly on new, unobserved data. 

Underfitting, on the other hand, is when your model is <i>too</i> generalized to your data. This model will also perform poorly on new unobserved data. This usually means we should increase the number of considered features, which will expand the hypothesis space. 

### Open Data 

What's open data, you ask? Simple, it's data that's freely  for anyone to use! Some examples include things you might have already heard of, like APIs, online zip files, or by scraping data!

You might be wondering where this data comes from - well, it can come from a variety of sources, but some common ones include large tech companies like Facebook, Google, Instagram. Others include large institutions, like the US government! Otherwise, you can find tons of data from all sorts of organizations and individuals. 


## Types of Learning

Generally speaking, Machine Learning can be split into three types of learning: supervised, unsupervised, and reinforcement learning. 


### Supervised Learning

This algorithm consists of a target / outcome variable (or dependent variable) which is to be predicted from a given set of predictors (independent variables). Using these set of variables, we generate a function that map inputs to desired outputs. The training process continues until the model achieves a desired level of accuracy on the training data. Examples of Supervised Learning: Regression, Decision Tree, Random Forest, Logistic Regression, etc.

All supervised learning algorithms in the Python module scikit-learn hace a `fit(X, y)` method to fit the model and a `predict(X)` method that, given unlabeled observations X, returns predicts the corresponding labels y.

### Unsupervised Learning

In this algorithm, we do not have any target or outcome variable to predict / estimate.  We can derive structure from data where we don't necessarily know the effect of the variables. Examples of Unsupervised Learning: Clustering, Apriori algorithm, K-means.


### Reinforcement Learning

Using this algorithm, the machine is trained to make specific decisions. It works this way: the machine is exposed to an environment where it trains itself continually using trial and error. This machine learns from past experience and tries to capture the best possible knowledge to make accurate business decisions. Example of Reinforcement Learning: Markov Decision Process.

### Subfields

Though Machine Learning is considered the overarching field of prediction analysis, it's important to know the distinction between its different subfields.

#### Natural Language Processing

Natural Language Processing, or NLP, is an area of machine learning that focuses on developing techniques to produce machine-driven analyses of textual data.


#### Computer Vision

Computer Vision is an area of machine learning and artificial intelligence that focuses on the analysis of data involving images.


#### Deep Learning

Deep Learning is a branch of machine learning that involves pattern recognition on unlabeled or unstructured data. It uses a model of computing inspired by the structure of the brain, which we call this model a neural network.


## Fundamentals

### Inputs vs Features

The variables we use as inputs to our machine learning algorithms are commonly called inputs, but they are also frequently called predictors or features. Inputs/predictors are independent variables and in a simple 2D space, you can think of an input as x-axis variable.

### Outputs vs Targets

The variable that we’re trying to predict is commonly called a target variable, but they are also called output variables or response variables. You can think of the target as the dependent variable; visually, in a 2-dimensional graph, the target variable is the variable on the y-axis.

### Function Estimation

When you’re doing machine learning, specifically supervised learning, you’re using computational techniques to reverse engineer the underlying function your data. 

With that said, we'll go through what this process generally looks like. In the exercise, I intentionally keep the underlying function hidden from you because we never know the underlying function in practice. Once again, machine learning provides us with a set of statistical tools for estimate <i>f(x)</i>.

So we begin with getting our data:
``` R
df.unknown_fxn_data <- read.csv(url("http://www.sharpsightlabs.com/wp-content/uploads/2016/04/unknown_fxn_data.txt"))
```

#### Exploratory Analysis

Now we’ll perform some basic data exploration. First, we’ll use `str()` to examine the “structure” of the data:
``` R
str(df.unknown_fxn_data)
```

Which gets us two variables, `input_var` and `target_var`. 
``` bash
'data.frame':	28 obs. of  2 variables:
 $ input_var : num  -2.75 -2.55 -2.35 -2.15 -1.95 -1.75 -1.55 -1.35 -1.15 -0.95 ...
 $ target_var: num  -0.224 -0.832 -0.543 -0.725 -1.026 ...
```

Next, let’s print out the first few rows of the dataset using the head() function.

``` R
head(df.unknown_fxn_data)
```
You can think of the two columns as (x,y) pairs.

#### Data Visualization

Now, let’s examine the data visually. Since `input_var` is an independent variable, we can plot it on the x-axis and `target_var` on the y-axis. And since these are (x, y) pairs, we'll use a scatterplot to visualize it.

``` R
require(ggplot2)
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
  geom_point()
```

This is important because by plotting the data visually, we can visually detect a pattern which we can later use to compare to our estimated function.

#### Linear Model

Now that we’ve explored our data, we’ll create a very simple linear model using caret. Here, the `train()` function is the “core” function of the caret package, which builds the machine learning model. 

Inside of the `train()` function, we’re using `df.unknown_fxn_data` for the data parameter. `target_var ~ input_var` specifies the “formula” of our model and indicates the target variable that we’re trying to predict as well as the predictor we’ll use as an input. The `method = "lm"` indicates that we want a linear regression model. 

``` R
require(caret)

mod.lm <- train( target_var ~ input_var
                ,data = df.unknown_fxn_data
                ,method = "lm"
                )
```

Now that we’ve built the simple linear model using the `train()` function, let’s plot the data and plot the linear model on top of it.

To do this, we’ll extract the slope and the intercept of our linear model using the `coef()` function.

``` R
coef.icept <- coef(mod.lm$finalModel)[1]
coef.slope <- coef(mod.lm$finalModel)[2]
```

The following plots the data:
``` R
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
  geom_point() +
  geom_abline(intercept = coef.icept, slope = coef.slope, color = "red")
```

As I mentioned in the beginning of this example, I kept the underlying function hidden, which was a sine function with some noise. When we began this exercise, there was a hidden function that generated the data and we used machine learning to estimate that function.
``` R
set.seed(9877)
input_var <- seq(-2.75,2.75, by = .2)
target_var <- sin(input_var) + rnorm(length(input_var), mean = 0, sd = .2)
df.unknown_fxn_data <- data.frame(input_var, target_var)
```

Here, we just visualize that underlying function. 
``` R
ggplot(data = df.unknown_fxn_data, aes(x = input_var, y = target_var)) +
  geom_point() +
  stat_function(fun = sin, color = "navy", size = 1)
```

### Bias and Variance

Understanding how different sources of error lead to bias and variance helps us improve the data fitting process resulting in more accurate models.

#### Bias

Error due to bias is the difference between the expected (or average) prediction of our model and the actual value we're trying to predict. 

#### Variance

Error due to variance is taken as the variability of a model prediction for a given data point. In other words, the variance is how much the predictions for a given point vary between different realizations of the model.

![alt text](https://github.com/lesley2958/intro-ml/blob/master/biasvar.png?raw=true "Logo Title Text 1")


#### Bias-Variance Tradeoff

The bias–variance tradeoff is the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set.

## Optimization

In the simplest case, optimization consists of maximizing or minimizing a function by systematically choosing input values from within an allowed set and computing the value of the function. 

### Loss Function

The job of the loss function is to tell us how inaccurate a machine learning system's prediction is in comparison to the truth. It's denoted with &#8467;(y, y&#770;), where y is the true value and y&#770; is a the machine learning system's prediction. 

The loss function specifics depends on the type of machine learning algorithm. In Regression, it's (y - y&#770;)<sup>2</sup>, known as the squared loss. Note that the loss function is something that you must decide on based on the goals of learning. 

Since the loss function gives us a sense of the error in a machine learning system, the goal is to <i>minimize</i> this function. Since we don't know what the distribution does, we have to calculte the loss function for each data point in the training set, so it becomes: 

![alt text](https://github.com/lesley2958/intro-ml/blob/master/trainingerror.png?raw=true "Logo Title Text 1")

In other words, our training error is simply our average error over the training data. Again, as we stated earlier, we can minimize this to 0 by memorizing the data, but we still want it to generalize well so we have to keep this in mind when minimizing the loss function. 

### Boosting

Boosting is a machine learning meta-algorithm that iteratively builds an ensemble of weak learners to generate a strong overall model.

#### What is a weak learner?

A <i>weak learner</i> is any machine learning algorithm that has an accuracy slightly better than random guessing. For example, in binary classification, approximately 50% of the samples belong to each class, so a weak learner would be any algorithm that slightly improves this score – so 51% or more. 

These weak learners are usually fairly simple because using complex models usually leads to overfitting. The total number of weak learners also needs to be controlled because having too few will cause underfitting and have too many can also cause overfitting.

#### What is an ensemble?

The overall model built by Boosting is a weighted sum of all of the weak learners. The weights and training given to each ensures that the overall model yields a pretty high accuracy.

#### What do we mean by iteratively build?

Many ensemble methods train their components in parallel because the training of each those weak learners is independent of the training of the others, but this isn't the case with Boosting. 

At each step, Boosting tries to evaluate the shortcomings of the overall model built, and then generates a weak learner to battle these shortcomings. This weak learner is then added to the total model. Therefore, the training must necessarily proceed in a serial/iterative manner.

Each of the iterations is basically trying to improve the current model by introducing another learner into the ensemble. Having this kind of ensemble reduces the bias and the variance. 

#### Gradient Boosting


#### AdaBoost


#### Disadvantages

Because boosting involves performing so many iterations and generating a new model at each, a lot of computations, time, and space are utilized.

Boosting is also incredibly sensitive to noisy data. Because boosting tries to improve the output for data points that haven't been predicted well. If the dataset has misclassified or outlier points, then the boosting algorithm will try to fit the weak learners to these noisy samples, leading to overfitting. 

### Occam's Razor

Occam's Razor states that any phenomenon should make as few assumptions as possible. Said again, given a set of possible solutions, the one with the fewest assumptions should be selected. The problem here is that Machine Learning often puts accuracy and simplicity in conflict.
