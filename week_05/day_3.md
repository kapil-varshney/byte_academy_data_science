## 3.0 Non Linear Regression

Non-linear regression analysis uses a curved function, usually a polynomial, to capture the non-linear relationship between the two variables. The regression is often constructed by optimizing the parameters of a higher-order polynomial such that the line best fits a sample of (x, y) observations.

There are cases where non-linear models are <b>intrinsically linear</b>, meaning they can be made linear by simple transformation. But more interestingly, are the ones where it can't.

While a polynomial regression might seem like the best option to produce a low error, it's important to be aware of the possibility of overfitting your data. Always plot the relationships to see the fit and focus on making sure that the curve fits the nature of the problem. 

![alt text](https://github.com/lesley2958/regression/blob/master/und-over.png?raw=true "Logo Title Text 1")


### 3.1 Start Values

Finding good starting values is very important in non-linear regression to allow the model algorithm to converge. If you set starting parameters values completely outside of the range of potential parameter values the algorithm will either fail or it will return non-sensical parameter like for example returning a growth rate of 1000 when the actual value is 1.04.

The best way to find correct starting value is to “eyeball” the data, plotting them and based on the understanding that you have from the equation find approximate starting values for the parameters.

### 3.2 Example 1

In this first example, we'll be using the Michaelis-Menten equation:. 

Here, we simulate some data:

``` R
set.seed(20160227)
x<-seq(0,50,1)
y<-((runif(1,10,20)*x)/(runif(1,0,10)+x))+rnorm(51,0,1)
```

For simple models, `nls` finds good starting values for the parameters:

``` R
m<-nls(y~a*x/(b+x))
```

Now, we get some estimation of goodness of fit:

``` R
cor(y,predict(m))
```

And lastly, we plot:

``` R
plot(x,y)
lines(x,predict(m),lty=2,col="red",lwd=3)
```

### 3.3 Example 2 

Working off of the previous example, we simulate some data to go through an example where we <i>estimate</i> the parameter values:

``` R
y<-runif(1,5,15)*exp(-runif(1,0.01,0.05)*x)+rnorm(51,0,0.5)
```

So now let's visually estimate some starting parameter values:

``` R
plot(x,y)
```

From this graph set, we approximate the starting values. Parameter a is the y value when x is 0 and `b` is the decay rate. 

``` R
a_start<-8 
b_start<-2*log(2)/a_start 
```

Now we're ready for some modeling!

``` R
m<-nls(y~a*exp(-b*x),start=list(a=a_start,b=b_start))
```

Now we get some estimation of goodness of fit and plot it: 
``` R
cor(y,predict(m))
lines(x,predict(m),col="red",lty=2,lwd=3)
```

### 3.4 Example 3

We begin by loading in the needed modules and data: 
``` python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
xdata = np.array([-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9])
ydata = np.array([0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001])
```

Before we start, let's get a look at the scatterplot: 

``` python
plt.plot(xdata,ydata,"*")
plt.xlabel("xdata")
plt.ylabel("ydata")
plt.show()
```

Here, I define the fit function:
``` python
def func(x, p1,p2):
  return(p1*np.cos(p2*x) + p2*np.sin(p1*x))
```

This is where we calculate and show fit parameters: 

``` python
popt, pcov = curve_fit(func, xdata, ydata,p0=(1.0,0.2))
```

Next, we calculate and show sum of squares of residuals since it’s not given by the curve_fit function

``` python
p1 = popt[0]
p2 = popt[1]
residuals = ydata - func(xdata,p1,p2)
fres = sum(residuals**2)
```

And finally, let's plot the curve line along with our data:

``` python
curvex = np.linspace(-2,3,100)
curvey = func(curvex,p1,p2)
plt.plot(xdata,ydata,"*")
plt.plot(curvex,curvey,"r")
plt.xlabel("xdata")
plt.ylabel("xdata")
plt.show()
```

## 5.0 Logistic Regression

Logistic Regression is a statistical technique capable of predicting a <b>binary</b> outcome. It’s output is a continuous range of values between 0 and 1, commonly representing the probability of some event occurring. Logistic regression is fairly intuitive and very effective - we'll review the details now.

### 5.1 Example 1

Here, we'll use the Iris dataset from the Scikit-learn datasets module. We'll use 2 of the classes to keep this binary. 

First, let's begin by importing the needed modules and dataset: 
``` python
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import math
from __future__ import division
data = datasets.load_iris()
```

Now, we select the data for visualization: 
``` python
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]
```

Before we make the logistic regression function, let's take a look to see what we're working with:

``` python
setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
sns.despine()
plt.show()
```

Now you can see that the two classes are completely separate! That means we can [easily] find a function that separates the two classes. 
![alt text](https://github.com/lesley2958/regression/blob/master/log-scatter.png?raw=true "Logo Title Text 1")

We want to return a value between 0 and 1 to represent a probability. To do this we make use of the logistic function. The logistic function mathematically looks like this:

![alt text](https://github.com/lesley2958/regression/blob/master/logistic%20function.png?raw=true "Logo Title Text 1")

Let's take a look at this plot:

``` python
x_values = np.linspace(-5, 5, 100)
y_values = [1 / (1 + math.e**(-x)) for x in x_values]
plt.plot(x_values, y_values)
plt.axhline(.5)
plt.axvline(0)
sns.despine()
``` 

You can see why this is a great function for a probability measure. The y-value represents the probability and only ranges between 0 and 1. Also, for an x value of zero you get a .5 probability and as you get more positive x values you get a higher probability and more negative x values a lower probability.

Recall the function from earlier, Y<sub>i</sub> = m<sub>0</sub> + m<sub>1X<sub>1i</sub> + m<sub>2</sub>X<sub>2i</sub> + &isin;<sub>i</sub>. We can assume that x is a linear combination of the data plus an intercept, so we get the following formula:

x = &beta;<sub>0</sub> + &beta;<sub>1</sub>SW + &beta;<sub>2</sub>SL

where SW is our sepal width and SL is our sepal length. But how do we get our &beta; values? This is where the learning in machine learning comes in. 

### 5.2 Cost Function 

We want to choose β values to maximize the probability of correctly classifying our plants. If we assume our data are independent and identically distributed (iid), we can take the product of all our individually calculated probabilities and that is the value we want to maximize. We get the following formula:

![alt text](https://github.com/lesley2958/regression/blob/master/cost-logistic.png?raw=true "Logo Title Text 1")

This simplifies to: &prod;<sub>setosa</sub> h(x) &prod;<sub>versicolor</sub> 1 - h(x). So now we know what to maximize. We can also switch it to - &prod;<sub>setosa</sub> h(x) &prod;<sub>versicolor</sub> 1 - h(x) and minimize this since minimizing the negative is the same as maximizing the positive. 

We can implement this logistic function like this:

``` python
def logistic_func(theta, x):
    return (float(1) / (1 + math.e**(-x.dot(theta))))
```

And finally, in python, we put all the components together like this: 

``` python
def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return (np.mean(final))
```

### 5.3 Gradients

We now know what value to minimize, but now we need to figure out how to find the &beta; values. This is where convex optimization comes in. 

Since we know the logistic cost function is convex, it has a single global minimum which we can converge using gradient descent. 

The idea behind gradient descent is to pick a point on the curve and follow it down to the minimum. The way we follow the curve is by calculating the gradients or the first derivatives of the cost function with respect to each &beta;.

Now if we define y<sub>i</sub> to be 1 for sentose and 0 for when it's versicolor, then we can simplify to h(x) and 1 - h(x). Recall [log rules](http://www.mathwords.com/l/logarithm_rules.htm). If we take the log of our cost function, our product becomes a sum:

![alt text](https://github.com/lesley2958/regression/blob/master/cost%20funct%202.png?raw=true "Logo Title Text 1")

The next step is to take the derivative with respect to &beta;<sub>0</sub>. Remembering that the derivate of log(x) is 1/x, we get:

![alt text](https://github.com/lesley2958/regression/blob/master/deriv.png?raw=true "Logo Title Text 1")

We have to take the derivative of h(x), which we can do with the quotient rule to see that it's: 

![alt text](https://github.com/lesley2958/regression/blob/master/deriv1.png?raw=true "Logo Title Text 1")

Since the derivative of x with respect to &beta;<sub>0</sub> is just 1, we can put all of this together to get: 

![alt text](https://github.com/lesley2958/regression/blob/master/deriv2.png?raw=true "Logo Title Text 1")

Now we can simplify this to y<sub>i</sub>(1-h(x<sub>i</sub>))-(1-y<sub>i</sub>)h(x<sub>i</sub>) = y<sub>i</sub>-y<sub>i</sub>h(x<sub>i</sub>)-h(x<sub>i</sub>)+y<sub>i</sub>h(x<sub>i</sub>) = y<sub>i</sub> - h(x<sub>i</sub>).

So finally we get: 

![alt text](https://github.com/lesley2958/regression/blob/master/final-gradient.png?raw=true "Logo Title Text 1")

For &beta;<sub>1</sub>, we get:

![alt text](https://github.com/lesley2958/regression/blob/master/beta1.png?raw=true "Logo Title Text 1")

For &beta;<sub>2</sub>, we get: 

![alt text](https://github.com/lesley2958/regression/blob/master/beta2.png?raw=true "Logo Title Text 1")

In Python, we can write:

``` python
def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return (final_calc)
```

### 5.4 Gradient Descent

So now that we have our gradients, we can use the gradient descent algorithm to find the values for our &beta;s that minimize our cost function. The algorithm is as follows:

1. Initially guess any values for &beta;
2. Repeat until we converge: &beta;<sub>i</sub> = &beta;<sub>i</sub>-(&alpha;* gradient with respect to &beta;<sub>i</sub>) for i = 0, 1, 2

Note that &alpha; is our learning rate, which is the rate at which we move towards our cost curve. 

Basically, we pick a random point on our cost curve, check to see which direction we need to go to get closer to the minimum by using the negative of the gradient, and then update our &beta; values to move closer to the minimum.

If we implement this all in python, we would get something like:

``` python
def grad_desc(theta_values, X, y, lr=.001, converge_change=.001):
    # normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
    return(theta_values, np.array(cost_iter))
```

### 5.5 Prediction

The goal to this entire exercise was to show how Logistic Regression can be used for prediction. We went through the process of implementing a cost function, gradient descent -- now we have to put it all together to predict the values!

Let's walk through this code: 

``` python
def pred_values(theta, X, hard=True):
    # normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return (pred_value)
    return (pred_prob)
```

Here I use the above code. I initalize our β values to zeros and then run gradient descent to learn these values.

``` python
shape = X.shape[1]
y_flip = np.logical_not(y) #f lip Setosa to be 1 and Versicolor to zero to be consistent
betas = np.zeros(shape)
fitted_values, cost_iter = grad_desc(betas, X, y_flip)
print(fitted_values)
```

Now we run the `predicted_y()` function to see our probability: 

``` python
predicted_y = pred_values(fitted_values, X)
```

We get 99, which means we got all but 1 value correctly.

But can we do another check by taking a look at how our gradient descent converged:

``` python
plt.plot(cost_iter[:,0], cost_iter[:,1])
plt.ylabel("Cost")
plt.xlabel("Iteration")
sns.despine()
plt.show()
```

You can see that as we ran our algorithm, we continued to decrease our cost function and we stopped right at about when we see the decrease in cost to level out. Nice - everything seems to be working! Lastly, another nice check is to see how well a packaged version of the algorithm does:

``` python
from sklearn import linear_model
logreg = linear_model.LogisticRegression()
logreg.fit(X, y_flip)
sum(y_flip == logreg.predict(X))
```

It also gets 99, a great sign!


### 5.6 Example 2

In the case of logistic regression, the default multiclass strategy is the one versus rest. This example shows how to use both the strategies with the handwritten digit dataset, containing a class for numbers from 0 to 9. The following code loads the data and places it into variables.

``` python
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data[:1700,:], digits.target[:1700]
tX, ty = digits.data[1700:,:], digits.target[1700:]
```

First, let's note that the observations are actually a grid of pixel values. The grid’s dimensions are 8 pixels by 8 pixels. To make the data easier to learn by machine-learning algorithms, the code aligns them into a list of 64 elements.

``` python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
OVR = OneVsRestClassifier(LogisticRegression()).fit(X,y)
OVO = OneVsOneClassifier(LogisticRegression()).fit(X,y)
```

The two multiclass classes OneVsRestClassifier and OneVsOneClassifier operate by incorporating the estimator (in this case, LogisticRegression). After incorporation, they usually work just like any other learning algorithm in Scikit-learn. Interestingly, the one-versus-one strategy obtained the best accuracy thanks to its high number of models in competition.

``` python
"One vs rest accuracy: %.3f" % OVR.score(tX,ty)
"One vs one accuracy: %.3f" % OVO.score(tX,ty)
```
