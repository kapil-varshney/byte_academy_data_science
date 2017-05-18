
## Background

The purpose of this tutorial is to use your ability to code to help you understand probability and statistics.

### Probability

Probability is the study of random events - the study of how likely it is that some event will happen.

### Statistics

Statistics is the discipline that uses data to support claims about populations. Most statistical analysis is based on probability, which is why these pieces are usually presented together.

## Descriptive Statistics

Descriptive Statistics are the basic operations used to gain insights on a set of data.

### Mean 

An “average” is one of many summary statistics you might choose to describe the typical value or the central tendency of a sample.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/mean.png?raw=true "Logo Title Text 1")

In Python, the mean would look like this: 

``` python
def Mean(t):
    return(float(sum(t)) / len(t))
```

Alternatively, you can use built-in functions from the numpy module: 

``` python
import numpy as np
np.mean([1,4,3,2,6,4,4,3,2,6])
```

### Variance

In the same way that the mean is intended to describe the central tendency, variance is intended to describe the <b>spread</b>. 

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/variance.png?raw=true "Logo Title Text 1")

The x<sub>i</sub> - &mu; is called the "deviation from the mean", making the variance the squared deviation multiplied by 1 over the number of samples. This is why the square root of the variance, &sigma;, is called the <b>standard deviation</b>.

Using the mean function we created above, we'll write up a function that calculates the variance: 

``` python
def Var(t, mu=None):
    if mu is None:
        mu = Mean(t)
    # compute the squared deviations and returns their mean.
    dev2 = [(x - mu)**2 for x in t]
    var = Mean(dev2)
    return(var)
```
Once again, you can use built in functions from numpy instead:

```
numpy.var([1,3,3,6,3,2,7,5,9,1])
```

### Distributions

Summary statistics are concise, but dangerous, because they obscure the data. An alternative is to look at the <b>distribution</b> of the data, which describes how often each value appears.


#### Histograms

The most common representation of a distribution is a histogram, which is a graph that shows the frequency or probability of each value.

Let's say we have the following list: 

``` python
t = [1,2,2,3,1,2,3,2,1,3,3,3,3]
```

To get the frequencies, we can represent this with a dictionary:

``` python
hist = {}
for x in t:
	hist[x] = hist.get(x,0) + 1
```

Now, if we want to convert these frequencies to probabilities, we divide each frequency by n, where n is the size of our original list. This process is called <b>normalization</b>.

``` python
n = float(len(t))
pmf = {}
for x, freq in hist.items():
	pmf[x] = freq / n
```

This normalized histogram is called a PMF, “probability mass function”, which is a function that maps values to probabilities.

#### Mode

The most common value in a distribution is called the <b>mode</b>.

#### Shape

The shape just refers to the shape the histogram data forms. Typically, we look for asymmetry, or a lack there of.

#### Outliers

Outliers are values that are far from the central tendency. Outliers might be caused by errors in collecting or processing the data, or they might be correct but unusual measurements. It is always a good idea to check for outliers, and sometimes it is useful and appropriate to discard them.
