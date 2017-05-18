## 1.0 Introduction

In this tutorial, we'll go through some of the basics of data cleaning and normalization, all of which you'll apply to your machine learning knowledge later on. 

### 1.1 Glossary

<b> Imputation:</b> The process of replacing missing data with substituted values. <br>

## 2.0 Data Normalization


### 2.1 Dropping Observations

We've established our data isn't always perfect. Sometimes that means dropping values all together. In this example, we'll look at dorm data. We begin by loading the data into `pandas`. 

``` python
import pandas as pd
houses = pd.read_csv("./housing.csv")
```

For this example, we'll be getting rid of the first two rows, which we can easily do with the `drop()` function:

``` python
houses = houses.drop([0,1])
```

This gets us:
```
2      Shapiro    Craig Rhodes
3         Watt  Lesley Cordero
4  East Campus    Martin Perez
5     Broadway   Menna Elsayed
6      Wallach   Will Essilfie
```
Now, let's say one of the students graduated and moved out - obviously we no longer want them in our dataset anymore, so we want to filter it out with condition
``` python
houses = houses[houses.Name != "Lesley Cordero"]
```
```
          Dorm           Name
2      Shapiro   Craig Rhodes
4  East Campus   Martin Perez
5     Broadway  Menna Elsayed
6      Wallach  Will Essilfie
```

## 3.0 Strings


### 3.1 Lower and Upper

The `upper()` and `lower()` string methods return a new string where all the letters in the original string have been converted to uppercase or lower-case, respectively. Nonletter characters in the string remain unchanged. 


``` python
spam = 'Hello World!'
spam = spam.upper()
```

And that returns:
```
'HELLO WORLD!'
```

Likewise:

``` python
spam = spam.lower()
```

And that returns:
```
'hello world!'
```

These methods don't change the string itself but return new strings. If you want to change the original string, you have to call `upper()` or `lower()` on the string and then assign the new string to the variable where the original was stored. This is why you must use `spam = spam.upper()` to change the string in spam instead of simply `spam.upper()`. 

The `upper()` and `lower()` methods are helpful if you need to make a case-insensitive comparison. The strings 'great' and 'GREat' are not equal to each other. But in the many instances, it does not matter whether the user types Great, GREAT, or grEAT because the string is first converted to lowercase.

### 3.2 StartsWith and EndsWith

The `startswith()` and `endswith()` methods return `True` if the string value they are called on begins or ends (respectively) with the string passed to the method; otherwise, they return False. 

``` python
'Hello world!'.startswith('Hello')
```
returns `True`, as expected.

``` python
'Hello world!'.endswith('world')
```
also returns `True`, as expected.

Now, here's an example where we return a `false`:
``` python
'abc123'.startswith('abcdef')
```

These methods are useful alternatives to the == equals operator if you need to check only whether the first or last part of the string, rather than the whole thing, is equal to another string.

### 3.3 Join and Split

The `join()` method is useful when you have a list of strings that need to be joined together into a single string value. The `join()` method is called on a string, gets passed a list of strings, and returns a string. The returned string is the concatenation of each string in the passed-in list. 

``` python
', '.join(['python', 'R', 'Java'])
```

This returns:
```
'python, R, Java'
```

Oppositely, you can split a sentence into its word components. In natural language processing, this is called <b>word tokenization</b>. 

``` python
'My name is Lesley'.split()
```
And you get:
```
['My', 'name', 'is', 'Lesley']
```


## 4.0 Missing Values

Missing data can often be a huge hindrance in data science and taking missing values into account isn't always so simple either. We'll now go over the different methodology of missing values.

If the amount of missing data is very small relative to the size of the dataset, leaving out the few samples with missing features may be the best strategy to prevent biasing the analysis. 

Leaving out available datapoints, however, deprives the data of some amount of information. Depending on the situation, you may want to look for other fixes before deleting potentially useful datapoints from your dataset.

While some quick fixes such as mean-substitution may be fine in some cases, such simple approaches usually introduce bias into the data, for instance, applying mean substitution leaves the mean unchanged but decreases variance.

#### 4.1 Mice

The mice package in R helps you imputing missing values with plausible data values. These plausible values are drawn from a distribution specifically designed for each missing datapoint.

We'll now proceed with an example using the airquality dataset available in R:

``` R
data <- airquality
```

Let's remove some datapoints to work with in this tutorial:
```
data[4:10,3] <- rep(NA,7)
data[1:5,4] <- NA
```

Replacing categorical variables is usually not a good idea. Some data scientists opt to include replacing missing categorical variables with the mode of the observed ones, however, it's not always the best choice. 

Here, we'll remove the categorical variables for simplicity. Then we look at the data using `summary()`.

``` R
data <- data[-c(5,6)]
summary(data)
```

Ozone seems to be the variable with the most missing datapoints. 

#### 4.1.1 Missing Data Classification 

Understanding the reasons why data are missing is important to correctly handle the remaining data. If values are missing completely at random, the data sample is likely still representative of the population. But if the values are missing systematically, analysis may be biased. Here, we'll go into the different types of missing data:

<b>Missing Completely at Random (MCAR)</b> means the data is actually missing at random, which is the best case scenario when it comes to missing data. 

<b>Missing at Random (MAR)</b> means the missing data is not random, but can be accounted for when you take into account another variable.

<b>Missing Not at Random (MNAR)</b> means it's not missing at random, a much more serious issue because the reason why there's missing data is usually unknown. 

MCAR data is obviously the best scenario, but even that can pose a problem too if there's too much missing data. Typically, the maximum threshold for missing data is 5% of the total for large datasets. If it goes beyond that, it's probably a good idea to leave that feature or sample out.

With that said, we'll check to make sure we have sufficient data:

``` R
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(data,2,pMiss)
```

This gets us:

``` bash
    Ozone   Solar.R      Wind      Temp 
24.183007  4.575163  4.575163  0.000000 
```

Yikes. Ozone is missing almost 25% of its datapoints. This means we should drop it. 

### 4.1.2 Missing Data Pattern

The `mice` package provides the `md.pattern()` function to get a better understanding of the pattern of missing data: 

``` R
library(mice)
md.pattern(data)
```

Which gets us:

``` bash
    Temp Solar.R Wind Ozone   
107    1       1    1     1  0
 34    1       1    1     0  1
  4    1       0    1     1  1
  4    1       1    0     1  1
  1    1       0    1     0  2
  1    1       1    0     0  2
  1    1       0    0     1  2
  1    1       0    0     0  3
       0       7    7    37 51
```

This tells us that 104 samples are complete, 34 samples miss only the Ozone measurement, 4 samples miss only the Solar.R value and so on.

Just to make sense of what this means, let's try a visual representation using the `VIM` package:

``` R
library(VIM)
aggr_plot <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
```

This gets us:

![alt text](https://github.com/ByteAcademyCo/data-cleaning/blob/master/missing_data.png?raw=true "Logo Title Text 1")

The plot helps us understand that almost 70% of the samples are not missing any information, 22% are missing the Ozone value, and the remaining shows other missing patterns.


#### 4.1.3 Imputation

The `mice()` function takes care of the imputing process, like this:

``` R
tempData <- mice(data, m=5, maxit=50, meth='pmm', seed=500)
summary(tempData)
```

Here, `m = 5` refered to the number of imputed datasets - 5 is just a default value. `meth = 'pmm'` just refers to the imputation <b>method</b>. In this example, we're using mean matching. If you want to check out what other methods exist, type:

``` R
methods(mice)
```

Which gets you:

``` bash
 [1] mice.impute.2l.norm      mice.impute.2l.pan       mice.impute.2lonly.mean 
 [4] mice.impute.2lonly.norm  mice.impute.2lonly.pmm   mice.impute.cart        
 [7] mice.impute.fastpmm      mice.impute.lda          mice.impute.logreg      
[10] mice.impute.logreg.boot  mice.impute.mean         mice.impute.norm        
[13] mice.impute.norm.boot    mice.impute.norm.nob     mice.impute.norm.predict
[16] mice.impute.passive      mice.impute.pmm          mice.impute.polr        
[19] mice.impute.polyreg      mice.impute.quadratic    mice.impute.rf          
[22] mice.impute.ri           mice.impute.sample       mice.mids               
[25] mice.theme              
see '?methods' for accessing help and source code
```

Now, let's take a look at the imputed data for the Ozone variable:

``` R
tempData$imp$Ozone
```

If you want to check the imputation method used for each variable, you can do so with:

``` R
tempData$meth
```

``` bash 
  Ozone Solar.R    Wind    Temp 
  "pmm"   "pmm"   "pmm"   "pmm" 
```

Now, let's take a look at our finished product:

``` R
completedData <- complete(tempData,1)
```


## 5.0 Outlier Detection

An Outlier is an observation or point that is distant from other observations/points. They can also be referred to as observations whose probability to occur is very low. Outliers are important because they can impact accuracy of predictive models. 

### 5.1 Causes

Often, a outlier is present due to the measurements error. Therefore, one of the most important task in data analysis is to identify and only if it is necessary to remove the outlier.


### 5.2 Parametric vs Non-Parametric

There are two main types of outliers, representative and nonrepresentative. An outlier that is considered representative is one that is correct and not considered unique; and therefore, should not be disregarded from its dataset. 

A nonrepresentative outlier, then, is one that's incorrect because its cause is due to error or because there are no values like it in the rest of the population. These should typically be excluded. 


### 5.3 Example 1

Outlier detection varies between single dataset and multiple datasets. There isn't a concrete definition for what encompasses an outlier, so there are different methodologies to accomplish outlier detection. 
Two methods we'll focus on are Median Absolute Deviation (MAD) and Standard deviation (SD). Though MAD and SD give different results, they're used for the same work.

Let's generate a sample dataset:

``` python
from __future__ import division
import numpy

x = [10, 9, 13, 14, 15,8, 9, 10, 11, 12, 9, 0, 8, 8, 25,9,11,10]
```

#### 5.3.1 Median Absolute Deviation

``` python
axis = None
num = numpy.mean(numpy.abs(x - numpy.mean(x, axis)), axis)
mad = numpy.abs(x - numpy.median(x)) / num
```

#### 5.3.2 Standard Deviation

``` python
sd = numpy.abs(x - numpy.mean(x)) / numpy.std(x)
```
