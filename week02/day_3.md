## 5.0 Probability

Probability is a real value between 0 and 1 that is intended to be a quantitative measure corresponding to the qualitative notion that some things are more likely than others.

The “things” we assign probabilities to are <b>called events</b>. If E represents an event, then P(E) represents the probability that E will occur. A situation where E might or might not happen is called a trial.

### 5.1 Probability Rules

Generally speaking, P(A and B) = P(A) P(B), but this is not always true. 

If two events are mutually exclusive, that means that only one of them can happen, so the conditional probabilities are 0: P(A|B) = P(B|A) = 0. In this case it is easy to compute the probability of either event:
P(A or B) = P(A) + P(B)

### 5.2 Binomial Distribution 

If I roll 100 dice, the chance of getting all sixes is (1/6)<sup>100</sup>. And the chance of getting no sixes is (5/6)<sup>100</sup>. Those cases are easy, but more generally, we might like to know the chance of getting k sixes, for all values of k from 0 to 100. The answer is the binomial distribution, which has this PMF:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/binomial%20pmf.png?raw=true "Logo Title Text 1")

where n is the number of trials, p is the probability of success, and k is the number of successes. The binomial coefficient is pronounced “n choose k”, and it can be computed
recursively like this:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/binomial%20coeff.png?raw=true "Logo Title Text 1")

In Python, this looks like: 

``` python
def Binom(n, k, d={}):
    if k == 0:
        return(1)
    if n == 0:
        return(0)
    try:
        return(d[n, k])
    except KeyError:
        res = Binom(n-1, k) + Binom(n-1, k-1)
        d[n, k] = res
        return(res)
```

### 5.3 Bayes's Theorem

Bayes’s theorem is a relationship between the conditional probabilities of two events. A conditional probability, often written P(A|B) is the probability that Event A will occur given that we know that Event B has occurred. It's represented as follows:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/bayes.png?raw=true "Logo Title Text 1")

Bayes theorem is what allows us to go from a sampling distribution and a prior distribution to a posterior distribution. 


#### 5.3.1 What is a Sampling Distribution?

A sampling distribution is the probability of seeing a given data point, given our parameters (&theta;). This is written as p(X|&theta;). For example, we might have data on 1,000 coin flips, where 1 indicates a head.

In python, this might look like: 

``` python
import numpy as np
data_coin_flips = np.random.randint(2, size=1000)
np.mean(data_coin_flips)
```

As we said in the previous section, a sampling distribution allows us to specify how we think these data were generated. For our coin flips, we can think of our data as being generated from a Bernoulli Distribution. 

Therefore, we can create samples from this distribution like this:

``` python
bernoulli_flips = np.random.binomial(n=1, p=.5, size=1000)
np.mean(bernoulli_flips)
```

Now that we have defined how we believe our data were generated, we can calculate the probability of seeing our data given our parameters. Since we have selected a Bernoulli distribution, we only have one parameter, p. 

We can use the PMF of the Bernoulli distribution to get our desired probability for a single coin flip. Recall that the PMF takes a single observed data point and then given the parameters (p in our case) returns the probablility of seeing that data point given those parameters. 

For a Bernoulli distribution it is simple: if the data point is a 1, the PMF returns p. If the data point is a 0, it returns (1-p). We could write a quick function to do this:

``` python
def bern_pmf(x, p):
	if x == 1:
		return(p)
	elif x == 0:
		return(1 – p)
	else:
		return("Value Not in Support of Distribution")
```

We can now use this function to get the probability of a data point give our parameters. You probably see that with p = .5 this function always returns .5

``` python
print(bern_pmf(1, .5))
print(bern_pmf(0, .5)) 
```
More simply, we can also use the built-in methods from scipy:

``` python
import scipy.stats as st
print(st.bernoulli.pmf(1, .5))
print(st.bernoulli.pmf(0, .5))
```
