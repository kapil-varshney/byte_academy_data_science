## Hypothesis Testing

A statistical hypothesis is a hypothesis that is testable on the basis of observing a process that is modeled via a set of random variables. The underlying logic is similar to a proof by contradiction. To prove a mathematical statement, A, you assume temporarily that A is false. If that assumption leads to a contradiction, you conclude that A must actually be true.

Similarly, to test a hypothesis like, “This effect is real,” we assume, temporarily, that is is not. That’s the <b>null hypothesis</b>, which is what you typically want to disprove. Based on that assumption, we compute the probability of the apparent effect. That’s the <b>p-value</b>. If the p-value is low enough, we conclude that the null hypothesis is unlikely to
be true.

### Z-Values, P-Values & Tables

These are associated with standard normal distributions. Z-values are a measure of how many standard deviations away from mean is the observed value. P-values are the probabilities, which you can retrieve from its associated z-value in a [z-table](http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf). 

We've already reviewed how to retrieve the p-value, but how do we get the z-value? With the following formula:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/z%20value.png?raw=true "Logo Title Text 1")

where x is your data point, &mu; is the mean and &sigma; is the standard deviation. 

### Central Limit Theorem

The central limit theorem allows us to understand the behavior of estimates across repeated sampling and conclude if a result from a given sample can be declared to be “statistically significant".

The central limit theorem tells us exactly what the shape of the distribution of means will be when we draw repeated samples from a given population.  Specifically, as the sample sizes get larger, the distribution of means calculated from repeated sampling will approach normality. 

Let's take a look at an example: Here, we have data of 1000 students of 10th standard with their total marks. Let's take a look at the frequency distribution of marks: 

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/clt-hist.png?raw=true "Logo Title Text 1")

This is clearly an unnatural distribution. So what can we do? 

Let's take a sample of 40 students from this data. That makes for 25 total samples we can take (1000/40 = 25). The actual mean is 48.4, but it's very unlikely that every sample of 40 will have this same mean. 

If we take a large number of samples and compute the means and then make a probability histogram on these means, we'll get something similar to:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/clt-samp.png?raw=true "Logo Title Text 1")

You can see that distribution resembles a normally distributed histogram. 

### Significance Level

Significance Tests allow us to see whether there is a significant relationship between variables. It gives us an idea of whether something is likely or unlikely to happen by chance. 

### Steps

The initial step to hypothesis testing is to actually set up the Hypothesis, both the NULL and Alternate.  

Next, you set the criteria for decision. To set the criteria for a decision, we state the level of significance for a test. Based on the level of significance, we make a decision to accept the Null or Alternate hypothesis.

The third step is to compute the random chance of probability. Higher probability has higher likelihood and enough evidence to accept the Null hypothesis.

Lastly, you make a decision. Here, we compare p value with predefined significance level and if it is less than significance level, we reject Null hypothesis, else we accept it.

### Example

Blood glucose levels for obese patients have a mean of 100 with a standard deviation of 15. A researcher thinks that a diet high in raw cornstarch will have a positive effect on blood glucose levels. A sample of 36 patients who have tried the raw cornstarch diet have a mean glucose level of 108. Test the hypothesis that the raw cornstarch had an effect or not.

#### Hypothesis

First, we have to state the hypotheses. We set our NULL Hypothesis to be the glucose variable = 100 since that's the known fact. The alternative is that the glucose variable is greater than 100. 


#### Significance Level

Unless specified, we typically set the significance level to 5%, or `0.05`. Now, if we figure out the corresponding z-value from the [z-table](http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf), we'll see that it corresponds to `1.645`. This is now the z-score cut off for significance level, meaning the area to the right (or z-scores higher than 1.645) is the rejection hypothesis space. 

#### Computation

Now, we can compute the random chance probability using z scores and the z-table. Recall the formula from earlier, z = (x - &mu;)/ &sigma;. Now, before we go into computing, let's overview the difference between standard deviation of the mean and standard deviation of the distribution. 

When we want to gain a sense the precision of the mean, we calculate what is called the <i>sample distribution of the mean</i>. Assuming statistical independence, the standard deviation of the mean is related to the standard deviation of the distribution with the formula &sigma;<sub>mean</mean> = &sigma / &radic;N. 

With that knowledge in mind, we've been given the standard deviation of the distribution, but we need the standard deviation of the mean instead. So before we begin calculating the z value, we plug in the values for the formula above. Then we get &sigma;<sub>mean</sub> = 15 / &radic;36, or `2.5`.

Now we have all the needed information to compute the z value:

```
z = (108-100) / 2.5 = 3.2
```

#### Hypotheses 

Awesome! Now we can get the p-value from the z-value above. We see that it corresponds to `.9993`, but we have to remember to subtract this number from 1, making our p-value `0.0007`. Recall that a p-value below 0.05 is grounds for rejecting the null hypothesis. There, we do just that and conclude that there <i>is</i> an effect from the raw starch.

## Correlation

Now, we'll look at relationships between variables. <b>Correlation</b> is a description of the relationship between two variables.

### Covariance

Covariance is a measure of the tendency of two variables to vary together. If we have two series, X and Y, their deviations from the mean are

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/covariance.png?raw=true "Logo Title Text 1")

where &mu;<sub>X</sub> is the mean of X and &mu;<sub>Y</sub> is the mean of Y. If X and Y vary together, their deviations tend to have the same sign. If we multiply them together, the product is positive when the deviations have the same sign and negative when they have the opposite sign. So adding up the products gives a measure of the tendency to vary together.

Therefore, covariance is the mean of these two products:

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/cov%20final.png?raw=true "Logo Title Text 1")

Note that n is the length of the two series, so they have to be the same length.

``` python
def Cov(xs, ys, mux=None, muy=None):
    """Computes Cov(X, Y).

    Args:
        xs: sequence of values
        ys: sequence of values
        mux: optional float mean of xs
        muy: optional float mean of ys

    Returns:
        Cov(X, Y)
    """
    if mux is None:
        mux = thinkstats.Mean(xs)
    if muy is None:
        muy = thinkstats.Mean(ys)

    total = 0.0
    for x, y in zip(xs, ys):
        total += (x-mux) * (y-muy)

    return(total / len(xs))
```

### Correlation

One solution to this problem is to divide the deviations by &sigma;, which yields standard scores, and compute the product of standard scores.

![alt text](https://github.com/lesley2958/stats-programmers/blob/master/pearson%20coeff.png?raw=true "Logo Title Text 1")

Pearson’s correlation is always between -1 and +1. The magnitude indicates the strength of the correlation. If p = 1 the variables are perfectly correlated. The same is true if p = -1. It means that the variables are negatively correlated.

It's important to note that Pearson's correlation only measures <b>linear</b> relationships. 

Using the mean, varainces, and covariance methods above, we can write a function that calculates the correlation. 

``` python 
import math
def Corr(xs, ys):
    xbar = Mean(xs)
    varx = Var(xs)
    ybar = Mean(ys)
    vary = Var(ys)

    corr = Cov(xs, ys, xbar, ybar) / math.sqrt(varx * vary)
    return(corr)
```

### Confidence Intervals

The formal meaning of a confidence interval is that 95% of the confidence intervals should, in the long run, contain the true population parameter.
