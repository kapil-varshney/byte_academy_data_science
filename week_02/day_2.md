## Cumulative Distribution Functions

### Percentile Rank & Percentiles

The percentile rank is the fraction of people who scored lower than you (or the same). So if you are “in the 90th percentile,” you did as well as or better than 90% of the people who took the exam.

``` python
def percentileRank(scores, your_score):
	count = 0
	for score in scores:
		if score <= your_score:
			count += 1
	percentile_rank = 100.0 * count / len(scores)
	return(percentile_rank)

percentileRank([1,42,53,23,12,3,35,2], 17.5)
```

Alternatively, we can use the `scipy` module to retrieve the percentile rank!

``` python
from scipy import stats
stats.percentileofscore([1,42,53,23,12,3,35,2], 17.5)
```

Both of these output the 50th percentile since 17.5 is the median!

Now, what if we want the reverse? So instead of what percentile a value is, we want to know what value is at a given percentile. In other words, now we want the inputs and outputs to be switched. Luckily, this is available with `numpy`:

``` python
import numpy as np
np.percentile([1,42,53,23,12,3,35,2], 50)
```

This code returns the 50th percentile, e.g median, `17.5`.

### CDFs

The Cumulative Distribution Function (CDF) is the function that maps values to their percentile rank in a distribution.

The following function should look familiar - it's almost the same as percentileRank, except that the result is in a probability in the range 0–1 rather than a percentile rank in the range 0–100.

``` python
def cdf(t, x):
	count = 0.0
	for value in t:
		if value <= x:
			count += 1.0
	prob = count / len(t)
	return(prob)
```

### Interquartile Range

Once you have computed a CDF, it's easy to compute other summary statistics. The median is just the 50th percentile. The 25th and 75th percentiles are often used to check whether a distribution is symmetric, and their difference, which is called the interquartile range, measures the spread.
