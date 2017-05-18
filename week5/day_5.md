## 8.0 Ridge and Lasso Regression

Ridge and Lasso regression are powerful techniques used for making efficient models with a large number of features. They work by penalizing the magnitude of coefficients of features along with minimizing the error between predicted and actual observations. These are called <b>regularization</b> techniques.

Ridge Regression specifically performs <b>L2</b> regularization, which means that it adds penalty equality to the <i>square</i> of magnitude of coefficients.

Lasso Regression, on the other hand, performs <b>L1</b> Regression, which means that it adds penalty equivalent to the <i>absolute</i> value of magnitude of coefficients.


### 8.1 Penalization 

Lets try to understand the impact of model complexity on the magnitude of coefficients. As an example, I have simulated a sine curve (between 60° and 300°) and added some random noise using the following code:

``` python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10
```

Here, we just define input array with angles from 60 degrees to 300 degrees converted to radians:

``` python
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
```

And let's see what we get:
``` python
plt.plot(data['x'],data['y'],'.')
plt.show()
```

The visualization resembles a sine curve but not exactly because of the noise we've input. Lets try to estimate the sine function using polynomial regression with powers of x from 1 to 15. Lets add a column for each power upto 15 in our dataframe. This can be accomplished using the following code:


``` python
for i in range(2,16):  # power of 1 is already there
    colname = 'x_%d'%i      # new var will be x_power
    data[colname] = data['x']**i
```

Now that we have all the 15 powers, lets make 15 different linear regression models with each model containing variables with powers of x from 1 to the particular model number. For example, the feature set of model 8 will be – {x, x<sub>2</sub>, x<sub>3</sub>, … ,x<sub>8</sub>}.

First, we’ll define a generic function which takes in the required maximum power of x as an input and returns a list containing – [model RSS, intercept, coef_x, coef_x2, … coef_xy ], where RSS refers to ‘Residual Sum of Squares’ which is nothing but the sum of square of errors between the predicted and actual values in the training data set:

``` python
from sklearn.linear_model import LinearRegression
```

Here, we start out linear regression model by initializing the predictors:
``` python
def linear_regression(data, power, models_to_plot):
    #initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
```

Then we fit the model:

``` python
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
```

Next, we check if a plot is to be made for the entered power:
``` python
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%power)
```


Lastly, we return the result in pre-defined format
``` python
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return (ret)
```

Now, we can make all 15 models and compare the results. We’ll store all the results in a Pandas dataframe and plot 6 models to get an idea of the trend.

First, we initialize a dataframe to store the results:

``` python
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)
```

Next, we define the powers for which a plot is required:
``` python
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}
```

Lastly, we iterate through all powers and assimilate results
``` python
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)

plt.show()
```

As you can, models with increasing complexity to better fit the data and result in lower RSS values. This makes sense because as the model complexity increases, the models tends to overfit. 


As far as coefficients go, the size of coefficients increases exponentially with increase in model complexity. We can see this here:


``` python
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_simple
```

What does a large coefficient mean? It means that we’re putting a lot of emphasis on that feature, i.e. the particular feature is a good predictor for the outcome. When it becomes too large, the algorithm starts modelling intricate relations to estimate the output and ends up overfitting to the particular training data.


### 8.2 Ridge Regression

Recall, ridge regression performs <b>L2 </b>regularization, i.e. it adds a factor of sum of squares of coefficients in the optimization objective, so it ends up optimizing:

```
Objective = RSS + α * (sum of square of coefficients)
```

Here, α is the parameter which balances the amount of emphasis given to minimizing RSS vs minimizing sum of square of coefficients. α can take various values:

α = 0:
- The objective becomes the same as simple linear regression.
- We’ll get the same coefficients as simple linear regression.

α = ∞:
- The coefficients will be zero. Why? Because of infinite weightage on square of coefficients, anything less than zero will make the objective infinite.

0 < α < ∞:
- The magnitude of α will decide the weightage given to different parts of objective.
- The coefficients will be somewhere between 0 and ones for simple linear regression

This tells us that any non-zero value would give values less than that of simple linear regression. Let's define a generic function for ridge regression similar to the one defined for simple linear regression:

``` python
from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
```

Note that the function above takes ‘alpha’ as a parameter on initialization. Now, we fit the model:

``` python
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
```

We check to se if a plot is to be made for the entered alpha:
``` python
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
```
Finally, we return the result in pre-defined format:
``` python
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return(ret)
```

Now, lets analyze the result of Ridge regression for 10 different values of α ranging from 1e-15 to 20. These values have been chosen so that we can easily analyze the trend with change in values of α. 

First, we initialize predictors to be set of 15 powers of x

``` python
predictors = ['x']
predictors.extend(['x_%d'%i for i in range(2,16)])
```

Then, we set the different values of alpha to be tested
``` python
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
```

We want to store the coefficients, so we initialize a dataframe for them:

``` python
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
```

Now we actually plot them: 

``` python
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
plt.show()
```

Here, we can see that as the value of alpha increases, the model complexity reduces. Though higher values of alpha reduce overfitting, significantly high values can cause underfitting as well, so alpha should be chosen wisely. A widely accept technique is cross-validation, i.e. the value of alpha is iterated over a range of values and the one giving higher cross-validation score is chosen.

Lets have a look at the value of coefficients in the above models:

``` python
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge
```

Ridge Regression is a technique used when the data suffers from multicollinearity (independent variables are highly correlated). In multicollinearity, even though the least squares estimates are unbiased, their variances are large which deviates the observed value far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.

Ridge regression solves the multicollinearity problem through shrinkage parameter &lambda;, shown below:

![alt text](https://github.com/lesley2958/regression/blob/master/ridge.png?raw=true "Logo Title Text 1")

In this equation, we have two components. First, is the least square term and other is lambda of the summation of β2 (beta- square) where β is the coefficient. This is added to least square term in order to shrink the parameter to have a very low variance.


### 8.3 Assumptions

The assumptions of this regression is same as least squared regression, except normality is not to be assumed.
