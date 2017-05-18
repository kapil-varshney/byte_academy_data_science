## Functions
We've seen things like `len(my_list)` and `my_list.append(4)` that perform some action on a piece of data. In Python, we call these things functions.

### Defining and Calling Functions
To define a function, we simply use the reserved word `def`.

```python
def multiply(a,b):
	return(a * b)
```

Notice some syntactical measures. After the name of the function, we include a colon. Also, we can specify parameters, which are pieces of data that the function takes that it can work on. In addition, we can return something from this function, in this case the product of the two parameters.

To call a function, one simply passes the parameters.

```python
var product = multiply(4,7) # = 28
```

### Variable Arguments
Imagine that we didn't know how many parameters we want to pass. Python allows us to handle this situation using asterisks. We can pass an arbitrary number of positional arguments (list elements) using a single asterisk, or an arbitrary number of keyword arguments (dictionary elements) using a double asterisk.

```python
# Returns a list from a variable number of parameters
def positional_args(*args):
	return(args)

# Returns a dictionary from a variable number of parameters
def keyword_args(**kwargs):
	return(kwargs)

var my_list = positional_args(1,2,3,4) # = [1,2,3,4]
var my_dict = keyword_args(name="joe", age="17") # = {'name':'joe', 'age':'17}
```

We could also do the opposite of `*args` and `**kwargs` by attaching asterisks to our parameters to unpack them. For example:

```python
positional_args(*my_list) # = positional_args(1,2,3,4)
keyword_args(**my_dict) # = keyword_args(name="joe", age="17")
```

### First Class and Anonymous Functions

Interestingly, Python functions can themselves return functions. This is something JavaScript developers may be familiar with. These functions are called first-class functions.

```python
def get_multiplier_function():
	def multiplier(a,b):
		return(a * b)
	return(multiplier)

multiplier = get_multiplier_function()
product = multiplier(8,7) # = 56
```

Also, what if you had a function that you only wanted to use once, and thus you didn't want to give it a name? These functions are called anonymous functions, and Python provides us with the word `
` to declare these:

```python
g = lambda x: x * x
square = g(8) # = 64
```

### Exercise 4

Write the `square` function, which takes in a number and returns its square, and the `squarify` function which uses the `square` function to square all the numbers in a list.  All `print` statements

```python
# Write the "square" function here


# Write the "squarify" function here


# Don't edit anything below this comment
# --------------------------------------

print(square(4) == 16)
print(square(square(3)) == 81)
print(squarify([3,4,9]) == [9, 16, 81])
```

### Decorator Functions
As we covered previously, in Python, functions are known as first-class objects, which basically means they can be referred to be name and passed as parameters to a function. A [decorator function](http://www.brianholdefehr.com/decorators-and-functional-python), then, is a function that takes another function as a parameter and makes some kind of change to it. These changes can be small or large, but normally they can be very helpful. For example:

```python
def mod_decorator(function):
	def inner_wrapper();
		inner_wrapper.mod_count += 1
		print("I have modified this function " + inner_wrapper.mod_count + " times.")
		function()
	inner_wrapper.mod_count = 0
	return(inner_wrapper)
```

As we can see, we take a function, then wrap that function with the added functionality to print how many times the modified function has been called, then simply call the function we have been passed. We can then call this modified function here.

```python
def my_function():
	print("This is MY function.")

decorated_function = mod_decorator(my_function)

decorated_function() 
# "I have modified this function 1 times."
# "This is MY function."

decorated_function()
# "I have modified this function 2 times."
# "This is MY function."
```

While writing it like this is perfectly legal, Python provides us some additional syntax to make this easier:

```python
@mod_decorator
def my_function():
	"My decorator has been applied automatically!"
```

While this provides a basic introduction to decorator functions, they are a hard concept to understand from a very low level. Please refer to the [following link][stackflow].

### Lambda Functions

Lambda functions are another way of defining short-hand functions. 

``` python
x = lambda x: x ** 2
```
