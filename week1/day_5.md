
## R & R Studio

Download [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

## Packages

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("")
```

## What is R?

R is a powerful language used primarily for data analysis and statistical computing. R has what we call `packages`, which can used for power tasks. Packages such as dplyr, tidyr, readr, data.table, SparkR, ggplot2 have made data manipulation, visualization and computation much easier and faster.

## Why use R?

- It’s open source
- Availability of instant access to over 7800 packages customized for various computation tasks.
- High performance computing experience 
- One of highly sought skill by analytics and data science companies.

## Comments 

Comments in the context of computer science are used for providing details throughout your code. They're particularly useful when you're working on something complex and want to remember why or what you did, as well as for when other people need to read your code and don't have you to explain it to them!

In R, we denote comments with the `#` symbols, such as follows:

``` R
# This is a comment!
```

## Data in R

### Variables

Variables are names we assign values to. Why do we want to do this? Because without variables, we don't have a way of referencing and using data. Value can be many things, including another variable, but in most casses, the value is a <b>data type</b>. 

In R, there are actually <b>two</b> ways of assigning values: `=` and `<-`. Typically though, we use `<-`, such as `my_val <- 4`.


### Data Types and Operators

Every programming language needs to store data and a way to work with this data. R, like other languages, breaks these data into types and provides different ways to interact with them. 

Everything you see or create in R is an <b>object</b>. A vector, matrix, data frame, even a variable is an object. R treats it that way. So, R has 5 basic classes of objects, including:

- Character
- Numeric (Real Numbers)
- Integer (Whole Numbers)
- Complex
- Logical (True/False)

These classes have attributes, such as the following:

- names 
- dimension names
- dimensions
- class
- length

Attributes of an object can be accessed using `attributes()` function. We will get into what functions are later.

#### Challenge

Assign three variables called `var1`, `var2`, and `var3` to the values `1`, `"Byte"`, and `5.43`. 


## Vectors

The most basic object in R is known as vector, which contains objects of the same class. You can create an empty vector using `vector()`. 

Let’s try creating vectors of different classes. We can create vector using `c()`:

``` R
a <- c(1.8, 4.5)   #numeric
b <- c(1 + 2i, 3 - 6i) #complex
d <- c(23, 44)   #integer
e <- vector("logical", length = 5)
```

### Lists

Lists are present in R, as well as most other programming languages. A list is a data structure that can hold any number of any types of other data structures. For example, if you have vector, a dataframe, and a character object, you can put all of those into one list object. 

#### Initiliazation

To initialize a list to a certain number of components, we use the vector function like this:

``` R
list2 <- vector("list", 3)
```

#### Constructing a List

``` R
vec <- 1:4
df <- data.frame(y = c(1:3), x = c("m", "m", "f"))
char <- "Hello!"
```

Then you can add all three objects to one list using list() function:

``` R
list1 <- list(vec, df, char)
```

You can also turn an object into a list by using the `as.list()` function. Notice how every element of the vector becomes a different component of the list:


#### Manipulating a List
We can put names on the components of a list using the `names()` function, which is useful for extracting components. We could have also named the components when we created the list.

``` R
names(list1) <- c("Numbers", "Some.data", "Letters")
```

#### Extracting Components

The first way in which you can extract an object from the list is by using the [[ ]] operator. 

``` R
list1[[3]]
```

It's also possible to extract components using the component’s name, as shown below:

``` R
list1$Letters
```

#### Subsetting a List

If you want to take a subset of a list, you can use the [ ] operator and c() to choose the components: 

``` R
list1[c(1, 3)]
```

We can also add a new component to the list or replace a component using the $ or [[ ]] operators, such as the following two examples: 

``` R
list1$newthing <- lm(y ~ x, data = df)
```

``` R
list1[[5]] <- "new component"
```

Finally, we can delete a component of a list by setting it equal to NULL:

``` R
list1$Letters <- NULL
```

### Describing Lists

#### Class

The class of the list and the class of one of the components of the list.

``` R
class(list1)
```
and 
```
class(list1[[1]])
```

#### Size 

You can find the size of a list with the `length()` method, like in the following:

``` R
length(list1)
```

#### Converting

Finally, we can convert a list into a matrix, dataframe, or vector in a number of different ways. The first, most basic way is to use unlist(), which just turns the whole list into one long vector:

``` R
unlist(list1)
```

### Matrices

When a vector is introduced with row and columns (the dimension attribute), it becomes a matrix. It consist of elements of same class, such as the following:

``` R
my_matrix <- matrix(1:6, nrow=3, ncol=2)
```

### DataFrame

DataFrames are used to store tabular data. It's similar to a matrix in that there are rows and columns, but it's different because every element does <b>not</b> have to be the same class. In a dataFrame, you can put list of vectors containing different classes. This means that every column of a data frame acts like a list. 

``` R
df <- data.frame(name = c("ash","jane","paul","mark"), score = c(67,56,87,91))
```

Which looks like this:

```
  name score
1  ash    67
2 jane    56
3 paul    87
4 mark    91
```

DataFrame objects are incredibly useful when working with data that has relational relationships, such as a csv file. You'll soon see the extent to which these become useful soon enough!

#### Challenge

Using the variable `df1`, create a 3x3 dataframe using three lists.


## Control Flow

What good is a program if we can't make decisions? Luckily, we have several tools at our disposal that allow us to make these decisions, which direct the way our program executes in such a way to make it meaningful.


### If/Else

R, among many other programming languages, provides us with an if/else statement to test a <b>condition</b> in our code. Conditions allow us to have control flow in our R programs, which means we have control over whether a particular piece of code is run or not. Our code decides this by testing the condition. 

A condition is an expression that functions similar to a question and evaluates to either True or False. Below is the syntax:

``` R
if (condition) {
         ## statement 1
} 
else {
         ## statement 2
}
```


Multiple conditions can be combined with repeating if - else. Below is the syntax for 4 conditions: 

```R
if (condition 1) {
         ## statement 1
} else if (condition 2) {
         ## statement 2
} else if (condition 3) {
         ## statement 3
} else
         ## statement 4
```

#### Challenge 

For all the below if else statements, test with the values -1, 0, 1.

1. Write an if else statement that if an input is greater than zero, it prints "this value is positive", else it prints "this value is negative or zero".

2. Write an if else statement that if an input is greater than zero, it returns double the value,  else it returns triple the value

3. Write an if else statement that if `x > 0` it prints "positive", if x<0 prints "negative and  if x = 0 it prints "zero"
4. Write an if else statement that  if x>0  it returns the value doubled, if x<0, returns the value tripled and if x = 0 it returns 0


### For Loops

Remember those lists and vectors we made earlier? The ones that hold multiple values? If you don't, here is an example.

``` R
list1 <- list("dog", "cat", "bird", "turtle", "fish", "hamster", "lizard")
```

What if we  wanted to print every value in that list? We could do something like this:

``` R
print(my_pets[1])
print(my_pets[2])
print(my_pets[3])
print(my_pets[4])
print(my_pets[5])
print(my_pets[6])
```

But there is an easier way to do this: For loops. Loops are fundamental to all programming languages. Their purpose is to iterate through a data structure and interact with each element one by one.

Loops are very powerful programming tools, and you'll use them fairly frequently. They're useful because computers are very good at repeating identical or similar tasks without making errors.

Let's use a for loop to print each pet from the above example:

``` R
for (i in list1) { 
   print(i)
}
```

And we get: 

```
[1] "dog"
[1] "cat"
[1] "bird"
[1] "turtle"
[1] "fish"
[1] "hamster"
[1] "lizard"
```

Awesome! Let's look at another example below. 

## Example

Let's write a for loop to print all numbers between 0 and 10. 

``` R
for (i in 1:10) {
    print(i)
}
```

#### Challenge

FizzBuzz is a common interview brain teaser. 

- The `fizz_buzz` function will take a number as an argument.
- The function should print all integers starting at one, and going up to, and including, the input number.
- When you print the numbers, if the number you're printing is divisible by 3, print "Fizz" instead.
- When you print the numbers, if the number you're printing is divisible by 5, print "Buzz" instead.
- If the number is divisible by both 3 and 5, print "FizzBuzz".
- If the number is not divisible by 3 or 5, simply print the integer.

Your program's output should look like this:

```
1
2
Fizz
4
Buzz
...
```

### While Loops

Another common loop is the while loop.

This is similar to a for loop, but instead of iterating through a data structure, this loop will continue to run until a condition is no longer true.

Let's look at an example:

``` R
i <- 0
while (i < 10) {
   print(i)
   i <- i + 1
}
```

Which prints the following:

```
[1] 0
[1] 1
[1] 2
[1] 3
[1] 4
[1] 5
[1] 6
[1] 7
[1] 8
[1] 9
```
Here's the flow of execution in a while loop: * Evaluate the condition, i < 10, yielding False or True. * If the condition is false, exit the while statement and continue execution at the next statement. * If the condition is true, execute each of the statements in the body and then go back to step 1.

In the example above, notice that at the end of every turn the loop will increment "i" by 1, so eventually "i" will be greater than 10.

Be careful when using while loops. If your condition always remains true, the loop will never end. This is known as an inifite loop

Below is a similar example except the condition was changed and it has now become an infinite loop:

``` R
i <- 10
while (i > 5) {
   print(i)
   i <- i + 1
}
```

#### Challenge

Using a **while** loop, print all the even numbers between **1 and 10** (including 10)

### List Iteration

For loops are incredibly useful for iteraton, but unfortunately have the issue of memory consumption and slowness in executing a repetitive task. When dealing with large data, the for loop is usually something we stay away from. R provides a few alternatives that can be applied on vectors for iteration. 

In this section, we deal with apply function and its variants! 

To understand the apply functions in R, we'll be using the data from 1974 Motor Trend US magazine which comprises fuel consumption and 10 aspects of automobile design and performance for 32 automobiles (1973–74 models).

``` R
data("mtcars")
head(mtcars)
```

Just so we can get a glimpse of what's going on with our data: 

```
                   mpg cyl disp  hp drat    wt  qsec vs am gear carb
Mazda RX4         21.0   6  160 110 3.90 2.620 16.46  0  1    4    4
Mazda RX4 Wag     21.0   6  160 110 3.90 2.875 17.02  0  1    4    4
Datsun 710        22.8   4  108  93 3.85 2.320 18.61  1  1    4    1
Hornet 4 Drive    21.4   6  258 110 3.08 3.215 19.44  1  0    3    1
Hornet Sportabout 18.7   8  360 175 3.15 3.440 17.02  0  0    3    2
Valiant           18.1   6  225 105 2.76 3.460 20.22  1  0    3    1
```

We'll also be using the following dataset: Reynolds (1994) describes a small part of a study of the long-term temperature dynamics of beaver Castor canadensis in north-central Wisconsin. Body temperature was measured by telemetry every 10 minutes for four females, but data from a one period of less than a day for each of two animals is used there. 

``` R
data(beavers)
head(t(beaver1)[1:4,1:10])
```

Again, just to take a look:

```
        [,1]   [,2]   [,3]   [,4]   [,5]   [,6]   [,7]   [,8]    [,9]   [,10]
day   346.00 346.00 346.00 346.00 346.00 346.00 346.00 346.00  346.00  346.00
time  840.00 850.00 900.00 910.00 920.00 930.00 940.00 950.00 1000.00 1010.00
temp   36.33  36.34  36.35  36.42  36.55  36.69  36.71  36.75   36.81   36.88
activ   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00    0.00    0.00
```

#### Apply

The `apply()` function is our base function that will be used to apply family functions. The `apply()` function takes 3 arguments:

- data matrix
- row/column operation: 1 for a row operation, 2 for a column operation
- function to be applied on the data.

In the example below, the row  maximum value is calculated. Since we have four types of attributes, there are 4 results.

``` R
apply(t(beaver1), 1, max) 
```
```
    day    time    temp   activ 
 347.00 2350.00   37.53    1.00 
```

In the below example, we apply the mean function on each column! 

``` R
apply(mtcars, 2, mean)
```

Notice that this apply function operated on the columns instead:

``` 
       mpg        cyl       disp         hp       drat         wt       qsec 
 20.090625   6.187500 230.721875 146.687500   3.596563   3.217250  17.848750 
        vs         am       gear       carb 
  0.437500   0.406250   3.687500   2.812500 
```

So far we've only applied simple built in methods like `mean()` or `max()`, but we can also pass functions which we have created! For example, in the below example we feed in a function that divides each column element with modulus of 10:

``` R
head(apply(mtcars,2,function(x) x%%10))
```

Awesome stuff: 

``` 
                  mpg cyl disp hp drat    wt qsec vs am gear carb
Mazda RX4         1.0   6    0  0 3.90 2.620 6.46  0  1    4    4
Mazda RX4 Wag     1.0   6    0  0 3.90 2.875 7.02  0  1    4    4
Datsun 710        2.8   4    8  3 3.85 2.320 8.61  1  1    4    1
Hornet 4 Drive    1.4   6    8  0 3.08 3.215 9.44  1  0    3    1
Hornet Sportabout 8.7   8    0  5 3.15 3.440 7.02  0  0    3    2
Valiant           8.1   6    5  5 2.76 3.460 0.22  1  0    3    1
```

#### Lapply

The `lapply()` function is applied for operations on list objects and returns a list object of same length of original set. For this example, we'll create a simple list of 2 elements and then use the `lapply()` method:

``` R
l = list(a=1:10, b=11:20)  
lapply(l, mean)
```

Notice the `lapply()` function above only has <i>two</i> parameters. This is because we don't have to worry about whether it's a row-wise or column-wise operation. We get the result: 

```
$a
[1] 5.5

$b
[1] 15.5
```

Just to confirm that `lapply()` returns a list, let's check out the class: 

``` R
class(lapply(l, mean))
```

Which gets us a list, just as expected!

```
[1] "list"
```

#### Sapply

Now, the `sapply()` method is similar to the `lapply()` -- in fact, it's what we call a "wrapper class", but the different here is that it returns a vector or matrix instead of a list object.

Using the example from the previous section, let's look at the different between each method! 

``` R
l = list(a=1:10,b=11:20)
sapply(l, mean)
```

This yields the output, which is slightly different than `lapply()`s output:

``` 
   a    b 
 5.5 15.5 
 ```
 
 The reason for this is because the output of `sapply`, as we said above, is <b>not</b> a list and instead a vector! We can confirm this by checking the class:
 
 ```
 [1] "numeric"
```

And what do you know, we've got a numeric vector!

#### Tapply()

Lastly, we have `tapply()`, which allows your code to break a vector into subsets and apply a function to each of these sets. In the below code, we group the data by cylinder type and have the `mean()` function calculated for each type.

``` R
levels(as.factor(mtcars$cyl))
```

With this output, we can see that there are three categories of cylinders, which is how we'll group the vector! 
tapply(mtcars$mpg,mtcars$cyl,mean)

```
[1] "4" "6" "8"
```

And finally, we call the `tapply()` function! The first parameter is the data we'll perform the actual function on, the second parameter is how we will group the subsets, and the mean is the function we'll apply onto this data!

``` R
tapply(mtcars$mpg, mtcars$cyl, mean)
```

Which outputs the following below! As you can see, we now have <b>three</b> different calculates, each of which represents the mean of the three different types of cylinders:

```
       4        6        8 
26.66364 19.74286 15.10000 
```

#### Challenge

Given this matrix, A
```R
     [,1] [,2] [,3]
[1,]    1    3    5
[2,]    2    4    6
```
from the following code:
```R
A = matrix( 
   c(1,2,3,4,5,6), 
   nrow=2,
   ncol=3)
```

Given this following list, B
```R
[[1]]
[1] -2 -1  0  1  2

[[2]]
[1] 4 1 0 1 4
```
from the following code:
```R
B <-list(-2:2, (-2:2)^2)
```

1. Find the mean of the rows for matrix A. - call this `mean1`
2. Find the max of the columns for matrix A. - call this `max1`
3. Find the min of each list from listB. - call this `min1`
4. Using the first element in list B, B[1], generate a list with the values squared using lapply(). (hint: this will look like the second element of list B) - call this `list1` 


## Input/Output

Input and Output refers to how our code interacts with a user or computer There are many different methods to input/output data for R. In this section, we'll review the different ways in which R handles input and output.

### Source Command

``` R
source("/path-to-file/name_of_file.R")
```

Source will load and execute a script of R commands.  For instance - if you have saved functions in another file -  you can use source to access the file instead of rewriting the function. Make sure to have both files in the current working directory or to include the path. 

### Print 

When interacting with a user, we might want to send messages to them. To do that, we send messages to the console via the `print` command.

``` R
print ("Hello World")
```

This function, as shown above, allows us to print "Hello World". 

### Read a CSV

``` R
read.csv("/path-to-file/name_of_file.csv")
```

There are many formats and standards of text documents for storing data.  One common format for storing data are delimiter-separated values (CSV or tab-delimited).

### Read a Table

``` R
read.table("/path-to-file/name_of_file.csv")
```

This function can read delimited files and store the results in a data frame.

### Challenge

1. Use the source command to load the `using_source_example.R` script and define a new variable `var` that runs this function for the value 7. Also print `var`. 

2. Load the file `values_squared.csv` as a variable `csv_var`. Print `csv_var` to see the results. Make sure to not include the first line of the csv file in the header. (hint: look at the header argument)

3. Load the file `values_squared.csv` as a variable `table_var`. Print `table_var` to see the results. Make sure to not include the first line of the csv file in the header. (hint: look at the header argument)


## Functions

A function is a block of code that we invoke by using the function name with parenthesis `()`.

We initialize a function with the reserved function `function()`. In the function's parenthesis we state the parameters it takes when called.

We call this "function declaration". It looks like this: `y <- function(x)`. Recall that x is the input and y is the output.  

More broadly put, we have something like: 

```R
function_name <- function( arguments ) {
  body - returns some computation of the arguments 
}
```
The arguments are the input and the body is the output.

### Challenge 

1.  Write a function called `fun1` that takes x as in input and returns 2x.  what value do we get when we run 5 through the function?  Test your function.  (ie.  when you run fun1(3), do you get the number 6?) 

2. Write a function called `fun2` that takes two inputs, a and b,  and returns (a + b) ^2. 

3.  Write a function called `fun3` that takes two inputs, a and b,  it returns a list.  the first element of the list return the opperation a+b (call this `add`)  and second element will return the opperation a-b ( call it `sub`).  
