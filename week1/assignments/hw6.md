## Homework Assignment 6

Submit the following by Monday, January 30th, 2017 at 6:10pm. 

### Part 1

In a file named `README.txt`, answer the following: 

The Python interpreter has strict rules for variable names. Which of the following are legal Python names? If the name is not legal, state the reason.

1. `and`
2. `and`
3. `var`
4. `var1`
5. `1var`
6. `my-name`
7. `your name`
8. `COLOR`


It is important that we know the type of the values stored in a variable so that we can use the correct operators (as we have already seen!). Python automatically infers the type from the value you assign to the variable. Write down the type of the values stored in each of the variables below. Pay special attention to punctuation: values are not always the type they seem!

1. `a = False`
2. `b = 3.7`
3. `c = ’Alex’`
4. `d = 7`
5. `e = ’True’`
6. `f = 17`
7. `g = ’17’`
8. `h = True`
9. `i = ’3.14159’`


### Part 2 

Write a function `def countdown()` that uses a while loop that asks the user for a number, and then prints a countdown from that number to zero. What should your program do if the user inputs a negative number? As a programmer, you should always consider “edge conditions” like these when you program! (Another way to put it- always assume the users of your program will be trying to find a way to break it! If you don’t include a condition that catches negative numbers, what will your program do?). Include this in a file named `hw6.py`.

### Part 3

Write a function `def exp(base, exp)` using a for loop that calculates exponentials. Your program should take the  `base` and an exponent `exp` as variables, and calculate base<sup>exp</sup>. Include this in a file named `hw6.py`.

### Part 4 

Write a function `roots` that computes the roots of a quadratic equation. Check for complex roots and print an error message saying that the roots are complex. Include this in `hw6.py`.

Hint 1: Your function should take three parameters - what are they?
Hint 2: We know the roots are complex when what condition about the discriminant is met?

Be sure to use a variety of test cases, that include complex roots, real roots, and double roots.

### Part 5

Include the following exercises in `lists.py`. Make sure to comment which list goes to which problem.

1. Write a list comprehension that prints a list of the cubes of the numbers 1 through 10.
2. Write a function that takes in a string and uses a list comprehension to return all the vowels in the string
3. Run this list comprehension in your prompt: <br>
  ` [x+y for x in [10,20,30] for y in [1,2,3]]` <br>
Figure out what is going on here, and write a nested for loop that gives you the same result. Make sure what is going on makes sense to you!
