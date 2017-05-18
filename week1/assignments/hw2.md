## Homework Assignment 2

Submit the following by January 21st, 2017 at 3:10pm. 

### Part 1 

In your `README.md` answer the following questions:

1. What are logical errors and how are they different from syntax errors?
2. Is Python a compiled or scripting language? Explain the difference between the two. 
3. What is the outcome of the following conditional statements if the value of variable x is 5?  Show your work.

    a.  ``` x <= 4 or (x != 9  and x > 10) ```
    
    b.  ``` x > 0 and x < 10 and x != 6 ```
    
    c.  ``` x == 1 or x > 0 ```

### Part 2

The datetime module provides data and time objects and a rich set of methods and operators. Read the documentation [here](https://docs.python.org/2/library/datetime.html). Submit the following in a file named `date.py`.

1. Use the datetime module to write a program that gets the current date and prints the day of the week.
2. Write a program that takes a birthday as input and prints the user’s age and the number of days, hours, minutes and seconds until their next birthday.
3. For two people born on different days, there is a day when one is twice as old as the other. That’s their Double Day. Write a program that takes two birthdays and computes their Double Day.
4. For a little more challenge, write the more general version that computes the day when one person is n times older than the other.

### Part 3 

Submit the following in a file named `prime.py`.

1. A prime number is an integer greater than 1 that only has 1 and itself as factors. Write a program in Python than accepts an integer input from a user and then tells the user if the number is prime or not. Your program does not have to accept integers larger than 1 million.

### Part 4 

The [Fibonacci Sequence](http://en.wikipedia.org/wiki/Fibonacci_number) is a list of numbers where each is the sum of the previous two. 

Take a look at the example below.

```
0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ... 
```

Write a function that takes in a number that frints Fibonacci Sequence up to, but not including, that number. Each number should be printed on a new line.

### Part 5

Using the [Luhn Algorithm](http://en.wikipedia.org/wiki/Luhn_algorithm), also known as "modulus 10", we will be determining the validity of a given credit card number.

For now, you are just editing the included python file. You will find the skeleton of the `CreditCard` class inside. We want our class to have its three main properties set on [instantiation](http://en.wikipedia.org/wiki/Instance_(computer_science)) - card_number, card_type, and valid. Look at the code, you'll see this already there.

If the card number given passes the Luhn algorithm, valid should be true and cardType should be set to 'VISA', 'AMEX', etc. If it does not pass, valid should be false and cardType should be set to "INVALID"

This way, we can do this:
```python
    myCard = CreditCard('347650202246884')

    myCard.valid ## true
    myCard.card_type ## 'AMEX'
    myCard.card_number ## '347650202246884'
```

There are three instance methods. You may perform these methods in the order you see fit.

`determine_card_type` should check whether or not the credit card has valid starting numbers and return the card type.

Visa must start with 4  
Mastercard must start with 51, 52, 53, 54 or 55  
AMEX must start with 34 or 37  
Discover must start with 6011  

`check_length` should check whether or not a credit card number is a valid length.

Visa, MC and Discover have 16 digits  
AMEX has 15  

`validate` should run the Luhn Algorithm and return true or false.

The algorithm should works as follows: 

From the right most digit, double the value of every second digit. For example:

`4 4 8 5 0 4 0 9 9 3 2 8 7 6 1 6`

becomes

`8 4 16 5 0 4 0 9 18 3 4 8 14 6 2 6`

Now, sum all the individual digits together. That is to say, split any numbers with two digits into their individual digits and add them. Like so:

`8 + 4 + 1 + 6 + 5 + 0 + 4 + 0 + 9 + 1 + 8 + 3 + 4 + 8 + 1 + 4 + 6 + 2 + 6`

Now take the sum of those numbers and modulo 10.

80 % 10

If the result is 0, the credit card number is valid.

Keep your code super clean and [DRY](http://en.wikipedia.org/wiki/Don't_repeat_yourself). If you are repeating yourself, stop and think about how to better approach the problem. Keep your code encapsulated - imagine your CreditCard class is an interface other code will need to read. You want it as separate and unentangled as possible. Your class should not be dependent on any code outside of it - except, of course, code to instantiate it.

