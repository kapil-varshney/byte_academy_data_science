## Homework Assignment 3

Submit the following by Thursday, January 26th, 2017 at 6:10pm. 

### Part 1 

In your `README.txt` answer the questions below. Recall that `scope` refers to  which parts of your code can use a given object. For example, global variables can be used by the rest of the code since it is global.

Here is an example of a simple class which stores information about a person:
``` python
import datetime # we will use this for date objects

class Person:

    def __init__(self, name, surname, birthdate, address, telephone, email):
        self.name = name
        self.surname = surname
        self.birthdate = birthdate

        self.address = address
        self.telephone = telephone
        self.email = email

    def age(self):
        today = datetime.date.today()
        age = today.year - self.birthdate.year

        if today < datetime.date(today.year, self.birthdate.month, self.birthdate.day):
            age -= 1

        return(age)
        
person = Person(
    "Jane",
    "Doe",
    datetime.date(1992, 3, 12), # year, month, day
    "No. 12 Short Street, Greenville",
    "555 456 0987",
    "jane.doe@example.com"
)

print(person.name)
print(person.email)
print(person.age())
```
Explain what the following variables refer to, and their scope:

1. `Person`
2. `person`
3. `surname`
4. `self`
5. `age` (the function name)
6. `age` (the variable used inside the function)
7. `self.email`
8. `person.email`

### Part 2 

Continuing in your `README.txt`, rewrite the following slices of code as list comprehensions:

``` python
ans = []
for i in range(3):
    for j in range(4):
        ans.append((i, j))
print(ans)
```

``` python
ans = map(lambda x: x*x, filter(lambda x: x%2 == 0, range(5)))
print(ans)
```

### Part 3

Suppose you're a hospital manager and are considering the use of a new method to diagnose a rare form of heart disease. You know that only 0.1% of the population suffers from that disease and that if a person has the disease, the test has a 99% of chance of turning out positive. If the person doesn’t have the disease, the test has a 98% chance of turning negative.

In `README.txt`, explain how feasible is this diagnostics method. That is, given that a test turned out positive, what are the chances of the person really having the disease? Please give a thorough explanation and steps as to how you completed this problem. 

### Part 4

We're trying to find the best quarterback of the 2015 NFL season using passer rating and quarterback rating, two different measures of how the quarterback performs during a game. The numbers in the sets below are the different ratings for each of the 16 games of the season (A being passer rating, B being quarterback rating, the first element being the first game, the second element being the second, etc.) The better game the quarterback has, the higher each of the two measures will be; I’m expecting that they’re correlated and dependent on each other to some degree. You can assume that they’re normally distributed.

``` python
A = {122.8, 115.5, 102.5, 84.7, 154.2, 83.7, 122.1, 117.6, 98.1, 111.2, 80.3, 110.0, 117.6, 100.3, 107.8, 60.2}
B = {82.6, 99.1, 74.6, 51.9, 62.3, 67.2, 82.4, 97.2, 68.9, 77.9, 81.5, 87.4, 92.4, 80.8, 74.7, 42.1}
```

Your job is to find out the probability that (91.9 <= A <= 158.3) and (56.4 <= B <= 100), the mean, and standard deviation. Lastly, is there any way that I can find out P(A|B) and P(B|A) given the data we have? 

Submit your code in a file named `nfl.py`. Make sure to thoroughly comment your code.

### Part 5

The following code should be written in a file called `part5.py`.

Consider a sequence of n Bernoulli trials with success probabilty p per trial. A string of consecutive successes is known as a success `run`. Write a function called `run_counts` that returns the counts for runs of length k for each k observed in a dictionary.

For example: if the trials were [0, 1, 0, 1, 1, 0, 0, 0, 0, 1], the function should return

``` python
{1: 2, 2: 1})
```

Continuing on, what is the probability of observing at least one run of length 5 or more when n=100 and p=0.5?. Estimate this from 100,000 simulated experiments. Is this more, less or equally likely than finding runs of length 7 or more when p=0.7? Implement this as a function called `run_prob`. 


