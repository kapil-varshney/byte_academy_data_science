# Optimizing Queries and Data Comparisons

## Description

In this exercise you will be utilizing your SQl and Python knowledge to build a database and make efficient queries and comparisons with the data. 

### Part 1 

* Look through the `createdb.py` file and complete the file's methods
* With this file we will be inserting our seed data right after the creation of the database. We will have to invoke our seed data method.
* Create a database `schedules.db` and provide tables with rows and columns to complete the below parameters
	* Students have a name and a major
	* Classes have a title and a field of study
	* Major and field of student are the same (e.g. Economics is a major and a field of study)
	* Students have many classes
	* Classes have many students

### Part 2

* Write a function `shared_classes()` in Python that takes two student names
* It will return the classes these students take together, if any. 
* How would you optimize this speed? Think about the Big O time complexity

### Part 3

* Check out database indexes [here](http://www.programmerinterview.com/index.php/database-sql/what-is-an-index/)
* Check out Python Benchmarking - You can use this to test how fast your queries are running

### Part 4

* Let's optimize further using [Python Sets](https://docs.python.org/3.4/library/stdtypes.html#set)
* A set is similar to a dictionary, only it does not store values. Only keys. 
* Syntax Example:

```
{'Programming', 'Calculus', 'Literature'}
```
* Use sets to store the returned class data about both students and use set's built in methods to find the intersection if it exists.
* The Big O complexity of this operation is `O(len(x) * len(y))`. If you did the most simple comparison with arrays, it was probably `O(len(x) * len(y)**2)`.
* Read about sets. Why is this? Here's a [list](https://wiki.python.org/moin/TimeComplexity) of all Python datatype method's time complexity.

### Part 5

* Instead of doing all your comparisons on the Python side, can we pull the information straight out of the database with a more advanced query?
* Before you write the query to replace your original answers- let's write a new function that takes a Major and number of students, and returns classes where that number of students or more in that class have that major.
* Now, write your enhanced query to find the intersect without any Python data parsing.
* Benchmark this function against your first two.
* Sandbox!! You will want to get familiar with the following SQL commands: GROUP BY, HAVING, IN.
