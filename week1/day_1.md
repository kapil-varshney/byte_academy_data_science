
## Preface: Comments
Comments are like notes a programmer takes while writing code. Comments do not affect the code, and should be used liberally to ensure code can be easily understood later both by yourself and other developers.

```python
# If you only need one line for your comment, use a hash.

""" 
If you want a comment that spans multiple lines, 
you can use triple quotes at the beginning and end.
"""
```

We will use these throughout this tutorial to make it easier to follow our code. You should do the same when writing your own code!

## Data Types and Operators
Every programming language needs to store data and a way to work with this data. Python, like other languages, breaks these data into types and provides different ways to interact with them.

<a id="math"></a>
### Math and Numbers
Numbers in Python are, well, numbers. They act like normal numbers. They can be added, subtracted, multiplied, and divided.

```python
3 + 5 # = 8
9 - 2 #  = 7
4 * 5 # = 20
```

In Python 2.7, uneven numbers would round down a whole number, so `11/4` would output `4`. In Python 3, however we get the actual value:

```python
11 / 4 # = 2.75
```

### Booleans
Booleans are used to store whether something is true or false. We represent these using two values:

```python
# Notice the capital letters!
True
False
```

We can negate these values using the word `not`:

```python
not True # False
not False # True
```

In addition, we have one more datatype, called `None`. It is similar to `null` in Java, and can be used to check whether an object contains anything. It is seen as being equal to `False`.

### Comparisons
Channel your pre-algebra skills; remember all the mathematical comparisons available to us. These comparisons evaluate to boolean values, and can be used the same ways booleans can be.

```python
# Greater/Less Than
2 < 3 # True
4 > 7 # False
3 >= 3 # True

# Equality uses ==
5 == 4 # False

#  Inequality uses !=
6 != 4 # True

# Don't forget about chaining!
1 < 5 < 10 # True
```

One more comparison can check the type of an object. This is with the keyword `is`. Use `is` when checking if an object is `None`.

```python
"Python" is None # False
```

### Strings
We use strings to store text. Use either `'` or `"`, whichever you prefer.

```python
'This is my string. I like it very much.'
"This is my other string. It is even nicer than the first."

# Concatenate (add) strings
'Hello ' + ' ADI!' # 'Hello ADI!'

# Pull out individual characters by their index
'Hello'[0] # 'H'

# Format them too!
"I like to eat %s and %s" % ("cookies", "cake") # "I like to eat peas and carrots"
```
### Printing

We'll probably want to be able to print things to the screen. We can easily do this with:

```python
print("Here is how we print.")
```

<a href="#top" class="top" id="variablescolls">Top</a>

## Variables and Collections

What happens if we want to use a value more than once? Or what happens if we want to store the result of, say, a comparison or an operation? Or, even better, what if we have the results of several operations that we want to store together? For these things, we use variables and collections.

### Variables

Unlike in Java and other languages, we do not specify a type! Also, Python uses a different naming convention than Java, going for underscores instead of capital letters.

```python
my_var = 5 + 4
print(my_var) # 9
```
### Lists

Lists store collections of data indexed by a number, starting at 0. We can start with an empty list, or add some elements to start with:

```python
my_list = []
my_other_list = [1,5,9,6,7,8]
```

We can add and remove from the end of the list:

```python
# Add an element to the end
my_other_list.append(6) # [1,5,9,6,7,8,6]

# Remove an element from the end
my_other_list.pop() # 6
```

Access and remove elements at certain positions, or even ranges:

```python
# Access element at position 0
my_num = my_other_list[0] # 1

# Delete element at position 1
del my_other_list[1] # [1,9,6,7,8]

# Get elements between elements 1 and 3
range_1 = my_other_list[1:3] # [9,6]

# Get elements from beginning to element 3
range_2 = my_other_list[:3] # [1,9,6]

# Don't go too high though!
my_other_list[9] # IndexError - there is no element at position 9
```

Add lists together:

```python
["first", "second"] + ["third"] # ["first", "second", "third"]
```

Check if an element is within a list:

```python
4 in [1,2,5] # False
```

And get how many items are in a list:

```python
len([8,6,4,3]) # 4
```

### Tuples

Tuples are just like lists, except they use parentheses instead of brackets. Also, they are immutable, which means you cannot change the elements in them.

```python
my_tuple = (1,2,3) # Use parentheses instead!
my_tuple[0] # 1
my_tuple[0] = 3 # Illegal! You get a TypeError
```

You can use all the same operations we saw for lists. In addition, you can unpack tubles into variables!

### Dictionaries

Think of a dictionary as a key and value pair. Each value has a key, which, instead of a numeric index, is a string that helps to identify a value. We can start with an empty list or a full list, and add elements at any time using a key.

```python
my_dict = {}
my_other_dict = {"first" : 10, "second": 7}
my_dict["my_key"] = "my_value" # {"my_key" : "my_value"}
```

If we want to access an element, we simply use its key:

```python
var_1 = my_other_dict["first"] # = 10
```

But using this method, if we don't find the key we pass, we will get an error! We have a solution though, the `get()` method:

```python
my_other_dict["third"] # KeyError
my_other_dict.get("third") # = None, no error!
```

If we want all the keys or all the values in a dictionary, we can get these as lists:

```python
my_other_dict.keys() # ["first", "second"]
my_other_dict.values() # [10, 7]
```

We can find the length of a dictionary (the number of keys) using the `len()` function.

```python
len(my_other_dict) # 2
```

We can again use the `in` keyword to check to see if a key exists in a dictionary:

```python
"first" in my_other_dict # True
```

### Sets

For you mathy folks, we can also represent sets, which remove duplicate elements from lists. We can declare these two ways:

```python
my_set = set([1,2,2,4,4,4,5]) # set([1,2,4,5])
# OR
my_set = {1,2,2,4,4,4,5} # {1,2,4,5}
```

Adding to a set is simple:

```python
my_set.add(6) # {1,2,4,5,6}
```

We can find the intersection and the union of two sets:

```python
my_intersection = {1,2,5} & {4,2,6} # {2}
my_union = {1,2,5} | {4,2,6} # {1,2,4,6}
```

We can find difference:

```python
my_difference = {1,2,3} - {2,4} # {1,3}
```

And find the length of a set

```python
len(my_difference) # 2
```

Or we can check if an element is in a set:

```python
1 in my_set # True
```

## Control Flow

What good is a program if we can't make decisions? Luckily, we have several tools at our disposal that allow us to make these decisions, which direct the way our program executes in such a way to make it meaningful.

### If/Else

We probably want the ability to do something if another condition is met. For this, we have `if`. It is here our boolean values become important:

```python
# Important: Use your tabs for whitespace! It matters!
if 4 > 3: # True
print("Yay!" # Prints!)
```

But what happens if our condition isn't true! You guessed it! `else`!

```python
if 4 != 4:
	print("This is true.")
else:
	print("This is false.") # Prints!
```

We can even add other conditions if our first one isn't met with `elif`. Notice this is different from the Java `else if`.

```python
if 4 == 3:
	print("First condition is true.")
elif 109 > 105:
	print("Second condition is true.") # Prints!
else:
	print("Third condition is true.")
```

Say we have more than one condition we want to check at the same time. We have two other words, `and` and `or`. Notice that these are different than the traditional `&&` and `||`.

```python
if 4 == 4 or 8 < 4:
	print("This is true.") # Prints!
if 4 == 4 and 8 < 4:
	print("This is true.") # Does not print!
```

### For Loops

If we want to iterate through all the elements in a list, a set, or a tuple, we can easily do this using a `for` loop.

```python
# Prints each element in the list
for i in ["Mom", "Dad", "Brother", "Sister"]:
	print(i)
```

We can also use the `range()` function combined with `for` to do some interesting things:

```python
# Range from 0 to 6
for i in range(6):
	print(i) # Prints 0, then 1, then 2, …, then 5
# Range from 1 to 7
for i in range(1,7):
	print(i) # Prints 1, then 2, then 3,…,then 6
# Range from 1 to 100, counting by three
for i in range(1,100,3):
	print(i) # Prints 1, then 4, then 7, …, then 97

```
Note: The syntax `for (x = 0; x < 10; x = x + 1)` from Java is not valid in Python.

### While Loops

While loops are similar to for loops, but instead execute until a condition is no longer true. We can do this with any boolean condition, but while loops are often used with mathematical comparisons.

```python
x = 0
while x < 4:
	print(x) # Prints 0, then 1, then 2, then 3
	x += 1

# Infinite Loop, must make sure inner loop changes condition
while True:
	print("This is an infinite loop.")
```

### Try/Except

Sometimes you will run into errors in your code. We saw this when we tried to access an index in a list that didn't exist. For this, we can use a `try/except` clause, similar to a `try/catch` statement in Java. Let's see here:

```python
my_list = [1,2,4]
try:
	print(my_list[3]) # Causes IndexError
except IndexError:
	print("This is an index error!") # Executes, prevents the stopping of the program
```

This is just an example of one error that can be caught. Other common errors include `IOError`, `ValueError`, and `TypeError`. We can catch one or more using `except` statements, or catch any remaining exceptions as a default.

```python
try:
# Some code
except IndexError:
	print("Index error!")
except IOError:
	print("IO error!")
except ValueError:
	print("Value error!")
except:
	print("Some other kind of error!")
```
This can be vital to creating robust programs that DO NOT BREAK. They should be used whenever you are doing 'risky' manipulations, such as dealing with files or user input.

# Git, Github, and Your SSH Connection

* Git is the toolkit for programming collaboration and version control.

## Introduction

* First thing first. Git is NOT the same as Github
* Git is a version control system that is used to store and navigate information
* Github is a web service that utilizes Git. There are other services similar to Github such as Bitbucket and Gitlab. Github is currently the most popular and the one we will all be using for this class.

![git does not equal github](http://1.bp.blogspot.com/-WY2YpNr3W6g/UY6tZAc-H3I/AAAAAAAABLY/xJ9x3wIY8V8/s440/Github2.png)


## Command / Necessary Git Commands

### Initializing a repository

*  `git init` - This will initialize the folder you are currently in as a git repository

### Save your changes

* `git add <filename>` - a new file / edits to a specific file will be staged and ready to be committed to a repository
* `git add .` - the dot notation will target everything in your current directory. All new / updated files will be staged
* `git commit -m <message>` - commits changes you've made to the repository. The message will be wrapped in quotes and can be anything you want. usually it will describe the updates you are committing.

### Working with Remotes

* `git remote add <remote_name> <url>` - We can connect a folder to a repo with a remote url. 
* `git remote rm <remote_name>` - This will remove a remote that was added earlier
* `git remote -v` - list all remotes
* `git push <remote_name> <branch>` - push your changes to a remote git repository
* `git pull <remote_name> <branch>` - pull and merge any changes from a remote git repository
* `git clone <url>` - copy a repository from github. This will pull everything down into a new folder

### Helpful Commands

* `git checkout -b <branch_name>` - Make your own branch on that repo so you don't interfere with the master branch
* `git checkout <branch_name>` - Move between branches inside of the repo
* `git help` - list possible git commands
* `git status` - will show you your current branch, and what files have been changed
* `git log` - show the commit history
* `git diff` - show the changes between commits / commit and working tree
* `git config --global user.name "John Doe"` - use this to attach a name to the commits. Make sure the name is in quotes
* `git config --global user.email johndoe@example.com` - use this to attach an email to commits. Make sure the email is in quotes


## Git History

* It was invented by [Linus Torvalds](https://en.wikipedia.org/wiki/Linus_Torvalds) while he was inventing Linux
* In short he invented Git to help him keep his versions organized while inventing Linux
* Feel free to [read more here](https://git-scm.com/book/en/v2/Getting-Started-A-Short-History-of-Git) or google some history on your own time. Pretty cool stuff


## What is Git?

* So what the hell is git?
* It is a version control too that you can run from your terminal
* Programmers use git so they can track the history of changes made on specific projects
* We can also use git to undo changes as far back as we want (or at least have committed to)
* Git also allows us to have multiple people working on the same projects simultaneously 
	* Similar to how google docs can have multiple people editing a file at the same time


## What is a repository?

* This is just a codebase in git. 
* Git repos will be connected to a specific folder to your laptop


## What is Github?

* A web service that utilizes Git
* Acts as a social network for programmers
	* As mentioned earlier, there are many web services that do this. Github is just the most popular
* You should all have an account on Github
* You will be able to put your own code on Github
* View your classmates codes
* And work on a project with multiple people


## Basic Git Workflow

***Making your own repo***

1. Make a new directory with `mkdir directory_name`
2. Go into that directory with change directory `cd directory name`
3. Initialize a new repository with `git init`
4. Create a README.md file or any file you want to send to GitHub `touch README.md`
5. Write something in the file and stage it `open README.md` and save something
6. Add entire directory and all files within `git add .`
7. Commit entire directory with `git commit -m "your save text"`
7. Create a new repo on your Github account by visiting your profile, clicking the `+` on the top right and creating a new repo
8. Connect that repo to your directory (make sure you are in the right folder)
	9. You can copy the `ssh` link in the directory and paste it into the initilized folder using `git remote add origin url_you_copied` 
9. Push to your repository with `git push origin master`

***Pulling down somebody elses EXISTING Github repo. Then creating your own branch***

1. Change directory into the folder you want this repository to be copied to
2. When you copy this repo it will create IT'S OWN REPO FOLDER inside the current directory
3. Go to the repo you want to copy and take the "ssh" link
4. Go back to your bash/terminal window and clone the repository using `git clone ssh_url`
5. If you list out the items in the folder `ls` you should see the new repository
6. Change directory into it `cd repo_name` 
7. Make a new branch for your own edits using `git checkout -b the_branch_name_you_want`
8. Now you can jump between your branch, the master branch, or any other branches using the `git checkout branch_name` command
9. If you made any edits and want to push up your own branch make sure you are in the branch you intend to push. You can check this using the `git status` command
10. Add branch `git add .`
11. Commit the branch files `git commit -m "my branch save"`
12. Push up YOUR branch `git push origin branch_name`
13. Now if you check the repository on github that branch should appear in the drop down menu. 

## SSH

* I mentioned `ssh` several times in the lesson above
* SSH stands for secure shell and basically allows two programs to securely connect to each other
* In this case we are putting an ssh key onto our github account so Github knows that specific device is okay. 
* This way we don't have to put in our login credentials / authenticate every time we pull or push to a github repo
* To add an ssh key to your github account [follow these directions](https://help.github.com/articles/generating-an-ssh-key/)
* There are four main steps
	* Checking if you have a ssh key
	* Generating a new ssh key and adding it to the ssh-agent
	* Add the ssh key to your github account
	* Testing the SSH connection

