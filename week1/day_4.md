## 6.0 More Goodies
While the previous five sections provide a very thorough introduction to the basics of Python, this section will provide you with some extra information about the language. You will find many of these features to be very helpful to you when you start developing your web application and as your Python programs become more complex.

### 6.1 Modules
As you start to write longer and more complex programs, you can imagine there will be lots of functions you want to be able to use. Along with that, your files will begin to get very long, to the point where you may lose control of your program. For this, we have modules.

Modules are regular Python files that contain definitions of functions that can later be imported for use in another file. Many very important functions are given to us in the Python language, all we need to do is important them! For example, Python comes with a default module called `math`, which provides many important function for mathematical operations. If we wanted to use these functions, all we have to do is import it:

```python
# your_math.py
import math

answer_1 = math.factorial(4) # 24
answer_2 = math.sqrt(4) # 2
```

If you don't want to have to say `math` everytime you use a function, you could import the specific functions you want to use:

```python
from math import pow, fabs 

answer_1 = pow(4,2) # 16
answer_2 = fabs(-5) # 5
```

You can even import all the functions from a module using an asterisk:

```python
from my_math import *
```
Though, remember that this is generally a bad idea, as you could run in to collisions with function names.

Also, we can store constants in modules! We simply access them with the dot syntax.

```python
import math

pi = math.pi #3.1415...
```

### 6.2 User Input

Nearly every useful program takes some kind of input from the user. Python gives us several ways to do this, the first of which is text from the command line.

We do this using the function `input()`, which takes a line of text typed in by a user on the command line interface and stores it into a variable. For example:

```python
name = input("What is your name? ")
```

In name will be stored whatever the user types in. Within the parentheses, you can put a prompt, which is meant to prompt the user to instruct them on what to input.

We can also read from a file to get input from the user:

```python
# Get the file name from the user
file_name = input("Enter a file name: ")
# Open the file for reading
file = open(file_name)
# Print the contents of the file
print (file.read())
# Remember to close the file
file.close()
```

If we wanted to, we could even write to the file:

```python
file_name = input("Enter a file name: ")
file = open(file_name)
file.write("Hello File!")
```

However, there is another way to read files that prevents having to close the file after you finish. It is generally preferred to use this method:

```python
# 'r' stands for 'read'
with open('my_file', 'r') as f:
	data = f.read()
```
