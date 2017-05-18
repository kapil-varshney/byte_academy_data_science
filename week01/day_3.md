## 5.0 Classes

You may have heard the term 'Object-Oriented Programming.' If you are unfamiliar, this basically means that things should be represented as 'objects,' each of which stores data, called fields, and associated operations, called methods. In Python and other object-oriented programming languages, we instantiate objects by defining classes.

<a id="defineclass"></a>
### 5.1 Defining a Class

To define a class in Python, we use the reserved word `class`:

```python
class Animal(object):
```

In the parentheses, we put the class from which we are subclassing. In most cases, this will be the object class.

### 5.2 Class Attributes

Each object is going to have attributes, pieces of data that are stored along with it. Some will be shared by all objects of the class. For example, all Animals belong to the same kingdom, Animalia. We call these attributes class attributes.

```python
class Animal(object):
	kingdom = "Animalia"
```

### 5.3 Initializer

However, classes will also have data that is specific to each instance of that class. For example, each Animal will have its own species or its own name. These attributes are called instance attributes, or sometimes instance variables. We handle these in what is called an initializer, which provides a way for us to create new objects.

We write an initializer like this:

```python
class Animal(object):
	kingdom = "Animalia"

	def __init__(self, name, species):
		self.name = name
		self.species = species
```

Notice the double underscore character before and after the word `init`. This signifies an initializer, which is known in Java as a constructor. Also notice the word `self`. The initializer always takes this as its first parameter, as it refers to the object being created.

In addition to `self`, we can pass any number of parameters, in this case the name and species.  Both `self.name` and `self.species` refer to instance attributes, which each instance of the class has.

To create a new Animal object, simply use the initializer as follows, passing each parameter in order:


```python
# Sequentially uses positioning of parameters
animal_1 = Animal("Bobo", "Monkey")

# Access attributes
animal_1.name # = "Bobo"
animal_1.species # = "Monkey"

# Change attributes
animal_1.name = "Sugar"
animal_1.name # = "Sugar"
```

<a id="instance"></a>
### 5.4 Instance Methods
Instance methods are functions defined within a class that deal specifically with the instance. They often modify or return instance attributes. They, like initializers, must take `self` as their first parameter.

```python
class Animal(object):
	# Class Attribute
	kingdom = "Animalia"

	# Initializer
	def __init__(self, name, species):
		self.name = name
		self.species = species

	# Instance Method
	def speak(self, msg):
		return("%s says %s" % (self.name, msg))

# Instantiate an Animal object
animal = Animal("Butch", "Dog")
# Call our instance method
animal.speak("woof") # = "Butch says woof"

```

###5.5 Class Methods
Class methods, like class attributes, are shared for all instances of the class. Thus, they cannot access any instance attributes or methods. They always take the `cls`, the calling class, as their first argument.

```python
class Animal(object):
	# Class Attribute
	kingdom = "Animalia"

	# Initializer
	def __init__(self, name, species):
		self.name = name
		self.species = species

	# Instance Method
	def speak(self, msg):
		return("%s says %s" % (self.name, msg))

	# Class Method
	@classmethod
	def get_kingdom(cls):   # "cls" refers to the class that calls get_kingdom
		return(cls.kingdom)

# Instantiate an Animal object
animal = Animal("Polly", "Parrot")
# Call our class method w/ the class
Animal.get_kingdom() # = "Animalia"
```
<a id="static"></a>
### 5.6 Static Methods
Static methods, unlike class methods or instance methods, need neither a class or an instance. They are called using the class name.

```python
class Animal(object):
	# Class Attribute
	kingdom = "Animalia"

	# Initializer
	def __init__(self, name, species):
		self.name = name
		self.species = species

	# Instance Method
	def speak(self, msg):
		return("%s says %s" % (self.name, msg))

	# Class Method
	@classmethod
	def get_kingdom(cls):
		return(cls.kingdom)

	# Static Method
	@staticmethod
	def boo():
		return("BOO!")

# Call our static method
Animal.boo() # = "BOO!"
```


### 5.7 Inheritance

Inheritance is used to indicate that one class will get most or all of its features from a parent class. 

When you are doing this kind of specialization, there are three ways that the parent and child classes can interact:

- Actions on the child imply an action on the parent.
- Actions on the child override the action on the parent.
- Actions on the child alter the action on the parent.

Here I will show you the implicit actions that happen when you defi ne a function in the parent, but not in the child. 

Here we have the Parent class:
``` python
class Parent(object):
	def implicit(self):
		print("PARENT implicit()")
```

Here, we define the Child class:
``` python
class Child(Parent):
	pass
```
Now, we're just calling the two classes.
``` python
dad = Parent()
son = Child()
```
Now, let's see what happens when we call the implicit methods on each:
``` python
dad.implicit()
son.implicit()
```
And we get:
``` 
PARENT implicit()
PARENT implicit()
```
Notice how even though I’m calling `son.implicit()` and the Child class doesn't have an `implicit` function defined, it still works and calls the one defined in Parent. This shows you that, if you put functions in a parent class, then all child classes will automatically get those features. 

<a id="exercise5"></a>
### 5.8 Exercise 5

Write a class called `Human`.  It should have a constructor that takes in a height and weight (in inches and pounds respectively), and they should be stored in instance variables.

Write an instance method called `get_bmi` that returns the BMI of the human.  Note that BMI is `weight/(height^2) * 703`.

```python
# Write the "Human" class here


# Don't edit anything below this comment
# --------------------------------------

my_human = Human(120, 67)
print("my_human's height and weight is %s and %s respectively, and it's BMI is %s." % (my_human.height, my_human.weight, my_human.get_bmi()))

# Should print "my_human's height and weight is 120 and 67 respectively, and it's BMI is 3.27090277778."
```
