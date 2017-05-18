## Homework Assignment 5

Submit the following by Thursday, February 2nd, 2017 at 6:10pm. 

### Part 1 

We can create a dictionary that maps people to sets of their friends. For example, we might say that a dictionary "friends" looks like:

``` python
friends["Lesley"] = set(["Helen", "Jesus", "Menna"])
friends["Jacob"] = set(["Ojas", "Martin", "Eli"])
friends["Shreyas"] = set(["Ojas", "Jesus", "Martin", "Eli"])
friends["Ojas"] = set(["Jacob", "Shreyas", "Jesus"])
friends["Jesus"] = set(["Lesley", "Eli", "Keala"])
```
With this in mind, write the function `create_FOF(friends)` that takes a dictionary which maps people to sets of their direct friends and returns a new dictionary `friends_of_friends`, which maps all the same people to sets that consist ONLY of friends of friends.

For example, the output would look like: 

``` python
friends_of_friends["Lesley"] = set(["Eli", "Keala"])
friends_of_friends["Jacob"] = set(["Shreyas", "Jesus"])
friends_of_friends["Shreyas"] = set(["Jacob", "Keala", "Lesley"])
friends_of_friends["Ojas"] = set(["Martin", "Eli", "Martin", "Lesley", "Keala"])
friends_of_friends["Jesus"] = set(["Helen", "Menna"])
```

### Part 2

You are building the course registration system for Byte Academy in the hope of making it better. The registration system needs to be able to respond to some queries. 

In following questions, `d` is the registration record (represented by a dictionary) whose keys are course codes (like the string "Python 101") and where the corresponding value of each key is a list of student names (strings) that are enrolled in that course.

(a) Write a function `count_all_enrolled_students(d)`, that takes the registration record and return the count the total number of enrolled students in all classes. Note that students may enroll in multiple classes.

(b) Some classes have labs associated with it. Students must enroll in both the class and its associated lab. Write a function `find_invalid_registrations(d, course_code, lab_code)`, where `course_code` is the course code for class that has an associated lab and lab_code is the course code for the lab associated with `course_code`. It returns a list of students who have not properly registered for the class indicated by `course_code`.

(c) In the final week, several classes may have the same exam time. Write a function `notify_exam_conflict(d, course_codes)` that takes the registration record d and a list of course codes which share the same exam time slot, and returns a list of students who will have an exam time conflict.

(d) You want to hold a Python workshop for everyone at Byte Academy, except to those who are taking Python 101 (they are awesome at Python programming already!) Write a function `hold_workshop(d)` that returns a list of students who are not enrolled in Python 101 right now. Note that Python 101 may not be offered every semester, so there is no guarantee that it is in the registration record. In case you don't know, the course code for Python 101 is "Python 101". 
