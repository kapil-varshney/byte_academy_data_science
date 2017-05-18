# Assignment x

## Part 1 

## Querying a SQL Database

In this challenge you are given a SQL database with data inside. You will need to mine it for important information.

Open it up in terminal by typing
```bash
$ sqlite3 sitemetrics.db
```
To see the existing tables and columns, use the .schema command. Map this by hand or on SQL designer so you have a greater understanding of what we're working with.

Answer the following questions with the results and the sql you wrote to get it.

### How many people are from California?  
14
`select count(*) from users where state="CA";`

### Who has the most page views? How many do they have, and where are they from?


### Who has the least page views? How many do they have and where are they from?


### Who are the most recent visitors to the site?(at least 3)


### Who was the first visitor?


### Who has an email address with the domain 'horse.edu'?


### How many people are from the city Graford?


### What are the names of all the cities that start with the letter V, in alphabetical order?


### What are the names and home cities for people searched for the word "drain"?


### How many times was "trousers" a search term?


### What were the search terms used by visitors who last visited on August 22 2014?


### What was the most frequently used search term by people from Idaho?


### What is the name of user 391, and what are his search terms?


## Part 2 - Build a Movie Database

* Plan out your ERD using the [SQL schema designer](http://ondras.zarovi.cz/sql/demo/)
* Create a database that will house the information for all the Marvel Cinematic Universe movies below. 
* Feel free to pick your own movies.
* You will need THREE table for actors/actresses, a table for movies, a relationship table called movie_cast

***MARVEL MOVIES***

* Iron Man = ["Robert Downey Jr.", "Terrence Howard", "Gwyneth Paltrow", Jeff Bridges"]
* Iron Man 2 = ["Robert Downey Jr.", "Don Cheadle", "Gwyneth Paltrow", "Scarlett Johansson"]
* Iron Man 3 = ["Robert Downey Jr.", "Don Cheadle", "Gwyneth Paltrow"]
* Captain America = ["Chris Evans"]
* Captain America 2 = ["Scarlett Johansson", "Chris Evans", "Anthony Mackie"]
* Captain America 3 = ["Robert Downey Jr.", "Scarlett Johansson","Don Cheadle", "Chris Evans", "Anthony Mackie", "Jeremy Renner"]
* The Incredible Hulk = ["Edward Norton", "Liv Tyler", "Tim Roth", "William Hurt"]
* Thor = ["Chris Hemsworth", "Jeremy Renner", "Natalie Portman", "Tom Hiddleson", "Anthony Hopkins"]
* Thor 2 = ["Chris Hemsworth", "Tom Hiddleson", "Anthony Hopkins"]
* Avengers = ["Robert Downey Jr.", "Scarlett Johansson", "Chris Evans","Chris Hemsworth", "Gwyneth Paltrow", "Jeremy Renner", "Mark Ruffalo", "Tom Hiddleson"]
* Avengers 2 = ["Robert Downey Jr.", "Scarlett Johansson", "Don Cheadle", "Chris Evans", "Anthony Mackie", "Chris Hemsworth", "Jeremy Renner", "Mark Ruffalo"]
