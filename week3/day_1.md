## Introduction

All data problems begin with a question and hopefully end with a clear answer or insight. From there, the next step is getting your data. As a Data Scientist, you'll spend an incredible amount of time and skills on acquiring, prepping, cleaning, and normalizing your data. In this section, we'll review some of the best tools used in the rhelm of data acquisition. 

But first, let's go into the differences between Data Acquisition, Preparation, and Cleaning. 

### Data Acquisition

Data Acquisition is the process of getting your data, hence the term <i>acquisition</i>. Data doesn't come out of nowhere, so the very first step of any data science problem is going to be getting the data in the first place. 

### Data Preparation

Once you have the data, it might not be in the best format to work with. You might have scraped a bunch of data from a website, but need it in the form of a dataframe to work with it in an easier manner. This process is called data preparation - preparing your data in a format that's easiest to form with.

### Data Cleaning

Once your data is being stored or handled in a proper manner, that might still not be enough. You might have missing values or values that need normalizing. These inconsistencies that you fix before analysis refers to data cleaning. 

## Reading, Writing, and Handling Data Files

The simplest way of acquiring data is downloading a file - either from a website, straight from your desktop, or elsewhere. Once the data is downloaded, you'll open the files for reading and possible writing. 

### CSV files

Very often, you'll have to work with CSV files. A csv file is a comma-separated values file stores tabular data in plain text. 

In the following examples, we'll be working with NBA data, which you can download from [here](https://github.com/ByteAcademyCo/data-acq/blob/master/nba.csv).

#### CSV 

Python has a csv module, which you can utilize to work with CSV files.

``` python
import csv
```

Then, with the following 4 lines, you can print each row

``` python
with open('nba.csv', ‘rt') as f:
	reader = csv.reader(f)
	for row in reader:
		print(row)
```

Fairly straightforward, but let's see how else we can accomplish this. 

#### Pandas

Alternatively, you can use Pandas. Pandas is great for working with CSV files because it handles DataFrames. 

We begin by importing the needed libraries: pandas.

``` python
import pandas as pd
```

Then we use pandas to read the CSV file and show the first few rows. 
``` python
task_data = pd.read_csv("nba.csv")
task_data.head()
```
As you can see, by using pandas, we're able to fasten the process of viewing our data, as well as view it in a much more readable format. 

#### Programming

We've just gone through how to read CSV files in Python. But how do you do this in R? Pretty simply, actually. R has built in functions to handle CSV files, so you don't even have to use a library to accomplish what we just did with Python.

``` R
data <- read.csv("nba.csv")
```

### JSON

Because HTTP is a protocol for transferring text, the data you request through a web API (which we'll go through soon enough) needs to be serialized into a string format, usually in JavaScript Object Notation (JSON). JavaScript objects look quite similar to Python dicts, which makes their string representations easy to interpret:

```
{ 
 "name" : "Lesley Cordero",
 "job" : "Data Scientist",
 "topics" : [ "data", "science", "data science"] 
}
```

Python has a module sepcifically for working with JSON, called `json`, which we can use as follows:

``` python
import json
serialized = """ { 
 "name" : "Lesley Cordero",
 "job" : "Data Scientist",
 "topics" : [ "data", "science", "data science"] 
} """
```
Next, we parse the JSON to create a Python dict, using the json module: 
 
``` python
deserialized = json.loads(serialized)
print(deserialized)
```

And we get this output:

```
{'name': 'Lesley Cordero', 'job': 'Data Scientist', 'topics': ['data', 'science', 'data science']}
```

#### jsonlite

Now, in R, working with JSON can be a bit more complicated. Unlike Python, R doesn't have a data type that resembles JSON closely (dictionaries in Python). So we have to work with what we do have, which is lists, vectors, and matrices.

Working with the same data from the Python example, we have:

``` python
serialized = '{ 
 "name" : "Lesley Cordero",
 "job" : "Data Scientist",
 "topics" : [ "data", "science", "data science"] 
} '
```

Now, if we want to properly load this into R, we'll be using the `jsonlite` library. 

``` R
library("jsonlite")
```
Once we've loaded the library, we'll use the `fromJSON` function to convert this into a data type R is more familiar with: <b>lists</b>.

``` R
l <- fromJSON(serialized, simplifyVector=TRUE)
```

Notice that `simplifyVector` is set to `TRUE`. When simplifyMatrix is enabled, JSON arrays containing equal-length sub-arrays simplify into a matrix. 

And to convert this back to JSON, we type:

``` R
toJSON(l, pretty=TRUE)
```
Not too horrible!

## APIs

There are several ways to extract information from the web. Use of APIs, Application Program Interfaces, is probably the best way to extract data from a website. APIs are especially great if your data is constantly changing. Many websites have public APIs providing data feeds via JSON or some other format. 

There are a number of ways to access these APIs from Python. In order to get the data, we make a request to a webserver, hence an easy way is to use the `requests` package. 

### GET request

There are many different types of requests. The most simplest is a GET request. GET requests are used to retrieve your data. In Python, you can make a get request to get the latest position of the international space station from the `OpenNotify` API.

``` python
import requests
response = requests.get("http://api.open-notify.org/iss-now.json")
```

If you print `response.status_code`, you'll get 

``` 
200
```

Which brings us to status codes. 

### Status Codes

What we just printed was a status code of `200`. Status codes are returned with every request made to a web server and indicate what happened with a request. The following are the most common types of status codes:

- `200` - everything worked as planned!
- `301` - the server is redirecting you to anotehr endpoint (domain).
- `400` - it means you made a bad request by not sending the right data or some other error.
- `401` - you're not authenticated, which means you don't have access to the server.
- `403` - this means access is forbidden. 
- `404` - whatever you tried to access wasn't found. 

Notice that if we try to access something that doesn't exist, we'll get a `404` error:

``` python
response = requests.get("http://api.open-notify.org/iss-pass")
print(response.status_code)
```

Let's try a get request where the status code returned is `404`. 
``` python
response = requests.get("http://api.open-notify.org/iss-pass.json")
print(response.status_code)
```
Like we mentioned before, this indicated a bad request. This is because it requires two parameters, as you can see [here](http://open-notify.org/Open-Notify-API/ISS-Pass-Times/). 

We set these with an optional `params` variable. You can opt to make a dictionary and then pass it into the `requests.get` function, like follows:

``` python 
parameters = {"lat": 40.71, "lon": -74}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
```

You can skip the variable portion with the following instead: 
``` python
response = requests.get("http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74")
```

If you print the content of both of these with `response.content`, you'll get: 
```
b'{\n  "message": "success", \n  "request": {\n    "altitude": 100, \n    "datetime": 1441417753, \n    "latitude": 40.71, \n    "longitude": -74.0, \n    "passes": 5\n  }, \n  "response": [\n    {\n      "duration": 329, \n      "risetime": 1441445639\n    }, \n    {\n      "duration": 629, \n      "risetime": 1441451226\n    }, \n    {\n      "duration": 606, \n      "risetime": 1441457027\n    }, \n    {\n      "duration": 542, \n      "risetime": 1441462894\n    }, \n    {\n      "duration": 565, \n      "risetime": 1441468731\n    }\n  ]\n}'
```
This is pretty messy, but luckily, we can clean this up into JSON with:

``` python
data = response.json()
```

And we get: 
``` 
{'response': [{'risetime': 1441456672, 'duration': 369}, {'risetime': 1441462284, 'duration': 626}, {'risetime': 1441468104, 'duration': 581}, {'risetime': 1441474000, 'duration': 482}, {'risetime': 1441479853, 'duration': 509}], 'message': 'success', 'request': {'latitude': 37.78, 'passes': 5, 'longitude': -122.41, 'altitude': 100, 'datetime': 1441417753}}
```

### APIs with R

So far we've seen APIs with Python. Let's take a look on how you can use R to do some simple API calls. We'll be working with the `httr` library and the EPDB API, which we load in the next three lines:

``` R
library("httr")
url  <- "http://api.epdb.eu"
path <- "eurlex/directory_code"
```

With `httr`, you can make GET requests, like this:
``` R
raw.result <- GET(url=url, path=path)
```
If you check out what `raw.result` is, you'll see the following information:
```
Response [http://api.epdb.eu/eurlex/directory_code/]
  Date: 2017-02-06 21:41
  Status: 200
  Content-Type: application/json
  Size: 121 kB
```

Now let's pull the name entities from this GET request:
``` R
names(raw.result)
```

Which results in, as we'd expect:
```
 [1] "url"         "status_code" "headers"     "all_headers" "cookies"    
 [6] "content"     "date"        "times"       "request"     "handle" 
```

You can extract each of the entitites above with the `$` character, like this:
``` R
raw.result$status_code
```

which results in a `200` status code!
