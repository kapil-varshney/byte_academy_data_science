import pandas as pd
import numpy as np
#import math
import string

data= pd.read_csv('test.csv')



def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)
data['bio'] = data['bio'].apply(remove_punctuation)


def split_it(X):
    return X.split()
data['bio'] = data['bio'].apply(split_it)




states= pd.read_csv('state_abbreviations.csv')  #upload states
def change_state(X):
    return states[states['state_abbr'] == X]['state_name'].values[0]
data['state'] = data['state'].apply(change_state)



data['start_date_description']= data.apply(lambda _: '', axis=1)  # add blank column

def poor_dates(X):  # will add these to the start_date_desciption
    if '.' in X:
        return X
    elif '/' in X:       #split dates in form M/D/Y
        if len(X.split('/')) < 3:  #year or month/year
            return X  
        else:
            return ''
    elif '-' in X:       #split dates in form M-D-Y
        if len(X.split('-')) < 3:
            return X
        else:
            return ''
    else:                 #split dates in form M D, Y
        if len(X.split(' ')) < 3:
            return X
        else:
            return ''

data['start_date_description'] = data['start_date'].apply(poor_dates)

def standard_dates(X):  # change dates for dates with enough information
    if '.' in X:
        return X
    elif '/' in X:
        if len(X.split('/')) < 3:
            return X
        else:
            return pd.to_datetime(X).strftime('%Y-%m-%d')  #change to form Y-M-D
    elif '-' in X:
        if len(X.split('-')) < 3:
            return pd.to_datetime(X).strftime('%Y-%m-%d')
        else:
            return X
    else:
        if len(X.split(' ')) < 3:
            return X
        else:
            return pd.to_datetime(X).strftime('%Y-%m-%d')

data['start_date'] = data['start_date'].apply(standard_dates)

# see if its possible to combine two arguements into one ?




#output csv 
data.to_csv('solution.csv')
