import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

## get data 
#https://data.cdc.gov/Health-Statistics/Provisional-Percent-of-Deaths-for-COVID-19-Influen/53g5-jf7x

datalink = 'https://data.cdc.gov/resource/53g5-jf7x.csv'

df = pd.read_csv(datalink)


## get column names
df.columns

#get data size and 5 random rows
df.size
df.sample(5)


## save as csv to model_dev_1/data/raw
df.to_csv('model_dev_2/data/raw/covid.csv', index=False)

## save as pickle to model_dev_1/data/raw
df.to_pickle('model_dev_2/data/raw/covid.pkl')