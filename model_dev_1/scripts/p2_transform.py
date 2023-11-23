import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

## get data raw

#datalink = 'https://data.cdc.gov/api/views/i2vk-mgdh/rows.csv?accessType=DOWNLOAD'
#df = pd.read_csv(datalink)

df = pd.read_pickle('model_dev_1/data/raw/heart_disease.pkl')

## get column names
df.columns

## clean column names 
# convert column names to all lower case letters, replace white spaces with _
df.columns = df.columns.str.lower().str.replace(' ', '_')

#get column names and check for conversion
df.columns

#check shape of dataframe for number of rows and columns
df.shape

df.sample(5)

## get data types
df.dtypes # nice combination of numbers and strings/objects
len(df)

## drop columns
to_drop = [
    'geographiclevel',
    'datasource',
    'class',
    'topic',
    'year',
    'data_value_unit',
    'data_value_type',
    'data_value_footnote_symbol',
    'data_value_footnote',
    'stratificationcategory1',
    'stratificationcategory2',
    'topicid',
    'locationid',
    'location_1'
]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')

#check shape of dataframe for number of rows and columns
df.shape
df.sample(5)

#drop rows with NaN
df = df.dropna()

df.shape
df.sample(5)

#convert float to integer
df['data_value'] = df['data_value'].astype(int)

#check for conversion and length of dataframe
df.dtypes 
len(df)


## stratification1 --> will need to encode this
#stratification1: gender
#check for counts in each of the catgories
df.stratification1.value_counts()

#remove "Overall" category
df = df[df['stratification1'] != 'Overall' ]

#check for counts in each of the catgories and to see if "Overall" is removed
df.stratification1.value_counts()

## perform ordinal encoding on county
enc = OrdinalEncoder()
enc.fit(df[['stratification1']])
df['stratification1'] = enc.transform(df[['stratification1']])

## create dataframe with mapping
df_mapping_stratification1 = pd.DataFrame(enc.categories_[0], columns=['stratification1'])
df_mapping_stratification1['stratification1_ordinal'] = df_mapping_stratification1.index
df_mapping_stratification1.head(5)

# save mapping to csv
df_mapping_stratification1.to_csv('model_dev_1/data/processed/mapping_stratification1.csv', index=False)





## stratification2 --> will need to encode this
#stratification2: race and ethnicity
df.stratification2.value_counts()


#remove "Overall" category
df = df[df['stratification2'] != 'Overall' ]

#check for counts in each of the catgories and to see if "Overall" is removed
df.stratification2.value_counts()

## perform ordinal encoding on county
enc = OrdinalEncoder()
enc.fit(df[['stratification2']])
df['stratification2'] = enc.transform(df[['stratification2']])

## create dataframe with mapping
df_mapping_stratification2 = pd.DataFrame(enc.categories_[0], columns=['stratification2'])
df_mapping_stratification2['stratification2_ordinal'] = df_mapping_stratification2.index
df_mapping_stratification2.head(5)
# save mapping to csv
df_mapping_stratification2.to_csv('model_dev_1/data/processed/mapping_stratification2.csv', index=False)