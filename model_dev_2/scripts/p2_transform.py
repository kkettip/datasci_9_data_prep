import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

## get data raw

#datalink = 'https://data.cdc.gov/api/views/i2vk-mgdh/rows.csv?accessType=DOWNLOAD'
#df = pd.read_csv(datalink)

df = pd.read_pickle('model_dev_2/data/raw/covid.pkl')

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
    'data_as_of',
    'start_date',
    'end_date',
    'group',
    'mmwr_week',
    'weekending_date',
    'demographic_type',
    'provisional',
    'suppressed',
    'stratificationcategory1',
    'stratificationcategory2',
    'percent_deaths',
    'state'  
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
df['deaths'] = df['deaths'].astype(int)

df['total_deaths'] = df['total_deaths'].astype(int)

#df['percent_deaths'] = df['percent_deaths'].astype(int)

#check for conversion and length of dataframe
df.dtypes 
len(df)


## stratification1 --> will need to encode this
#stratification1: gender
#check for counts in each of the catgories
df.demographic_values.value_counts()

#remove "Overall" category
#df = df[df['stratification1'] != 'Overall' ]

#check for counts in each of the catgories and to see if "Overall" is removed
#df.stratification1.value_counts()

## perform ordinal encoding on stratification1
enc = OrdinalEncoder()
enc.fit(df[['demographic_values']])
df['demographic_values'] = enc.transform(df[['demographic_values']])

## create dataframe with mapping
df_mapping_demographic_values = pd.DataFrame(enc.categories_[0], columns=['demographic_values'])
df_mapping_demographic_values['demographic_values_ordinal'] = df_mapping_demographic_values.index
df_mapping_demographic_values.head(5)

# save mapping to csv
df_mapping_demographic_values.to_csv('model_dev_2/data/processed/mapping_demographic_values.csv', index=False)





## stratification2 --> will need to encode this
#stratification2: race and ethnicity
df.pathogen.value_counts()


#remove "Overall" category
#df = df[df['stratification2'] != 'Overall' ]

#check for counts in each of the catgories and to see if "Overall" is removed
#df.stratification2.value_counts()

## perform ordinal encoding on stratification2
enc = OrdinalEncoder()
enc.fit(df[['pathogen']])
df['pathogen'] = enc.transform(df[['pathogen']])

## create dataframe with mapping
df_mapping_pathogen = pd.DataFrame(enc.categories_[0], columns=['pathogen'])
df_mapping_pathogen['pathogen_ordinal'] = df_mapping_pathogen.index
df_mapping_pathogen.head(5)
# save mapping to csv
df_mapping_pathogen.to_csv('model_dev_2/data/processed/mapping_pathogen.csv', index=False)




## locationdesc --> will need to encode this
#locationdesc: counties
df.locationdesc.value_counts()

## perform ordinal encoding on locationdesc
enc = OrdinalEncoder()
enc.fit(df[['locationdesc']])
df['locationdesc'] = enc.transform(df[['locationdesc']])

## create dataframe with mapping
df_mapping_locationdesc = pd.DataFrame(enc.categories_[0], columns=['locationdesc'])
df_mapping_locationdesc['locationdesc_ordinal'] = df_mapping_locationdesc.index
df_mapping_locationdesc.head(5)
# save mapping to csv
df_mapping_locationdesc.to_csv('model_dev_1/data/processed/mapping_locationdesc.csv', index=False)





# ## locationabbr --> will need to encode this
# #locationabbr: US states
# df.locationabbr.value_counts()

# ## perform ordinal encoding on locationabbr
# enc = OrdinalEncoder()
# enc.fit(df[['locationabbr']])
# df['locationabbr'] = enc.transform(df[['locationabbr']])

# ## create dataframe with mapping
# df_mapping_locationabbr = pd.DataFrame(enc.categories_[0], columns=['locationabbr'])
# df_mapping_locationabbr['locationabbr_ordinal'] = df_mapping_locationabbr.index
# df_mapping_locationabbr.head(5)
# # save mapping to csv
# df_mapping_locationabbr.to_csv('model_dev_1/data/processed/mapping_locationabbr.csv', index=False)



#### save temporary csv files for model testing
df.head(1000).to_csv('model_dev_1/data/processed/heart_disease_1k.csv', index=False)
df.sample(5000).to_csv('model_dev_1/data/processed/heart_disease_5k.csv', index=False)
df.sample(10000).to_csv('model_dev_1/data/processed/heart_disease_10k.csv', index=False)