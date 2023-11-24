# datasci_9_data_prep

## Dataset 1: Heart-Disease-Stroke-Prevention/Heart-Disease-Mortality-Data-Among-US-Adults
`https://data.cdc.gov/Heart-Disease-Stroke-Prevention/Heart-Disease-Mortality-Data-Among-US-Adults-35-by/i2vk-mgdh`

The dataset contains data on heart disease and mortality for US adults ages 35 and over. It includes data from 2013 to 2015. The rates are age-standardized and the county rates are smooth. 

Columns of the dataset includes: `['year', 'locationabbr', 'locationdesc', 'geographiclevel', 'datasource', 'class', 'topic', 'data_value', 'data_value_unit', 'data_value_type', 'data_value_footnote_symbol', 'data_value_footnote', 'stratificationcategory1', 'stratification1', 'stratificationcategory2', 'stratification2', 'topicid', 'locationid', 'location_1']` 

This dataset contains 59076 rows and 19 columns. 

The indepedent variables for this dataset are `'stratification1'`, `'stratification2'`, `'year'`, `'locationabbr'`, and `'locationdesc'`. 

The dependent variable for this dataset is `'data_value'`.


## Dataset 2: Percent-of-Deaths-for-COVID-19-Influenza
`https://data.cdc.gov/Health-Statistics/Provisional-Percent-of-Deaths-for-COVID-19-Influen/53g5-jf7x`

This dataset provides the percentage of deaths for COVID-19, Influenza, and RSV for those living in the US. It includes data from 2019 to 2023.

Columns of the dataset includes: `['data_as_of', 'start_date', 'end_date', 'group', 'year', 'month','mmwr_week', 'weekending_date', 'state', 'demographic_type','demographic_values', 'pathogen', 'deaths', 'total_deaths','percent_deaths', 'provisional', 'suppressed']` 

This dataset contains 1000 rows and 15 columns. 

The indepedent variables for this dataset are `'start_date'`, `'end_date'`, `'group'`, `'year'`, `'month'`, `'mmwr_week'`, `'weekending_date'`, `'state'`, `'demographic_type'`,`'demographic_values`, and `'pathogen'`. 

The dependent variables for this dataset are `'deaths'`, `'total_deaths'`, and `'percent_deaths'`.


## Regression is the intended machine learning tasks for both datasets.

## Clean and transform the data
Steps
1. Standardize the column names by lowercasing and removing white spaces
2. Determine the columns that would be used for the machine learning experiments.
3. Drop columns that would not be included
4. Check for outliers and exclude if applicable
5. Drop rows with missing or NaN data
6. Convert floats to integers
7. Encode the categorical variables to numerical values
8. Scale the data to give equal weights to the categorical values
9. Save and create cleaned data files


# Challenges
1. Issue encountered with pushing to github. Received error that file size exceeds Github file limit. This issue was resolved by resetting the repo to its initial state.
2. Categorical names did not show in map generated after encoding. Issue was resolved by rerunning the code.






