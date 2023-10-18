---
title: Regression Modelling with Geographic Coordinates
tags: [ MySQL, Random Forest Regression, KMeans Clustering, RBF Kernel, Geographic Coordinate System(GCS) ]
---

![030_regression_modelling_001.png](../assets/images/030_regression_modelling_001.png){:class="img-responsive"}

# Motivation and Introduction

Previously, a [dashboard](http://3.96.175.190:8501) was built to visualize Bixi trips between 2014 and 2021. As a bicycle rental company, daily operations likely include the movement of bicycles between stations, whether to improve bicycle availability at popular stations or to make space at fully docked stations. In an effort to improve this process of restocking or emptying bicycle docking stations, a regression model was built and trained using the location of stations to predict how many trips start from each station on an annual basis. 

The foundation of this model was based on the assumption that the location of a bicycle station affects how busy a station will be, and thus how much effort is needed for restocking or emptying. Furthermore, an assumption was made that although the annual number of trips changes, the percentage of annual trips for each station remains constant throughout the years, thus avoiding the need to use time-series models.

In the final random forest regression model, using RBF Kernel, each station's latitude and longitude were converted to similarity measures to each of the 25 cluster centers identified through KMeans clustering. These similarity measures were used to predict the percentage of annual trips that start from each station. An intuitive understanding would be how locations are identified to be important based on its proximity to other landmarks.

The final model was tested to have a root mean square error 0.05608% percentage of annual trips, with a 95% confidence interval of 0.04770% and 0.06336%. These results indicate that the model only works well for stations with larger percentage of annual trips (>33% percentile). However, at an annual level, the final model's predictions were within 0.1 million trips for all years except 2021. This suggests that the model has not "learned" the one time effects of Covid-19, and, that the errors for stations below the 33% percentile of annual trips were insignificant compared to the annual trips for busier stations.

# Overview of Setup

## Docker Environments

To replicate the environment used to perform this analysis:
1. Fork the Github [repository](https://github.com/andrewyewcy/velocipede) on a local machine with Docker and Docker-compose installed.
2. Run below line in Terminal within the file directory of the repository


```python
# Run below line in terminal within the folder that contains the forked repo
docker-compose -f dev_setup.yaml up
```

Instructions on how to access Jupyter will be generated in the Terminal. Detailed installation documentation can be found within the README of the repository.

## Import Packages and Define Functions

The models and modelling pipelines were built from the `Scikit-learn` package.


```python
# For general data processing
import numpy as np
import pandas as pd
import math
import time

# Custom functions for measuring time of queries and to print lists
from notebook_functions.timer import print_time
from notebook_functions.print import print_list

# Packages for data visualization
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from adjustText import adjust_text # For fine tuning position of text above points in scatterplot 
import plotly.express as px

# For the engine that connects Python and MySQL
import sqlalchemy

# For building custom sklearn classes
from sklearn.base import BaseEstimator, TransformerMixin

# For data preprocessing and feature engineering
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

# Models for testing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Model hyperparameter tuning and performance evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy import stats

# For saving the model
import joblib
```

# Import Data From MySQL

A [`SQLAlchemy`](https://www.sqlalchemy.org) engine was created and used to import data from MySQL into the Jupyter notebook. Ideally, the `dotenv` packaged would be used to load hidden credentials for access to MySQL. However, since this database is only available locally, the defaults were inputted plainly in engine creation. 


```python
# Create the connection engine to MySQL database called 'velocipede'
engine = sqlalchemy.create_engine(
    "mysql+mysqldb://root:rootroot@sql:3306/velocipede"
)
```

The below SQL query was used to transform and extract the number of rides starting at each station for each year along with station details such as station latitude and longitude. A join table, `jt_rides_stations`, was used given the `rides` table and `stations` table have a many to many relationship. Explanations are in included as comments in the SQL query below.


```python
# SQL statement that gathers the number of rides per station per year along with station details
sql_stmt = """
# A CTE counting the number of rides per station per year
WITH a AS (
SELECT
    COUNT(ride_id) AS rides,
    YEAR(CONVERT_TZ(start_dt_utc,"+00:00","-04:00")) AS year,
    start_stn_code,
    data_source
FROM
    rides
WHERE
    company = "Bixi"
GROUP BY
    year,
    start_stn_code,
    data_source
)

# Main SQL statement that adds station details to the CTE using join table
SELECT
    SUM(a.rides) AS rides,
    a.year,
    a.start_stn_code,
    c.stn_name,
    c.stn_lat,
    c.stn_lon
FROM
    a

# left join #1 to join-table between rides and stations tables
LEFT JOIN
    jt_rides_stations AS b
ON
    a.data_source = b.ride_files

# left join #2 to stations table
LEFT JOIN
    stations AS c
ON
    b.station_files = c.data_source
    AND a.start_stn_code = c.stn_code
GROUP BY
    a.year,
    a.start_stn_code,
    c.stn_name,
    c.stn_lat,
    c.stn_lon  
"""
```


```python
# Establish MySQL connection through engine then run SQL query
with print_time():
    with engine.connect() as conn:
        df = pd.read_sql(
            sqlalchemy.text(sql_stmt),
            con = conn
        )
```

    Time taken: 468.9603 seconds.


The query was observed to take more than 5 minutes to execute on the local machine as it involves 40.9 million rows of data. This demonstrates MySQL's limitations in dealing with the scale of data, which can be overcome by using column oriented data file formats like [`Apache Parquet`](https://parquet.apache.org) along with a PySpark cluster in future iterations.

The data resulting from the MySQL query was briefly examined before being saved as processed data to avoid re-running the SQL query again.


```python
print(f"The data has {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"The number of null values is {df.isna().sum().sum()}.")

# Visually examine the first few rows of data
df.head()
```

    The data has 5394 rows and 6 columns.
    The number of null values is 9.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rides</th>
      <th>year</th>
      <th>start_stn_code</th>
      <th>stn_name</th>
      <th>stn_lat</th>
      <th>stn_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4478.0</td>
      <td>2021</td>
      <td>10</td>
      <td>Métro Angrignon (Lamont /  des Trinitaires)</td>
      <td>45.446910</td>
      <td>-73.603630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6871.0</td>
      <td>2021</td>
      <td>100</td>
      <td>16e avenue / St-Joseph</td>
      <td>45.552870</td>
      <td>-73.565631</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1093.0</td>
      <td>2021</td>
      <td>1000</td>
      <td>Parc Champdoré (Champdoré / de Lille)</td>
      <td>45.569176</td>
      <td>-73.636293</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4469.0</td>
      <td>2021</td>
      <td>1002</td>
      <td>Vendôme / Sherbrooke</td>
      <td>45.475963</td>
      <td>-73.607234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3179.0</td>
      <td>2021</td>
      <td>1003</td>
      <td>Parc Tino-Rossi (18e avenue / Everett)</td>
      <td>45.564811</td>
      <td>-73.595804</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save data to processed data folder for easy access
df.to_csv(f"../12_processed_data/{time.strftime('%Y%m%d-%H%M')}_rides_per_station_per_year.csv")
```


```python
# Uncomment this line for loading the saved processed data
# df = pd.read_csv(f"../12_processed_data/20231003-1926_rides_per_station_per_year.csv", index_col = 0)
```

# Exploratory Data Analysis (EDA)

Each row within the loaded data represents the number of Bixi bicycle rental trips(rides) that started at each station for each year. As a station's location and name may remain constant or change slightly across the years, this means that a given station name may appear multiple times within the data. The data was explored for null values and any trends that maybe helpful for modelling later on.

## Null values


```python
# Select rows where any value in null
cond = df.isna().any(axis=1)
df.loc[cond]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rides</th>
      <th>year</th>
      <th>start_stn_code</th>
      <th>stn_name</th>
      <th>stn_lat</th>
      <th>stn_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3151</th>
      <td>3800.0</td>
      <td>2019</td>
      <td>6034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3247</th>
      <td>5941.0</td>
      <td>2019</td>
      <td>6708</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5225</th>
      <td>3.0</td>
      <td>2019</td>
      <td>MTL-ECO5.1-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Given only 3 stations had null station details in 2019, these 3 stations were dropped from analysis.


```python
df = df.loc[~cond]

# Examine the data after dropping rows with null values
print(f"The data has {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"The number of null values is {df.isna().sum().sum()}.")
```

    The data has 5391 rows and 6 columns.
    The number of null values is 0.


## Rides by Year

Next, the numbers of Bixi trips were summed and plotted against years to identify annual trends.


```python
# Reaggregate the data by year to count the number of rides per year
plot_df = df.groupby(
    by = ["year"]
).agg(
    rides = ("rides","sum")
)

# Use built in pandas plot method for quick plotting
plot_df.plot(
    kind = "bar",
    legend = None
)

# Tidy up plot
plt.xlabel("Year")
plt.ylabel("Number of Rides (millions)")
plt.title("Figure 1. Number of Bixi Rides by Year", y = -0.25)
sns.despine()
plt.show()
```


    
![png](../assets/images/output_32_0.png){:class="img-responsive"}
    


An annual increasing trend was observed, with the year `2020` being an exception most likely due to [Covid-19](https://en.wikipedia.org/wiki/COVID-19) restrictions. To avoid training the model to fit special cases like Covid-19, the year `2020` was dropped from analysis. Furthermore, the year `2022` was also known to have incomplete data due to membership details being omitted starting 2022. As such, the years `2020` and `2022` were ommitted from analysis.


```python
# Create condition to identify rows for 2020 and 2022
cond = df["year"].isin([2020,2022])

# Remove rows based on condition
df = df.loc[~cond]

# Examine the data after dropping rows with null values
print(f"The data has {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"The number of null values is {df.isna().sum().sum()}.")
```

    The data has 3926 rows and 6 columns.
    The number of null values is 0.


## Illogical Latitude and Longitudes

After checking annual rides, the latitude and longitudes of each station were checked to identify any erroneous station locations that are likely not representative of any location within the city of Montreal. 


```python
# Use built in function to examine distribution of columns
df[["stn_lat","stn_lon"]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stn_lat</th>
      <th>stn_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3926.000000</td>
      <td>3926.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>45.507946</td>
      <td>-73.566578</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.743071</td>
      <td>1.158884</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.000000</td>
      <td>-73.746873</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>45.501402</td>
      <td>-73.602610</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>45.523276</td>
      <td>-73.578050</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>45.539632</td>
      <td>-73.564776</td>
    </tr>
    <tr>
      <th>max</th>
      <td>45.651406</td>
      <td>-1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Note that the minimum value for station latitude was -1, which is definitely not in Montreal.


```python
# The rows where station latitude was less than 0 were identified and examined
cond = df.stn_lat < 0
display(df[cond])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rides</th>
      <th>year</th>
      <th>start_stn_code</th>
      <th>stn_name</th>
      <th>stn_lat</th>
      <th>stn_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>704</th>
      <td>9749.0</td>
      <td>2021</td>
      <td>856</td>
      <td>Smith / Peel</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>


As the this occured to only 1 row of data, the row in question was dropped.


```python
df = df.loc[~cond]

# Examine the data after dropping rows with null values
print(f"The data has {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"The number of null values is {df.isna().sum().sum()}.")
```

    The data has 3925 rows and 6 columns.
    The number of null values is 0.


## Number of Stations per Year

Next the number of stations per year was examined. Given that the annual number of rides has increased steadily over the years in Figure 1, it was expected that the number of bicycle stations have also increased throughout the years to meet increasing demand.


```python
# Reaggregate the data by year to count the number of rides per year
plot_df = df.groupby(
    by = ["year"]
).agg(
    stations = ("start_stn_code","nunique")
)

# Use built in pandas plot method |or quick plotting
plot_df.plot(
    kind = "bar",
    legend = None
)

plt.xlabel("Year")
plt.ylabel("Number of Stations")
plt.title("Figure 2. Number of Stations by Year", y = -0.25)
sns.despine()
plt.show()
```


    
![png](../assets/images/output_44_0.png){:class="img-responsive"}
    


The number of stations were observed to increase as the years progress.

## % of Trips by Station (Target Feature)

To avoid the need of employing time-series algorithms to deal with the increasing number of rides through the years, the popularity of a station was expressed as a percentage of total rides for the year, with the assumption that the relative popularity of a station's location compared to other stations is roughly constant across the years. That way, the output of the model can be multiplied by the total number of rides expected for a given year to get the number of rides.


```python
# Create new target feature: percentage of annual rides
df["pct_of_annual_rides"] = df["rides"] / df.groupby(by = ["year"])["rides"].transform("sum")
```

Then, the stations were ranked annually by descending number of trips starting at each station.


```python
# Rank the stations by descending rides starting at each station for each year
df["annual_stn_rank"] = df.groupby(by = ["year"])["rides"].rank(method = "dense", ascending = False)
df["annual_stn_rank"] = df["annual_stn_rank"].astype(int)
```

Finally, the dataset was filtered and transformed to examine the percentage of annual rides held by the top 10 stations across the years regardless of station location.


```python
# Filter for only the top 10 stations
cond = df["annual_stn_rank"] < 11

# Pivot the data so that the % of rides for the top 10 stations are compared across the years
rank_df = df.loc[cond].pivot(
    columns = "year",
    index = "annual_stn_rank",
    values = "pct_of_annual_rides"
)

# Visually examine the percentage of rides for the top 10 stations for each year
# Create and apply color map horizontally to dataframe
cm = sns.light_palette("blue", as_cmap = True)
s = rank_df.style.background_gradient(cmap = cm,axis = 1)
display(s)
```


<style type="text/css">
#T_1f500_row0_col0, #T_1f500_row1_col0, #T_1f500_row2_col0, #T_1f500_row3_col2, #T_1f500_row4_col2, #T_1f500_row5_col2, #T_1f500_row6_col2, #T_1f500_row7_col2, #T_1f500_row8_col0, #T_1f500_row9_col0 {
  background-color: #0000ff;
  color: #f1f1f1;
}
#T_1f500_row0_col1 {
  background-color: #4242fc;
  color: #f1f1f1;
}
#T_1f500_row0_col2 {
  background-color: #a7a7f7;
  color: #000000;
}
#T_1f500_row0_col3, #T_1f500_row2_col3 {
  background-color: #6b6bfa;
  color: #f1f1f1;
}
#T_1f500_row0_col4 {
  background-color: #ababf6;
  color: #000000;
}
#T_1f500_row0_col5, #T_1f500_row1_col6, #T_1f500_row2_col6, #T_1f500_row3_col6, #T_1f500_row4_col6, #T_1f500_row5_col5, #T_1f500_row6_col5, #T_1f500_row7_col5, #T_1f500_row8_col6, #T_1f500_row9_col6 {
  background-color: #f0f0f3;
  color: #000000;
}
#T_1f500_row0_col6 {
  background-color: #ebebf3;
  color: #000000;
}
#T_1f500_row1_col1 {
  background-color: #3636fc;
  color: #f1f1f1;
}
#T_1f500_row1_col2 {
  background-color: #2727fd;
  color: #f1f1f1;
}
#T_1f500_row1_col3, #T_1f500_row5_col4 {
  background-color: #6969fa;
  color: #f1f1f1;
}
#T_1f500_row1_col4 {
  background-color: #8686f8;
  color: #f1f1f1;
}
#T_1f500_row1_col5 {
  background-color: #a3a3f7;
  color: #f1f1f1;
}
#T_1f500_row2_col1 {
  background-color: #5b5bfa;
  color: #f1f1f1;
}
#T_1f500_row2_col2 {
  background-color: #1b1bfe;
  color: #f1f1f1;
}
#T_1f500_row2_col4 {
  background-color: #7c7cf9;
  color: #f1f1f1;
}
#T_1f500_row2_col5 {
  background-color: #e0e0f4;
  color: #000000;
}
#T_1f500_row3_col0 {
  background-color: #2b2bfd;
  color: #f1f1f1;
}
#T_1f500_row3_col1 {
  background-color: #4141fc;
  color: #f1f1f1;
}
#T_1f500_row3_col3 {
  background-color: #5656fb;
  color: #f1f1f1;
}
#T_1f500_row3_col4 {
  background-color: #6262fa;
  color: #f1f1f1;
}
#T_1f500_row3_col5 {
  background-color: #c3c3f5;
  color: #000000;
}
#T_1f500_row4_col0 {
  background-color: #1010fe;
  color: #f1f1f1;
}
#T_1f500_row4_col1 {
  background-color: #1f1ffd;
  color: #f1f1f1;
}
#T_1f500_row4_col3 {
  background-color: #3e3efc;
  color: #f1f1f1;
}
#T_1f500_row4_col4 {
  background-color: #8585f8;
  color: #f1f1f1;
}
#T_1f500_row4_col5 {
  background-color: #c2c2f5;
  color: #000000;
}
#T_1f500_row5_col0 {
  background-color: #1818fe;
  color: #f1f1f1;
}
#T_1f500_row5_col1 {
  background-color: #3b3bfc;
  color: #f1f1f1;
}
#T_1f500_row5_col3 {
  background-color: #0b0bfe;
  color: #f1f1f1;
}
#T_1f500_row5_col6 {
  background-color: #e5e5f3;
  color: #000000;
}
#T_1f500_row6_col0 {
  background-color: #1212fe;
  color: #f1f1f1;
}
#T_1f500_row6_col1 {
  background-color: #6464fa;
  color: #f1f1f1;
}
#T_1f500_row6_col3 {
  background-color: #0707ff;
  color: #f1f1f1;
}
#T_1f500_row6_col4 {
  background-color: #b3b3f6;
  color: #000000;
}
#T_1f500_row6_col6 {
  background-color: #e3e3f3;
  color: #000000;
}
#T_1f500_row7_col0 {
  background-color: #1414fe;
  color: #f1f1f1;
}
#T_1f500_row7_col1 {
  background-color: #6a6afa;
  color: #f1f1f1;
}
#T_1f500_row7_col3 {
  background-color: #2e2efd;
  color: #f1f1f1;
}
#T_1f500_row7_col4 {
  background-color: #a2a2f7;
  color: #f1f1f1;
}
#T_1f500_row7_col6 {
  background-color: #ededf3;
  color: #000000;
}
#T_1f500_row8_col1 {
  background-color: #5353fb;
  color: #f1f1f1;
}
#T_1f500_row8_col2 {
  background-color: #2929fd;
  color: #f1f1f1;
}
#T_1f500_row8_col3 {
  background-color: #5757fb;
  color: #f1f1f1;
}
#T_1f500_row8_col4 {
  background-color: #bdbdf5;
  color: #000000;
}
#T_1f500_row8_col5 {
  background-color: #e8e8f3;
  color: #000000;
}
#T_1f500_row9_col1 {
  background-color: #4545fb;
  color: #f1f1f1;
}
#T_1f500_row9_col2 {
  background-color: #3939fc;
  color: #f1f1f1;
}
#T_1f500_row9_col3 {
  background-color: #4e4efb;
  color: #f1f1f1;
}
#T_1f500_row9_col4 {
  background-color: #9d9df7;
  color: #f1f1f1;
}
#T_1f500_row9_col5 {
  background-color: #b4b4f6;
  color: #000000;
}
</style>
<table id="T_1f500">
  <thead>
    <tr>
      <th class="index_name level0" >year</th>
      <th id="T_1f500_level0_col0" class="col_heading level0 col0" >2014</th>
      <th id="T_1f500_level0_col1" class="col_heading level0 col1" >2015</th>
      <th id="T_1f500_level0_col2" class="col_heading level0 col2" >2016</th>
      <th id="T_1f500_level0_col3" class="col_heading level0 col3" >2017</th>
      <th id="T_1f500_level0_col4" class="col_heading level0 col4" >2018</th>
      <th id="T_1f500_level0_col5" class="col_heading level0 col5" >2019</th>
      <th id="T_1f500_level0_col6" class="col_heading level0 col6" >2021</th>
    </tr>
    <tr>
      <th class="index_name level0" >annual_stn_rank</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1f500_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_1f500_row0_col0" class="data row0 col0" >0.014540</td>
      <td id="T_1f500_row0_col1" class="data row0 col1" >0.012901</td>
      <td id="T_1f500_row0_col2" class="data row0 col2" >0.010376</td>
      <td id="T_1f500_row0_col3" class="data row0 col3" >0.011894</td>
      <td id="T_1f500_row0_col4" class="data row0 col4" >0.010299</td>
      <td id="T_1f500_row0_col5" class="data row0 col5" >0.008547</td>
      <td id="T_1f500_row0_col6" class="data row0 col6" >0.008694</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_1f500_row1_col0" class="data row1 col0" >0.010599</td>
      <td id="T_1f500_row1_col1" class="data row1 col1" >0.009685</td>
      <td id="T_1f500_row1_col2" class="data row1 col2" >0.009940</td>
      <td id="T_1f500_row1_col3" class="data row1 col3" >0.008825</td>
      <td id="T_1f500_row1_col4" class="data row1 col4" >0.008339</td>
      <td id="T_1f500_row1_col5" class="data row1 col5" >0.007838</td>
      <td id="T_1f500_row1_col6" class="data row1 col6" >0.006525</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_1f500_row2_col0" class="data row2 col0" >0.010221</td>
      <td id="T_1f500_row2_col1" class="data row2 col1" >0.008819</td>
      <td id="T_1f500_row2_col2" class="data row2 col2" >0.009796</td>
      <td id="T_1f500_row2_col3" class="data row2 col3" >0.008549</td>
      <td id="T_1f500_row2_col4" class="data row2 col4" >0.008301</td>
      <td id="T_1f500_row2_col5" class="data row2 col5" >0.006743</td>
      <td id="T_1f500_row2_col6" class="data row2 col6" >0.006488</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_1f500_row3_col0" class="data row3 col0" >0.008999</td>
      <td id="T_1f500_row3_col1" class="data row3 col1" >0.008656</td>
      <td id="T_1f500_row3_col2" class="data row3 col2" >0.009719</td>
      <td id="T_1f500_row3_col3" class="data row3 col3" >0.008314</td>
      <td id="T_1f500_row3_col4" class="data row3 col4" >0.008117</td>
      <td id="T_1f500_row3_col5" class="data row3 col5" >0.006535</td>
      <td id="T_1f500_row3_col6" class="data row3 col6" >0.005788</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_1f500_row4_col0" class="data row4 col0" >0.008632</td>
      <td id="T_1f500_row4_col1" class="data row4 col1" >0.008437</td>
      <td id="T_1f500_row4_col2" class="data row4 col2" >0.008854</td>
      <td id="T_1f500_row4_col3" class="data row4 col3" >0.008030</td>
      <td id="T_1f500_row4_col4" class="data row4 col4" >0.007099</td>
      <td id="T_1f500_row4_col5" class="data row4 col5" >0.006289</td>
      <td id="T_1f500_row4_col6" class="data row4 col6" >0.005678</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_1f500_row5_col0" class="data row5 col0" >0.007795</td>
      <td id="T_1f500_row5_col1" class="data row5 col1" >0.007424</td>
      <td id="T_1f500_row5_col2" class="data row5 col2" >0.008046</td>
      <td id="T_1f500_row5_col3" class="data row5 col3" >0.007926</td>
      <td id="T_1f500_row5_col4" class="data row5 col4" >0.006965</td>
      <td id="T_1f500_row5_col5" class="data row5 col5" >0.005556</td>
      <td id="T_1f500_row5_col6" class="data row5 col6" >0.005677</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row6" class="row_heading level0 row6" >7</th>
      <td id="T_1f500_row6_col0" class="data row6 col0" >0.007740</td>
      <td id="T_1f500_row6_col1" class="data row6 col1" >0.006860</td>
      <td id="T_1f500_row6_col2" class="data row6 col2" >0.007942</td>
      <td id="T_1f500_row6_col3" class="data row6 col3" >0.007866</td>
      <td id="T_1f500_row6_col4" class="data row6 col4" >0.006015</td>
      <td id="T_1f500_row6_col5" class="data row6 col5" >0.005354</td>
      <td id="T_1f500_row6_col6" class="data row6 col6" >0.005501</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row7" class="row_heading level0 row7" >8</th>
      <td id="T_1f500_row7_col0" class="data row7 col0" >0.007199</td>
      <td id="T_1f500_row7_col1" class="data row7 col1" >0.006471</td>
      <td id="T_1f500_row7_col2" class="data row7 col2" >0.007371</td>
      <td id="T_1f500_row7_col3" class="data row7 col3" >0.006973</td>
      <td id="T_1f500_row7_col4" class="data row7 col4" >0.005993</td>
      <td id="T_1f500_row7_col5" class="data row7 col5" >0.005321</td>
      <td id="T_1f500_row7_col6" class="data row7 col6" >0.005356</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row8" class="row_heading level0 row8" >9</th>
      <td id="T_1f500_row8_col0" class="data row8 col0" >0.007104</td>
      <td id="T_1f500_row8_col1" class="data row8 col1" >0.006456</td>
      <td id="T_1f500_row8_col2" class="data row8 col2" >0.006778</td>
      <td id="T_1f500_row8_col3" class="data row8 col3" >0.006428</td>
      <td id="T_1f500_row8_col4" class="data row8 col4" >0.005639</td>
      <td id="T_1f500_row8_col5" class="data row8 col5" >0.005300</td>
      <td id="T_1f500_row8_col6" class="data row8 col6" >0.005233</td>
    </tr>
    <tr>
      <th id="T_1f500_level0_row9" class="row_heading level0 row9" >10</th>
      <td id="T_1f500_row9_col0" class="data row9 col0" >0.006951</td>
      <td id="T_1f500_row9_col1" class="data row9 col1" >0.006302</td>
      <td id="T_1f500_row9_col2" class="data row9 col2" >0.006418</td>
      <td id="T_1f500_row9_col3" class="data row9 col3" >0.006217</td>
      <td id="T_1f500_row9_col4" class="data row9 col4" >0.005481</td>
      <td id="T_1f500_row9_col5" class="data row9 col5" >0.005264</td>
      <td id="T_1f500_row9_col6" class="data row9 col6" >0.004693</td>
    </tr>
  </tbody>
</table>



Here, the assumption that the percentage of annual rides for each station being constant was observed to be roughly applicable for the years 2014 to 2018, with the years 2019 and beyond showing a deviation from the assumption. Specifically, in 2014 the top 1 station represented 1.45% of annual rides compared to only 0.85% in 2019.


```python
# Plot the % of rides taken by the top 10 stations for each year
(rank_df.sum()*100).plot()

# Add in labels
plt.xlabel("Year")
plt.ylabel("% of annual rides for top 10 stations")
plt.title("Figure 3. Percentage of Annual Rides Represented by Top 10 Stations", y = -0.25)
sns.despine()
plt.show()
```


    
![png](../assets/images/output_54_0.png){:class="img-responsive"}
    


Taking the sum of the top 10 stations revealed that the percentage of annual rides represented by the top 10 stations reduces as the years go on, with the observed dilution likely due to more stations being introduced with each new year. 

That being said, for the keeping the model simple in this analysis, the assumption that the percentage of annual per station remains constant was upheld with note for improvement in future iterations.

## Station Location

A brief scatterplot of all station locations was visually examined to confirm that the stations do indeed represent a rough map of the city of Montreal.


```python
# Create scatter plot of station location
df.plot(
    kind = "scatter",
    y = "stn_lat",
    x = "stn_lon",
    c = "cornflowerblue",
    alpha = 0.2
)

# tidy up chart
plt.xlabel(f"Station Longitude")
plt.ylabel(f"Station Latitude")
plt.title(f"Figure 4. Scatterplot of Bixi Stations", y = -0.25)
sns.despine()
plt.show()
```


    
![png](../assets/images/output_58_0.png){:class="img-responsive"}
    


The opacity represents the density fo stations across the years. The dark blue areas represent the downtown core of Montreal while the blank patch centered at -73.60 and 45.50 represents Mount Royal, confirming that the data is indeed stations located within Montreal.

## Correlations

As the station latitudes and longitudes will be preprocessed later using KMeans clustering and RBF Kernel, observed correlations should be taken with a grain of salt.


```python
# Define columns for correlation checking
cols = ["pct_of_annual_rides","stn_lat","stn_lon"] 

# Plot a scatterplot matrix for defined columns
scatter_matrix(df[cols],figsize = (12,8))
plt.title("Figure 5. Correlation Scatterplot Matrix", x = -0.5, y = -0.50)
plt.show()
```


    
![png](../assets/images/output_62_0.png){:class="img-responsive"}
    


The top left histogram suggests a right skewed distribution of the target feature. Looking at the middle and right scatterplots in the top row suggest that stations in the city center of Montreal have higher percentage of annual rides.

## Summary of Data Cleaning

| Data Cleaning Operation | Rows Removed | Rows Remaining |
|-------------------------|--------------|----------------|
| -                       | -            | 5,394          |
| Drop null values        | 3            | 5,391          |
| Remove 2020 and 2022    | 2,095        | 3,296          |
| Station latitude < 0    | 1            | 3,295          |

# Data Preprocessing

## Stratification

The target feature of prediction, `pct_of_annual_rides`, was observed to have a non-uniform, right skewed distribution. To ensure consistent target feature distribution across the test and training sets, the target feature, being continuous and numerical, was first converted into a categorical variable for stratification. As a balance between the number categories and the number of rows for each category, the data was split into 3 parts based onto their quantiles using the `pd.cut` method.


```python
# Define target_feature or y
y = df["pct_of_annual_rides"]

# Define boundaries to cut up numerical target feature into categorical bins.
# 3 parts, so four boundaries
cat_bin_boundaries = [
    0, 
    y.quantile(q=0.33).round(5), 
    y.quantile(q=0.66).round(5), 
    np.inf
]

# Define labels for each bin
cat_bin_labels = [
    "00 - 33% percentile", 
    "34 - 66% percentile",
    "67 - 100% percentile"
]

# Categorize target feature based on defined boundaries and labels
df["pct_cat"] = pd.cut(
    y,
    bins = cat_bin_boundaries,
    labels = cat_bin_labels
)

# Plot a count of the categorized target feature
df["pct_cat"].value_counts().sort_index().plot.bar(rot = 0)
plt.ylabel("Number of data points")
plt.xlabel("Target Feature Categories")
plt.title(f"Figure 6. 'pct_of_annual_rides' categorized by Percentiles", y = -0.2)
sns.despine()
plt.show()
```


    
![png](../assets/images/output_69_0.png){:class="img-responsive"}
    


The figure demonstrates how `pct_of_annual_rides` was split into 3 even categories for stratification. For further visual verification, the original distribution of the target feature was mapped according to the 3 defined categories.


```python
# List of colors for mapping categories
color_list = ["#606c38", "#dda15e", "#bc6c25","#283618"]

# Create figure object with 2 subplots
fig, ax = plt.subplots(1, 2, sharey = True, figsize = (15,6))

# Define min and max values to define bins for histogram
min = y.min()
max = y.max()
bins = np.linspace(min,max,101)

# In the first subplot, plot the target feature distribution without categories
ax[0].hist(
    y,
    bins,
    histtype = "bar",
    color = "cornflowerblue",
    lw = 0
)

# In the second subplot, loop over the defined category boundaries, shading each one
for index, cat_lower_bound in enumerate(cat_bin_boundaries):
    # Shading for first category
    if index == 1:
        mask = (y >= min) & (y <= cat_lower_bound)
        ax[1].hist(
            y[mask], 
            bins, 
            histtype='bar', 
            color = color_list[index], 
            lw=0,
            label = cat_bin_labels[0]
        )

    # Shading for last category
    elif index == len(cat_bin_boundaries)-1:
        mask = (y >= cat_bin_boundaries[index-1]) & (y <= max)
        ax[1].hist(
            y[mask], 
            bins, 
            histtype='bar', 
            color = color_list[index], 
            lw=0,
            label = cat_bin_labels[-1]
        )

    # Shading for any other category (first boundary doesnt generate a category
    elif index != 0:
        mask = (y >= cat_bin_boundaries[index-1]) & (y <= cat_lower_bound)
        ax[1].hist(
            y[mask], 
            bins, 
            histtype='bar', 
            color=color_list[index], 
            lw=0,
            label = cat_bin_labels[index-1]
        )
    else:
        pass

# Label and legends
ax[0].set_xlabel("pct_of_annual_rides")
ax[0].set_ylabel("Station Count")
ax[0].set_title("Original Distribution", y = -0.15)
ax[1].set_title("Categorized Distribution", y = -0.15)
ax[1].legend()
fig.suptitle("Figure 7. Distribution of Target Feature with Categories", y = -0.05)
sns.despine()

plt.show()

```


    
![png](../assets/images/output_71_0.png){:class="img-responsive"}
    


The main peak with a right skew was split into 3 equal categories based on percentileP. These categories were used to maintain constant distribution across the train and test sets.

## Train Test Split (Remainder, Validation, Test)

For model training and hyperparameter tuning, the data were split into 4 sets:
1) `test` set: This data was reserved until all training and tuning was completed for final reporting.
2) `remainder` set: `train` + `validation` set, used for hyperparameter tuning after model is selected.
3) `validation` set: Used for selecting the most suitable model for hyperparameter tuning.
4) `train` set: Used for selecting the most suitable model for hyperparameter tuning.


```python
# Reset index to make clean index after EDA
df.reset_index(inplace = True, drop = True)
```

The index of the data was reset for a clean row count after data cleaning. Then, the data was split with stratification using the 3 categories defined based on quantiles.


```python
# Split data into remainder and test set
remainder_df, test_df = train_test_split(
    df,
    test_size = 0.1,
    stratify = df["pct_cat"],
    random_state = 42
)

# Further split remainder into validation and test
train_df, validation_df = train_test_split(
    remainder_df,
    test_size = 0.1,
    stratify = remainder_df["pct_cat"],
    random_state = 42
)
```


```python
print(f"original data   : {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"test data       : {test_df.shape[0]} rows and {test_df.shape[1]} columns.")
print(f"remainder data  : {remainder_df.shape[0]} rows and {remainder_df.shape[1]} columns.")
print(f"validation data : {validation_df.shape[0]} rows and {validation_df.shape[1]} columns.")
print(f"train data      : {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
```

    original data   : 3925 rows and 9 columns.
    test data       : 393 rows and 9 columns.
    remainder data  : 3532 rows and 9 columns.
    validation data : 354 rows and 9 columns.
    train data      : 3178 rows and 9 columns.


Now, a simple function was written to check if distribution is constant based on the categories:


```python
# Define a simple function to check proportions by each category
def check_proportions(data, col):
    return np.round(data[col].value_counts() / len(data) *100,2)

compare_props = pd.DataFrame({
    "Original %"   : check_proportions(df,"pct_cat"),
    "Test %"       : check_proportions(test_df,"pct_cat"),
    "Remainder %"  : check_proportions(remainder_df,"pct_cat"),
    "Validation %" : check_proportions(validation_df,"pct_cat"),
    "Train %"      : check_proportions(train_df,"pct_cat")
    }
)

display(compare_props.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Original %</th>
      <th>Test %</th>
      <th>Remainder %</th>
      <th>Validation %</th>
      <th>Train %</th>
    </tr>
    <tr>
      <th>pct_cat</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67 - 100% quantile</th>
      <td>34.09</td>
      <td>34.10</td>
      <td>34.09</td>
      <td>34.18</td>
      <td>34.08</td>
    </tr>
    <tr>
      <th>00 - 33% quantile</th>
      <td>33.07</td>
      <td>33.08</td>
      <td>33.07</td>
      <td>33.05</td>
      <td>33.07</td>
    </tr>
    <tr>
      <th>34 - 66% quantile</th>
      <td>32.84</td>
      <td>32.82</td>
      <td>32.84</td>
      <td>32.77</td>
      <td>32.85</td>
    </tr>
  </tbody>
</table>
</div>


Based on the table above, it can be verified that the data has been split into various sets with the distribution of the target feature, `pct_of_annual_rides` preserved.

# Model Selection

`stn_lat` and `stn_lon`, were the only 2 independent features available for prediction of the target variable `pct_of_annual_rides`. As latitude and longitude are descriptors of location, using them as continuous numerical features would be misrepresentative as a location with larger latitude is not necessarily larger or more important. Rather, the proximity of each station to cluster centers would be a better representation of the intuitive understanding of location. For example, a station being near a cluster center representing Mount Royal Park would be a better way to justify that station having more rides annually.

To this end, the [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering method was used to find cluster center for stations, with the similarity of each station to each cluster center measured using the [Radial Basis Function(RBF)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html).
- KMeans was chosen given the relatively small dataset and its simplicity of only 1 hyperparameter for tuning.
- RBF Kernel was chosen as the method for measuring the distance between each station and each cluster center because of its ability to determine how fast the similarity measure falls off using its hyperparameter, gamma.

Thus, KMeans and RBF Kernel were used to transform latitude and longitude into numerical features that can be fed into different models to predict `pct_of_annual_rides`. Although clustering metrics such as `inertia` and `silhouette score` can be used to measure how tight the clusters are, ultimately, the KMeans and RBF Kernel hyperparameters were tuned using Root Mean Square Error (RMSE) from the regression models.

## Cluster Similarity Class

To begin, object oriented programming (OOP) was used to create a `ClusterSimilarity` class, which means that it will inherit the features of Scikit-Learn's API, such as `fit` and `fit_transform`. Although not essential, this step makes building pipelines for hyperparameter tuning much neater.


```python
# Create a class that combines KMeans with rbf_kernel to measure each station's similarity to the cluster center
class ClusterSimilarity(BaseEstimator, TransformerMixin):

    # Default number of KMeans clusters is 10, default RBF Kernel gamma is 1
    def __init__(self, n_clusters = 10, gamma = 1.0, random_state = None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y = None, sample_weight = None):
        self.kmeans_ = KMeans(self.n_clusters, random_state = self.random_state, n_init = "auto")
        self.kmeans_.fit(X, sample_weight = sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma = self.gamma)

    def get_feature_names_out(self, names = None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

Below, the newly created `ClusterSimilarity` class was tested with 10 clusters. Note how the new class inherited the attributes of KMeans without explicit definition.


```python
# Initiate a cluster similarity object
cluster_simil = ClusterSimilarity(
    n_clusters = 10,
    gamma = 1,
    random_state = 42
)

# Fit the cluster similarity object to the latitude and longitude columns
similarities = cluster_simil.fit_transform(remainder_df[["stn_lat","stn_lon"]])

# Initiate figure object
plt.figure()

# Plot the cluster centers by accessing the kmeans cluster center attribute
plt.plot(
    cluster_simil.kmeans_.cluster_centers_[:,0], # Latitude of cluster centre
    cluster_simil.kmeans_.cluster_centers_[:,1], # Longitude of cluster centre
    linestyle  = "",
    color      = "black",
    marker     = "X",
    markersize = 10,
    label      = "Cluster centers"
)

# Plot the latitude and longitude of stations in the remainder data
remainder_df.plot(
    kind  = "scatter",
    x     = "stn_lat",
    y     = "stn_lon",
    c     = "cornflowerblue",
    alpha = 0.2, # To deal with overlapping stations
    ax    = plt.gca()
)

# Tidy up plot
plt.legend()
plt.xlabel("Station Latitude")
plt.ylabel("Station Longitude")
sns.despine()
plt.title("Figure 8. Example of Cluster Centers Generated by KMeans", y = -0.2)
plt.show()
```


    
![png](../assets/images/output_88_0.png){:class="img-responsive"}
    


As expected, the new class also inherits the `inertia` attribute, which is a measure of the sum of squared distances from each point to the cluster centers as demonstrated below:


```python
# Initiate empty lists to store values
k_val = []       # Number of cluster centers
inertia_val = [] # Inertia for each number of cluster centers

# Iterate through 15 values of cluster centers
for k in range(1,16):

    # Create ClusterSimilarity object with defined number of clusters
    cluster_simil = ClusterSimilarity(
        n_clusters = k,
        gamma = 1,
        random_state = 42
    )

    # Fit the Cluster Similarity object
    cluster_simil.fit_transform(remainder_df[["stn_lat","stn_lon"]])

    # Append attributes
    k_val.append(k)
    inertia_val.append(cluster_simil.kmeans_.inertia_)

# Visualize inertia values using scree plot
plt.figure()
plt.plot(k_val, inertia_val, marker="o")

# Define axis
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Figure 9. Scree Plot for Various K-means Cluster Centers', y = -0.2)
sns.despine()

plt.show()
```


    
![png](../assets/images/output_90_0.png){:class="img-responsive"}
    


Although the scree plot shows that 10 clusters is relatively low on inertia compared to just 1 or 2 cluster centers, the final hyperparameter tuning will be done later using `RMSE` and `GridSearchCV`.

## Preprocessing Column Transformer

Before selecting the model, the `ClusterSimilarity` class was combined with other preprocessing methods in a `ColumnTransformer`. A simple imputer using the median and a standard scaler were used to process numerical columns. As only two columns, each station's latitude and longitude, were fed into the column transformer, the simple imputer and standard scaler were not used in this analysis, and were created for future analysis.


```python
# Create ClusterSimilarity object
cluster_simil = ClusterSimilarity(
    n_clusters=10, 
    gamma=1, 
    random_state=42
)

# Create a pipeline that imputes any null values then feeds it through a standard scaler
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
    )

# Combine numerical pipeline and ClusterSimilarity object into a column transformer.
preprocessing = ColumnTransformer(
    [
        ("stn_loc", cluster_simil, ["stn_lat","stn_lon"])
    ],
    remainder = num_pipeline    
)
```

The transformer was tested using the remainder data.


```python
# Test the column transformer pipeline
remainder_prepared = preprocessing.fit_transform(remainder_df[["stn_lat", "stn_lon"]])

# Examine data before column transformer
print(f"Before preprocessing pipeline")
display(remainder_df[["stn_lat", "stn_lon"]].head())

# Examine data after column transformer
print(f"After preprocessing pipeline")

# Extract column names from preprocessing pipeline
processed_remainder_df = pd.DataFrame(
    remainder_prepared,
    columns = preprocessing.get_feature_names_out(),
    index = remainder_df.index
)

display(processed_remainder_df.head())
```

    Before preprocessing pipeline



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stn_lat</th>
      <th>stn_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1424</th>
      <td>45.493909</td>
      <td>-73.559507</td>
    </tr>
    <tr>
      <th>2567</th>
      <td>45.471926</td>
      <td>-73.582470</td>
    </tr>
    <tr>
      <th>2512</th>
      <td>45.550698</td>
      <td>-73.609530</td>
    </tr>
    <tr>
      <th>1842</th>
      <td>45.520604</td>
      <td>-73.575984</td>
    </tr>
    <tr>
      <th>2356</th>
      <td>45.472503</td>
      <td>-73.539285</td>
    </tr>
  </tbody>
</table>
</div>


    After preprocessing pipeline



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stn_loc__Cluster 0 similarity</th>
      <th>stn_loc__Cluster 1 similarity</th>
      <th>stn_loc__Cluster 2 similarity</th>
      <th>stn_loc__Cluster 3 similarity</th>
      <th>stn_loc__Cluster 4 similarity</th>
      <th>stn_loc__Cluster 5 similarity</th>
      <th>stn_loc__Cluster 6 similarity</th>
      <th>stn_loc__Cluster 7 similarity</th>
      <th>stn_loc__Cluster 8 similarity</th>
      <th>stn_loc__Cluster 9 similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1424</th>
      <td>0.999153</td>
      <td>0.995976</td>
      <td>0.989462</td>
      <td>0.994232</td>
      <td>0.995808</td>
      <td>0.999893</td>
      <td>0.994748</td>
      <td>0.998497</td>
      <td>0.979020</td>
      <td>0.998324</td>
    </tr>
    <tr>
      <th>2567</th>
      <td>0.996720</td>
      <td>0.993267</td>
      <td>0.989735</td>
      <td>0.997415</td>
      <td>0.995443</td>
      <td>0.999260</td>
      <td>0.989657</td>
      <td>0.997097</td>
      <td>0.983171</td>
      <td>0.999878</td>
    </tr>
    <tr>
      <th>2512</th>
      <td>0.996372</td>
      <td>0.999105</td>
      <td>0.998815</td>
      <td>0.993763</td>
      <td>0.999690</td>
      <td>0.995419</td>
      <td>0.994366</td>
      <td>0.998599</td>
      <td>0.990748</td>
      <td>0.991329</td>
    </tr>
    <tr>
      <th>1842</th>
      <td>0.999599</td>
      <td>0.998866</td>
      <td>0.994365</td>
      <td>0.994678</td>
      <td>0.998611</td>
      <td>0.999351</td>
      <td>0.996609</td>
      <td>0.999942</td>
      <td>0.984222</td>
      <td>0.996389</td>
    </tr>
    <tr>
      <th>2356</th>
      <td>0.997191</td>
      <td>0.991739</td>
      <td>0.982754</td>
      <td>0.991204</td>
      <td>0.991204</td>
      <td>0.998531</td>
      <td>0.991952</td>
      <td>0.995382</td>
      <td>0.971312</td>
      <td>0.997882</td>
    </tr>
  </tbody>
</table>
</div>


The preprocessing pipeline worked as expected. The latitude and longitude columns were converted to similarity measures of each station to each of the 10 cluster centers discovered from the KMeans clustering method.

## Model Pipeline

The preprocessing `ColumnTransformer` was then included as part of a modelling pipeline. Before testing models with no hyperparameter tuning, a function was written to reduce repetitive code.


```python
def base_model_fit(X_tr, y_tr, X_va, y_va, model_object):
    '''
    Fits a model using default parameters and returns remainder and test root mean square errors(RMSE).
    Remainder: train + validation data
    
    Usecase in project: used to build the base model of each classifier
    
    Input
    -----
    X_tr : train independent variables
    y_tr : train dependent variables
    X_va : validation set independent variables
    y_va : validation set dependent variables 
    model_object : sklearn models such as DecisionTreeClassifier()
    
    Output
    ------
    Remainder RMSE
    Test RMSE
    Time taken (seconds)
    '''
    # Start timer
    start = time.perf_counter()
    
    # Initiate the model with no parameters set
    # Note preprocessing defaults to 10 KMeans cluster centers and a RBF Kernel gamma of 1
    model = make_pipeline(preprocessing, model_object)
    
    # Fit the model
    model.fit(X_tr,y_tr)

    # Evaluate the prediction
    y_tr_pred = model.predict(X_tr)
    y_va_pred = model.predict(X_va)

    # Evaluate the RMSE
    model_tr_rmse = mean_squared_error(y_tr,y_tr_pred, squared = False)
    model_va_rmse = mean_squared_error(y_va, y_va_pred, squared = False)

    # Print the train and validation scores
    print(f"Base fitting of a {str(model)} with no hyperparameter adjustment.")
    print(f"RMSE for train data: {np.round(model_tr_rmse*100,3)}%")
    print(f"RMSE for validation data: {np.round(model_va_rmse*100,3)}%")
    
    # End timmer
    end = time.perf_counter()
    time_taken = np.round(end-start,4)
    print(f"Time taken:{time_taken} seconds")
    return model
```

Then, the split data from the stratification processed was renamed for neatness.


```python
# Split the train and validation sets into more conventional X and y labelling
X_tr = train_df[["stn_lat","stn_lon"]]
y_tr = train_df["pct_of_annual_rides"]

# Validation set
X_va = validation_df[["stn_lat","stn_lon"]]
y_va = validation_df["pct_of_annual_rides"]
```

## Model Testing

### Linear Regression

To start, the simple Linear Regression model was used. 


```python
# Fit basic LinearRegression model
model_01 = base_model_fit(X_tr, y_tr, X_va, y_va, LinearRegression())
```

    Base fitting of a Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder=Pipeline(steps=[('simpleimputer',
                                                                  SimpleImputer(strategy='median')),
                                                                 ('standardscaler',
                                                                  StandardScaler())]),
                                       transformers=[('stn_loc',
                                                      ClusterSimilarity(gamma=1,
                                                                        random_state=42),
                                                      ['stn_lat', 'stn_lon'])])),
                    ('linearregression', LinearRegression())]) with no hyperparameter adjustment.
    RMSE for train data: 0.125%
    RMSE for validation data: 0.128%
    Time taken:0.3154 seconds


To give context, most stations range between 0.073% to 0.243% of annual rides, so a validation error of 0.128% does not provide much confidence for the base LinearRegression() model.  


```python
# Context for understanding mean square error
df["pct_of_annual_rides"].describe()*100
```




    count    392500.000000
    mean          0.178344
    std           0.150042
    min           0.000018
    25%           0.073391
    50%           0.138250
    75%           0.243030
    max           1.453954
    Name: pct_of_annual_rides, dtype: float64



### Decision Tree Regressor


```python
# Fit basic Decision Tree Regressor
model_02 = base_model_fit(X_tr,y_tr,X_va,y_va, DecisionTreeRegressor(random_state = 42))
```

    Base fitting of a Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder=Pipeline(steps=[('simpleimputer',
                                                                  SimpleImputer(strategy='median')),
                                                                 ('standardscaler',
                                                                  StandardScaler())]),
                                       transformers=[('stn_loc',
                                                      ClusterSimilarity(gamma=1,
                                                                        random_state=42),
                                                      ['stn_lat', 'stn_lon'])])),
                    ('decisiontreeregressor',
                     DecisionTreeRegressor(random_state=42))]) with no hyperparameter adjustment.
    RMSE for train data: 0.038%
    RMSE for validation data: 0.065%
    Time taken:0.3023 seconds


As expected for a decision tree with unlimited depth, the model has overfitted over the train data, explaining why the RMSE for validation data was almost double of the RMSE for train data. The performance in time was similar to LinearRegression.

### Random Forest Regressor


```python
model_03 = base_model_fit(X_tr, y_tr, X_va, y_va, RandomForestRegressor(random_state = 42))
```

    Base fitting of a Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder=Pipeline(steps=[('simpleimputer',
                                                                  SimpleImputer(strategy='median')),
                                                                 ('standardscaler',
                                                                  StandardScaler())]),
                                       transformers=[('stn_loc',
                                                      ClusterSimilarity(gamma=1,
                                                                        random_state=42),
                                                      ['stn_lat', 'stn_lon'])])),
                    ('randomforestregressor',
                     RandomForestRegressor(random_state=42))]) with no hyperparameter adjustment.
    RMSE for train data: 0.041%
    RMSE for validation data: 0.066%
    Time taken:2.3212 seconds


As expected of an ensemble model, the ensemble of decision trees produces a model that is less overfit on the train data compared to just a single decision tree. The time taken was observed to be roughly 7 times the time taken by a decision tree model. 

### Support Vector Regressor


```python
model_04 = base_model_fit(X_tr, y_tr, X_va, y_va, SVR())
```

    Base fitting of a Pipeline(steps=[('columntransformer',
                     ColumnTransformer(remainder=Pipeline(steps=[('simpleimputer',
                                                                  SimpleImputer(strategy='median')),
                                                                 ('standardscaler',
                                                                  StandardScaler())]),
                                       transformers=[('stn_loc',
                                                      ClusterSimilarity(gamma=1,
                                                                        random_state=42),
                                                      ['stn_lat', 'stn_lon'])])),
                    ('svr', SVR())]) with no hyperparameter adjustment.
    RMSE for train data: 0.569%
    RMSE for validation data: 0.57%
    Time taken:0.2038 seconds


Given the relatively small dataset (<5,000 rows), a support vector machine model was tested to produce a model that was well fitted to both train and validation data at the fastest time.

## Summary of Model Selection

| Model    | Model_type              | RMSE_train | RMSE_validation | Time   |
|----------|-------------------------|-----------|----------------|--------|
| model_01 | LinearRegression()      | 0.125%    | 0.128%         | 0.3154 |
| model_02 | DecisionTreeRegressor() | 0.038%    | 0.065%         | 0.3023 |
| model_03 | RandomForestRegressor() | 0.041%    | 0.066%         | 2.3212 |
| model_04 | SVR()                   | 0.569%    | 0.570%         | 0.2038 |

Based on the summarized results of quickly testing multiple models, the random forest regressor was chosen to proceed with hyperparameter tuning. The decision tree and random forest regressors performed better compared to other models in terms of validation RMSE. The observed gap between train and validation RMSE for both tree regressors were expected due to decision trees overfitting when max depth is not controlled. This can be reduced using cross validation during hyperparameter optimization. With the assumption that station usage planning happens only a few times a year, the higher time consumed by the random forest regressor was deemed acceptable given that an ensemble of decision trees will be more generalizable to the data compared to just a single decision tree.

# Hyperparameter Tuning

The goals of hyperparameter tuning are to find the model hyperparameters that provide an optimum balance between model fit to available data and its ability to generalize to new data. The following hyperparameters will be tuned through the use of grid search and cross validation (GridSearchCV):
1) `n_clusters`: Number of clusters from KMeans clustering of station locations
2) `gamma`: The rate at which similarity falls off when comparing each station to cluster centers from KMeans
3) `max_features`: This hyperparameter limits the number of randomly selected features each node considers for splitting in each decision tree within the random forest.


```python
# Extract the independent and dependent variables into X and y for readability
X_re = remainder_df[["stn_lat", "stn_lon"]]
y_re = remainder_df["pct_of_annual_rides"]
```

## Round 1

To start, a grid of 5 values for each hyperparameter was tested. Given a cross validation of 3 folds, the total number of model fittings performed will be 375 fits. Given the relative small data size (<5,000 rows) and only 3 hyperparameters to optimize, GridSearchCV was used over RandomizedSearchCV for its simplicity.


```python
# Build pipeline which GridSearchCV will run for each fit
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state = 42))
    ]
)

# Generate a grid of parameters for which gridsearchCV will test all combinations of parameters
param_grid = [
    {
        "preprocessing__stn_loc__n_clusters": [5, 10, 15, 20, 25],
        "preprocessing__stn_loc__gamma": [0.01, 0.1, 1, 10, 100],
        "random_forest__max_features": [3, 5, 7, 9, 12]
    }
]

# Build the grid search object
grid_search = GridSearchCV(
    pipeline, 
    param_grid = param_grid, 
    verbose = 5,
    n_jobs = -1,   # Use all processors available
    refit = False, # False = to not refit model with best parameters on the whole dataset
    cv = 3,        # 3 fold cross-validation
    scoring = "neg_root_mean_squared_error"
)

# Perform the grid search on remainder dataset
with print_time():
    grid_search.fit(X_re, y_re)
```

    Fitting 3 folds for each of 125 candidates, totalling 375 fits
    Time taken: 67.0657 seconds.


The grid search and cross validation process for 375 fits took only 65 seconds in this study, which is acceptable considering this process is most likely an annual process. The best parameters can be extracted from the grid search by accessing the `best_params_` attribute.


```python
# Examine the best hyperparameters
grid_search.best_params_
```




    {'preprocessing__stn_loc__gamma': 100,
     'preprocessing__stn_loc__n_clusters': 25,
     'random_forest__max_features': 9}



The best `gamma` and `n_clusters` hyperparameter were found to be on the right end of provided values, warranting a further grid search for values that are beyond the right boundaries. The best `max_features` hyperparameter was 9. This hyperparameter only has effect when the the number of KMeans clusters is greater than 9, meaning that the best `n_clusters` of 25 makes sense. Before moving on to the next round, the RMSE errors were examined in greater detail through the `cv_results` attribute.


```python
# Store the gridsearch results into a DataFrame
cv_res = pd.DataFrame(grid_search.cv_results_)

# Sort by mean test score in descending order since the values are negative RMSE
cv_res.sort_values(
    by = "mean_test_score", ascending = False, inplace = True
)

# Select specific columns for review
cv_res = cv_res.loc[:,[
    "param_preprocessing__stn_loc__gamma",
    "param_preprocessing__stn_loc__n_clusters",
    "param_random_forest__max_features",
    "split0_test_score",
    "split1_test_score",
    "split2_test_score",
    "mean_test_score"
]]

# Specify column names for renaming
score_cols = ["split0","split1","split2","mean_test_rmse"]

# Rename the columns in DataFrame for readability
cv_res.columns = ["gamma", "n_clusters", "max_features"] + score_cols

# For the score columns, multiply by negative to get positive RMSE, then multiply by 100 to convert to percentages
cv_res[score_cols] = -cv_res[score_cols]*100

# Examine the first 10 results
print(f"Hyperparameter Optimization Round 1: The Top 10 Hyperparameter Combinations based on Ascending RMSE:")
cv_res.head(10)
```

    Hyperparameter Optimization Round 1: The Top 10 Hyperparameter Combinations based on Ascending RMSE:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gamma</th>
      <th>n_clusters</th>
      <th>max_features</th>
      <th>split0</th>
      <th>split1</th>
      <th>split2</th>
      <th>mean_test_rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>123</th>
      <td>100</td>
      <td>25</td>
      <td>9</td>
      <td>0.062646</td>
      <td>0.070340</td>
      <td>0.070507</td>
      <td>0.067831</td>
    </tr>
    <tr>
      <th>122</th>
      <td>100</td>
      <td>25</td>
      <td>7</td>
      <td>0.062616</td>
      <td>0.070230</td>
      <td>0.070735</td>
      <td>0.067860</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.1</td>
      <td>25</td>
      <td>5</td>
      <td>0.062804</td>
      <td>0.070334</td>
      <td>0.070656</td>
      <td>0.067931</td>
    </tr>
    <tr>
      <th>96</th>
      <td>10</td>
      <td>25</td>
      <td>5</td>
      <td>0.062669</td>
      <td>0.070139</td>
      <td>0.071097</td>
      <td>0.067969</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1</td>
      <td>25</td>
      <td>12</td>
      <td>0.062594</td>
      <td>0.070449</td>
      <td>0.071089</td>
      <td>0.068044</td>
    </tr>
    <tr>
      <th>97</th>
      <td>10</td>
      <td>25</td>
      <td>7</td>
      <td>0.062821</td>
      <td>0.070062</td>
      <td>0.071258</td>
      <td>0.068047</td>
    </tr>
    <tr>
      <th>110</th>
      <td>100</td>
      <td>15</td>
      <td>3</td>
      <td>0.062723</td>
      <td>0.070378</td>
      <td>0.071073</td>
      <td>0.068058</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.1</td>
      <td>25</td>
      <td>3</td>
      <td>0.062775</td>
      <td>0.070357</td>
      <td>0.071054</td>
      <td>0.068062</td>
    </tr>
    <tr>
      <th>104</th>
      <td>100</td>
      <td>5</td>
      <td>12</td>
      <td>0.064152</td>
      <td>0.069341</td>
      <td>0.070739</td>
      <td>0.068077</td>
    </tr>
    <tr>
      <th>103</th>
      <td>100</td>
      <td>5</td>
      <td>9</td>
      <td>0.064152</td>
      <td>0.069341</td>
      <td>0.070739</td>
      <td>0.068077</td>
    </tr>
  </tbody>
</table>
</div>



7 of the top 10 hyperparameter had 25 as the number of clusters, further warranting a second grid search to explore the hyperparameter values to the right of the boundaries in the first grid search. It was worth noting that the best mean RMSE score was worse than the base model with no hyperparameter tuning (model_03, validation RMSE 0.066%). However, this lower RMSE was prefered as it produces a model that is more generizable to new data, not just to the data it was trained upon.

## Round 2

In round 2, aside from increasing the hyperparameter values to the right of their boundaries in round 1, a smaller step size will also be used in the hopes of finding an optimal combination that is within the middle of the hyperparameter values to be tested in round 2.


```python
# Build pipeline which GridSearchCV will run
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state = 42))
    ]
)

# Generate a grid of parameters for which gridsearchCV will test all combinations of parameters
param_grid = [
    {
        "preprocessing__stn_loc__n_clusters": np.arange(23, 28, 1),
        "preprocessing__stn_loc__gamma": np.arange(500,2600,500),
        "random_forest__max_features": np.arange(7, 12, 1)
    }
]

# Build the grid search object
grid_search = GridSearchCV(
    pipeline, 
    param_grid = param_grid, 
    verbose = 5,
    n_jobs = -1,   # Use all processors available
    refit = False, # False = to not refit model with best parameters on the whole dataset
    cv = 3,        # 3 fold cross-validation
    scoring = "neg_root_mean_squared_error"
)

# Perform the grid search on remainder dataset
with print_time():
    grid_search.fit(X_re, y_re)
```

    Fitting 3 folds for each of 125 candidates, totalling 375 fits
    Time taken: 73.7823 seconds.



```python
# Examine the best params
grid_search.best_params_
```




    {'preprocessing__stn_loc__gamma': 1000,
     'preprocessing__stn_loc__n_clusters': 25,
     'random_forest__max_features': 8}



The best hyperparameters from round 2 were in the middle of each hyperparameter's list of value to be tested, suggesting that an optimal hyperparameter combination had been found. Before finalizing the hyperparameters, the grid search results were again reviewed through a DataFrame.


```python
# Store the gridsearch results into a DataFrame
cv_res = pd.DataFrame(grid_search.cv_results_)

# Sort by mean test score in descending order since the values are negative RMSE
cv_res.sort_values(
    by = "mean_test_score", ascending = False, inplace = True
)

# Select specific columns for review
cv_res = cv_res.loc[:,[
    "param_preprocessing__stn_loc__gamma",
    "param_preprocessing__stn_loc__n_clusters",
    "param_random_forest__max_features",
    "split0_test_score",
    "split1_test_score",
    "split2_test_score",
    "mean_test_score"
]]

# Specify column names for renaming
score_cols = ["split0","split1","split2","mean_test_rmse"]

# Rename the columns in DataFrame for readability
cv_res.columns = ["gamma", "n_clusters", "max_features"] + score_cols

# For the score columns, multiply by negative to get positive RMSE, then multiply by 100 to convert to percentages
cv_res[score_cols] = -cv_res[score_cols]*100

# Examine the first 10 results
print(f"Hyperparameter Optimization Round 1: The Top 10 Hyperparameter Combinations based on Ascending RMSE:")
cv_res.head(10)
```

    Hyperparameter Optimization Round 1: The Top 10 Hyperparameter Combinations based on Ascending RMSE:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gamma</th>
      <th>n_clusters</th>
      <th>max_features</th>
      <th>split0</th>
      <th>split1</th>
      <th>split2</th>
      <th>mean_test_rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>1000</td>
      <td>25</td>
      <td>8</td>
      <td>0.062236</td>
      <td>0.069939</td>
      <td>0.070925</td>
      <td>0.067700</td>
    </tr>
    <tr>
      <th>23</th>
      <td>500</td>
      <td>27</td>
      <td>10</td>
      <td>0.062789</td>
      <td>0.069672</td>
      <td>0.070651</td>
      <td>0.067704</td>
    </tr>
    <tr>
      <th>22</th>
      <td>500</td>
      <td>27</td>
      <td>9</td>
      <td>0.062817</td>
      <td>0.069792</td>
      <td>0.070555</td>
      <td>0.067722</td>
    </tr>
    <tr>
      <th>9</th>
      <td>500</td>
      <td>24</td>
      <td>11</td>
      <td>0.062724</td>
      <td>0.069469</td>
      <td>0.071291</td>
      <td>0.067828</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2000</td>
      <td>25</td>
      <td>8</td>
      <td>0.062548</td>
      <td>0.070026</td>
      <td>0.070922</td>
      <td>0.067832</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1500</td>
      <td>24</td>
      <td>11</td>
      <td>0.062470</td>
      <td>0.069974</td>
      <td>0.071100</td>
      <td>0.067848</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1500</td>
      <td>23</td>
      <td>10</td>
      <td>0.062411</td>
      <td>0.070110</td>
      <td>0.071091</td>
      <td>0.067871</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1500</td>
      <td>27</td>
      <td>8</td>
      <td>0.062370</td>
      <td>0.070898</td>
      <td>0.070350</td>
      <td>0.067873</td>
    </tr>
    <tr>
      <th>11</th>
      <td>500</td>
      <td>25</td>
      <td>8</td>
      <td>0.062870</td>
      <td>0.070030</td>
      <td>0.070732</td>
      <td>0.067877</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1500</td>
      <td>27</td>
      <td>11</td>
      <td>0.062417</td>
      <td>0.070399</td>
      <td>0.070835</td>
      <td>0.067884</td>
    </tr>
  </tbody>
</table>
</div>



| Round     | Best `gamma` | Best `n_clusters` | Best `max_features` | Mean RMSE           |
|-----------|--------------|-------------------|---------------------|---------------------|
| No tuning | 1            | 10                | 10                  | 0.066% (Validation) |
| 1         | 100          | 25                | 9                   | 0.0678%             |
| 2         | 1000         | 25                | 8                   | 0.0677%             |

In round 2 of hyperparameter optimization, the higher gamma suggests a quicker falloff in similarity when comparing each station's locations to the cluster centers. Given that most Bixi stations are within the island of Montreal, the high gamma allows the model to better differentiate stations that are relatively close to each other. However, the high gamma value may also mean that a station which is far away from all 25 cluster centers would receive a low similarity score across all stations. This potential problem will be explored later when visually analyzing the cluster centers.

The `max_features` of 8 suggests that the best model looks at only a max of 8 cluster centers when making a split at each node. Surprisingly, this is similar to human intuition of recognizing city locations based on its proximity to just a few landmarks, rather than referring to all the landmarks in the city before recognizing a location.

Further hyperparameter tuning can be perfomed to find an even more optimized combination of hyperparameters. However, given that 2 rounds of tuning showed negligible improvement in RMSE, the 2 rounds were deemed sufficient to prepare a model that is generizable to new data, and, is not overfitted on the remainder data.

# Evaluation of Final Model

The best hyperparameters were used to build a final model in a simlar fashion to hyperparameter tuning, except only 1 combination of hyperparameters were provided and `refit` was set to `True`. Further analysis was performed on the final model before moving on to test it with the reserved test data for reporting.


```python
# Insert best parameters into a parameter grid
final_param_grid = param_grid = [
    {
        "preprocessing__stn_loc__n_clusters": [25],
        "preprocessing__stn_loc__gamma": [1000],
        "random_forest__max_features": [8]
    }
]

# Use grid search to fit the model with the final parameters
grid_search = GridSearchCV(
    pipeline, 
    param_grid = final_param_grid, 
    verbose = 5,
    n_jobs = -1,
    refit = True,
    cv = 3, 
    scoring = "neg_root_mean_squared_error"
)

# Perform the fit
grid_search.fit(X_re, y_re)

# Store the best estimator as the final model
final_model = grid_search.best_estimator_
```

    Fitting 3 folds for each of 1 candidates, totalling 3 fits


## Cluster Centers

The 25 cluster centers were accessed through the `transformers_` attribute of the final model.


```python
# cc_df: cluster center DataFrame
cc_df = pd.DataFrame({
    "cluster_name" : final_model["preprocessing"].get_feature_names_out(),
    "cluster_lat"  : final_model["preprocessing"].transformers_[0][1].kmeans_.cluster_centers_[:,0],
    "cluster_lon"  : final_model["preprocessing"].transformers_[0][1].kmeans_.cluster_centers_[:,1]
})

# # For simplicity, the cluster names are reduced to just the number within each string
# cc_df["cluster_name"] = cc_df["cluster_name"].str.extract('(\d+)').astype("int")+1

display(cc_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_name</th>
      <th>cluster_lat</th>
      <th>cluster_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>stn_loc__Cluster 0 similarity</td>
      <td>45.506621</td>
      <td>-73.573372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>stn_loc__Cluster 1 similarity</td>
      <td>45.552163</td>
      <td>-73.575543</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stn_loc__Cluster 2 similarity</td>
      <td>45.559606</td>
      <td>-73.652577</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stn_loc__Cluster 3 similarity</td>
      <td>45.445015</td>
      <td>-73.596257</td>
    </tr>
    <tr>
      <th>4</th>
      <td>stn_loc__Cluster 4 similarity</td>
      <td>45.540311</td>
      <td>-73.623883</td>
    </tr>
  </tbody>
</table>
</div>


Furthermore, the relative importance of each cluster center can be extracted from the final model using the `feature_importances` attribute. This measure of importance is calculated by examining how much the tree nodes which used that feature reduce impurity on average across all 100 decision trees in the final model. This measure of importance is scaled such that the sum across all features adds up to 1.

For ease of reference each cluster center, the cluster centers were renamed based on their importance.


```python
# Get importance of each cluster center from final model
cc_df["importance"] = final_model["random_forest"].feature_importances_

# order by feature importance
cc_df.sort_values(
    by = ["importance"],
    ascending = [False],
    inplace =  True
)

# Number of clusters (ascending)
cc_df["cluster_name"] = cc_df["importance"].rank(ascending = False).astype("int")

# Cumulative importance
cc_df["cum_importance"] = cc_df["importance"].cumsum()

# Reset index for neatness
cc_df.reset_index(drop = True, inplace = True)

# Examine the top 10 cluster centers
display(cc_df.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_name</th>
      <th>cluster_lat</th>
      <th>cluster_lon</th>
      <th>importance</th>
      <th>cum_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>45.506621</td>
      <td>-73.573372</td>
      <td>0.130542</td>
      <td>0.130542</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>45.510082</td>
      <td>-73.558289</td>
      <td>0.096973</td>
      <td>0.227515</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>45.529592</td>
      <td>-73.576008</td>
      <td>0.069114</td>
      <td>0.296629</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>45.489861</td>
      <td>-73.568448</td>
      <td>0.065767</td>
      <td>0.362396</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>45.499491</td>
      <td>-73.630374</td>
      <td>0.064089</td>
      <td>0.426484</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>45.445015</td>
      <td>-73.596257</td>
      <td>0.057919</td>
      <td>0.484403</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>45.516432</td>
      <td>-73.691777</td>
      <td>0.054201</td>
      <td>0.538604</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>45.457505</td>
      <td>-73.574330</td>
      <td>0.046285</td>
      <td>0.584889</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>45.513747</td>
      <td>-73.529658</td>
      <td>0.040833</td>
      <td>0.625722</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>45.523794</td>
      <td>-73.611104</td>
      <td>0.036474</td>
      <td>0.662196</td>
    </tr>
  </tbody>
</table>
</div>


The cluster centers were plotted along with the cumulative importance.


```python
# Initiate figure object
plt.figure()

# Create line plot
cc_df.plot(
    x = "cluster_name",
    y = "cum_importance",
    kind = "line",
    ax = plt.gca(),
    legend = False
)

# Define location for annotation
ann_x = 15
ann_y = cc_df.loc[cc_df["cluster_name"] == 15,"cum_importance"].values[0]

# Plot a marker to highlight 80% cumulated importance
plt.plot(
    ann_x,
    ann_y, 
    marker = ".",
    markersize = 10,
    color = "black"
)

# Add annotations
plt.annotate(
    text = "80% of cumulated importance \nexplained by 15 cluster centers",
    xy = (ann_x,ann_y),
    xytext = (ann_x*1.05, ann_y*0.70),
    arrowprops = dict(facecolor = "black", shrink = 0.05)
)

# Tidy up plot
plt.ylabel(f"Cumulative Sum of Feature Importance")
plt.xlabel(f"Number of Clusters, sorted by desc Feature Importance")
plt.title(f"Figure 10. Cumulative Importance vs Number of Clusters", y = -0.2)
sns.despine()
plt.show()
```


    
![png](../assets/images/output_149_0.png){:class="img-responsive"}
    


80% of cumulative importance was explained by the top 15 cluster centers. Further experiments maybe conducted to determine how much performance drops if only 15 cluster centers were used instead.

Then, the cluster centers were plotted against the station locations for a geo-visual understanding of the cluster centers. The size of markers represent the importance of the cluster center in the final models perspective.


```python
# Initiate figure object
plt.figure()

# Plot the latitude and longitude of stations in the remainder data
remainder_df.plot(
    kind  = "scatter",
    y     = "stn_lat",
    x     = "stn_lon",
    c     = "cornflowerblue",
    alpha = 0.2, # To deal with overlapping stations
    ax    = plt.gca()
)

# Plot the cluster centers
cc_df.plot(
    kind = "scatter",
    y = "cluster_lat",
    x = "cluster_lon",
    color = "black",
    linestyle = "",
    marker = "X",
    label = "Cluster centers",
    s = cc_df["importance"]*1000,
    ax = plt.gca())

# Annotate each cluster center
annotations = [plt.annotate(label, (cc_df["cluster_lon"][i], cc_df["cluster_lat"][i])) for i, label in enumerate(cc_df["cluster_name"])]

# Tidy annotation position
adjust_text(annotations)

# Tidy plot
sns.despine()
plt.xlabel(f"Station Longitude")
plt.ylabel(f"Station Latitude")
plt.title("Figure 11. Cluster Centers Generated by Final Model", y = -0.2)
plt.show()
```


    
![png](../assets/images/output_151_0.png){:class="img-responsive"}
    


The top 10 cluster centers in terms of importance were observed to be roughly evenly spread out across the city rather than being clumped together in the city center of Montreal. Many cluster centers that are not in the top 10 were also found around to be in the city center of Montreal. This suggests that the final model is dependent on these less important feature to finely differentiate stations in between the top 10 stations.

A deeper interactive visualization using plotly was also used to explore the cluster centers identified by the model.


```python
# Read in mapbox secret from .mapbox_token file
px.set_mapbox_access_token(open("../.mapbox_token").read())

# Create plotly express map object for the station lat and lon
fig = px.scatter_mapbox(
    remainder_df,
    lat        = "stn_lat",
    lon        = "stn_lon",
    text       = "stn_name",
    title      = "Figure 12. Cluster Centers from Final Model Overlayed over Montreal",
    hover_name = "stn_name",
    hover_data = {
        "rides" : ":,",
        "stn_lat" : ":.3f",
        "stn_lon" : ":.3f",
    },
    zoom       = 10.7,
    opacity    = 0.1
)

# Add plotly trace for cluster centers from final model
fig.add_scattermapbox(
    lat      = cc_df["cluster_lat"],
    lon      = cc_df["cluster_lon"],
    text     = [str(name) for name in cc_df["cluster_name"]],
    mode     = "text",
    textfont = {"size":25}
)

# Tidy up layout of figure
fig.update_layout(
    # Set title
    title = dict(
        x = 0.5,
        y = 0.05
    ),
    autosize   = False,
    width      = 1000,
    height     = 1000,
    showlegend = False
)

fig.show()
```


<div>                            <div id="2e6ca530-593a-48d8-8ba9-3b1a6bbf2dd7" class="plotly-graph-div" style="height:1000px; width:1000px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2e6ca530-593a-48d8-8ba9-3b1a6bbf2dd7")) {                    Plotly.newPlot(                        "2e6ca530-593a-48d8-8ba9-3b1a6bbf2dd7",                        [{"customdata":[[17465.0,45.493909,-73.559507],[5801.0,45.471926,-73.58247],[1860.0,45.550698,-73.60953],[10245.0,45.520604,-73.575984],[1219.0,45.472503,-73.539285],[3233.0,45.546408,-73.638428],[18777.0,45.525628,-73.581775],[8935.0,45.470303,-73.589848],[8271.0,45.48204,-73.574863],[4819.0,45.544846,-73.610536],[8417.0,45.533349,-73.601136],[10651.0,45.529693,-73.567401],[4015.0,45.544513,-73.571564],[15925.0,45.522365,-73.584146],[5766.0,45.530027,-73.563176],[4763.0,45.531674,-73.565413],[10485.0,45.49426,-73.563818],[8903.0,45.533348,-73.605834],[11890.0,45.509864,-73.576105],[7118.0,45.497011,-73.618575],[11986.0,45.499326,-73.571768],[2997.0,45.550737,-73.615222],[9962.0,45.546907,-73.551557],[10205.0,45.521094,-73.562251],[11037.0,45.533496,-73.566931],[7396.0,45.518263,-73.595775],[1681.0,45.553262,-73.638615],[10689.0,45.514115,-73.552645],[19277.0,45.491754,-73.583972],[12826.0,45.500779,-73.558826],[7969.0,45.537877,-73.618323],[7614.0,45.532137,-73.558713],[3202.0,45.466851,-73.631628],[10811.0,45.478228,-73.569651],[2418.0,45.539632,-73.634448],[557.0,45.573798,-73.550345],[1148.0,45.58077,-73.544766],[16135.0,45.525643,-73.581748],[5183.0,45.509188,-73.55458],[3944.0,45.539984,-73.638135],[8851.0,45.539,-73.614251],[3023.0,45.514905,-73.619682],[3708.0,45.549755,-73.605566],[3610.0,45.4771,-73.6212],[2355.0,45.568225,-73.579436],[7098.0,45.538938,-73.624293],[707.0,45.519492,-73.685104],[19934.0,45.526543,-73.598233],[10388.0,45.537259,-73.57917],[1940.0,45.415349,-73.626514],[1866.0,45.434434,-73.586694],[12004.0,45.5219,-73.55534],[5287.0,45.470696,-73.565422],[542.0,45.50623,-73.55976],[5193.0,45.522586,-73.612658],[515.0,45.609626,-73.522739],[2810.0,45.46714,-73.54259],[938.0,45.58077,-73.544766],[9513.0,45.523194,-73.555843],[4727.0,45.546408,-73.638428],[10895.0,45.49975,-73.55566],[5419.0,45.539259,-73.577338],[1885.0,45.529686,-73.601728],[4834.0,45.530674,-73.56625],[504.0,45.536378,-73.512642],[932.0,45.558457,-73.65884],[19717.0,45.50206,-73.56295],[6619.0,45.537205,-73.579254],[4810.0,45.547362,-73.579221],[726.0,45.506223,-73.711481],[19503.0,45.5171,-73.57921],[6028.0,45.542055,-73.597793],[31543.0,45.527432,-73.579917],[6869.0,45.504542,-73.559164],[13378.0,45.537259,-73.57917],[17510.0,45.536785,-73.614888],[3209.0,45.480508,-73.58661],[2592.0,45.599266,-73.512261],[499.0,45.52841,-73.517166],[2681.0,45.456823,-73.546804],[18999.0,45.52729,-73.56463],[5688.0,45.550692,-73.656367],[5501.0,45.539224,-73.573837],[6269.0,45.531475,-73.592421],[6636.0,45.508295,-73.580643],[4290.0,45.483006,-73.562187],[11688.0,45.519237,-73.577215],[2312.0,45.547218,-73.631103],[6132.0,45.496302,-73.629822],[21125.0,45.506448,-73.576349],[5098.0,45.542055,-73.597793],[6841.0,45.497974,-73.556141],[8023.0,45.537159,-73.597476],[4264.0,45.56244,-73.595333],[1416.0,45.459753,-73.596421],[1546.0,45.573733,-73.539446],[8757.0,45.516897,-73.563949],[6243.0,45.529155,-73.559411],[5485.0,45.489727,-73.564063],[3880.0,45.477353,-73.615372],[8381.0,45.49606,-73.57348],[7144.0,45.532674,-73.571776],[17078.0,45.514108,-73.57523],[5380.0,45.519373,-73.551958],[6500.0,45.552364,-73.602868],[4073.0,45.456884,-73.572607],[3513.0,45.531938,-73.617842],[2899.0,45.546408,-73.638428],[41503.0,45.49659,-73.57851],[3192.0,45.49351,-73.561728],[21434.0,45.503475,-73.572103],[9380.0,45.495894,-73.560097],[6777.0,45.530351,-73.624392],[16206.0,45.52689,-73.57264],[7405.0,45.54167,-73.607605],[7346.0,45.539864,-73.620313],[5181.0,45.4829,-73.591],[10177.0,45.51889,-73.56353],[14241.0,45.5341,-73.58852],[9863.0,45.527426,-73.579918],[13887.0,45.50345,-73.568404],[16346.0,45.518253,-73.588338],[9109.0,45.517643,-73.588928],[5309.0,45.540154,-73.623531],[12412.0,45.559978,-73.547791],[3968.0,45.456049,-73.571877],[3584.0,45.501864,-73.543961],[5827.0,45.480914,-73.56846],[7599.0,45.533735,-73.589307],[3651.0,45.456358,-73.64013],[10881.0,45.524683,-73.578897],[1883.0,45.563921,-73.548912],[1252.0,45.561396,-73.601544],[11595.0,45.505403,-73.621166],[4509.0,45.491384,-73.586013],[15602.0,45.519237,-73.577215],[10280.0,45.526027,-73.577859],[2608.0,45.4688,-73.6198],[11395.0,45.521556,-73.562264],[3104.0,45.553408,-73.569531],[6892.0,45.493569,-73.568576],[9986.0,45.500674,-73.561082],[6328.0,45.529504,-73.604865],[1460.0,45.577136,-73.546289],[8491.0,45.532977,-73.581222],[31767.0,45.506448,-73.576349],[10269.0,45.471926,-73.58247],[296.0,45.461701,-73.584072],[1946.0,45.435664,-73.605652],[510.0,45.487531,-73.56507],[4463.0,45.510042,-73.624835],[23409.0,45.498673,-73.552563],[7777.0,45.536178,-73.57641],[3123.0,45.475142,-73.623505],[18635.0,45.526252,-73.589758],[12825.0,45.50623,-73.55976],[8037.0,45.500877,-73.559006],[7232.0,45.52123,-73.553861],[4288.0,45.544867,-73.614841],[5950.0,45.550684,-73.615293],[11645.0,45.512587,-73.573739],[8528.0,45.514115,-73.552645],[3463.0,45.53923,-73.541082],[15863.0,45.526991,-73.558212],[12317.0,45.498112,-73.577615],[3790.0,45.521069,-73.563757],[1866.0,45.570081,-73.573047],[4978.0,45.540951,-73.626375],[11321.0,45.521751,-73.585268],[8045.0,45.532674,-73.571776],[11728.0,45.504278,-73.617977],[14152.0,45.497605,-73.55535],[1415.0,45.570081,-73.573047],[424.0,45.539824,-73.508752],[15120.0,45.493718,-73.579186],[2735.0,45.471179,-73.627002],[6965.0,45.53217,-73.55859],[4308.0,45.534377,-73.612278],[21748.0,45.527201,-73.564729],[11001.0,45.53848,-73.60556],[3120.0,45.555814,-73.577853],[9510.0,45.529802,-73.57029],[2932.0,45.553053,-73.532427],[5748.0,45.533409,-73.570657],[3744.0,45.505312,-73.560891],[5066.0,45.54257,-73.592144],[5817.0,45.509229,-73.55447],[1178.0,45.526856,-73.626672],[12009.0,45.49642,-73.57616],[8747.0,45.50015,-73.56928],[11590.0,45.513917,-73.560765],[7762.0,45.459488,-73.572092],[1304.0,45.460729,-73.634073],[10238.0,45.532008,-73.580444],[2767.0,45.543376,-73.632267],[8645.0,45.549598,-73.541874],[2076.0,45.550668,-73.549502],[4017.0,45.491322,-73.58781],[11492.0,45.546907,-73.551557],[2702.0,45.569789,-73.548081],[268.0,45.431648,-73.671125],[17047.0,45.519216,-73.57718],[7617.0,45.480914,-73.56846],[8939.0,45.499965,-73.556154],[11308.0,45.546978,-73.575515],[12723.0,45.52689,-73.57264],[4439.0,45.574147,-73.574621],[10782.0,45.506371,-73.564201],[10778.0,45.514821,-73.578296],[6249.0,45.53318,-73.61544],[26017.0,45.518593,-73.581566],[6183.0,45.544136,-73.545609],[15657.0,45.520604,-73.575984],[2164.0,45.529686,-73.601728],[4183.0,45.539833,-73.575559],[9885.0,45.523492,-73.56073],[13963.0,45.474419,-73.604059],[7429.0,45.479483,-73.61979],[13234.0,45.521293,-73.589365],[2937.0,45.486219,-73.573747],[7699.0,45.530505,-73.614708],[1126.0,45.611231,-73.531242],[12214.0,45.53293,-73.584151],[10508.0,45.480208,-73.577599],[2559.0,45.463257,-73.575621],[1624.0,45.559884,-73.633568],[3802.0,45.47995,-73.580536],[9501.0,45.519153,-73.616469],[4243.0,45.542579,-73.636557],[12560.0,45.534136,-73.595478],[18439.0,45.493029,-73.564818],[8124.0,45.545776,-73.562175],[8125.0,45.541484,-73.612441],[5545.0,45.481184,-73.579349],[7081.0,45.507144,-73.555119],[237.0,45.539316,-73.620661],[11287.0,45.520188,-73.590559],[17311.0,45.509361,-73.573864],[7871.0,45.491687,-73.576767],[1960.0,45.524634,-73.578755],[14823.0,45.532281,-73.610991],[11337.0,45.508144,-73.574772],[2444.0,45.499757,-73.629298],[9212.0,45.538984,-73.582445],[2350.0,45.502602,-73.527503],[9382.0,45.501715,-73.574131],[613.0,45.539824,-73.508752],[3584.0,45.485048,-73.571963],[1721.0,45.564944,-73.577639],[10130.0,45.514531,-73.568471],[14122.0,45.523401,-73.576456],[13295.0,45.53543,-73.5822],[3275.0,45.514379,-73.610871],[6319.0,45.549686,-73.591155],[31626.0,45.53229,-73.57544],[6476.0,45.519153,-73.616469],[2595.0,45.550769,-73.620911],[4792.0,45.50931,-73.554431],[4122.0,45.45073,-73.572575],[1509.0,45.592596,-73.566175],[3489.0,45.491722,-73.57689],[4969.0,45.485048,-73.571963],[5624.0,45.524266,-73.611274],[5506.0,45.549755,-73.605566],[7277.0,45.535614,-73.566014],[12132.0,45.542898,-73.561846],[18087.0,45.523027,-73.60184],[4601.0,45.474565,-73.624036],[18762.0,45.50501,-73.57069],[9268.0,45.5219,-73.55534],[543.0,45.5332,-73.5156],[7311.0,45.543452,-73.60101],[2028.0,45.521768,-73.557103],[92.0,45.502989,-73.689113],[8692.0,45.545776,-73.562175],[7651.0,45.517643,-73.588928],[16127.0,45.5359,-73.622443],[5536.0,45.514773,-73.578312],[8780.0,45.53611,-73.576287],[23339.0,45.517354,-73.582129],[4365.0,45.483928,-73.577312],[17640.0,45.524963,-73.575559],[5831.0,45.494514,-73.583368],[11802.0,45.497411,-73.578271],[4039.0,45.552923,-73.565752],[6925.0,45.53576,-73.57867],[11728.0,45.53696,-73.61199],[2973.0,45.459986,-73.543956],[14081.0,45.509759,-73.576107],[6725.0,45.4816,-73.6],[15535.0,45.499677,-73.578849],[16750.0,45.529666,-73.567336],[689.0,45.533682,-73.515261],[4034.0,45.446934,-73.572199],[387.0,45.42506,-73.649599],[6120.0,45.551119,-73.540764],[9978.0,45.561783,-73.546611],[5591.0,45.513542,-73.562871],[4298.0,45.52974,-73.59527],[7442.0,45.496851,-73.62332],[1615.0,45.520075,-73.629776],[15648.0,45.521342,-73.589419],[5024.0,45.463397,-73.596436],[927.0,45.449901,-73.600251],[12209.0,45.520688,-73.608428],[8535.0,45.539385,-73.541],[9411.0,45.553898,-73.571387],[2318.0,45.506373,-73.524577],[11647.0,45.496583,-73.576502],[7034.0,45.498767,-73.61887],[614.0,45.500339,-73.682621],[6220.0,45.54458,-73.588069],[5008.0,45.541955,-73.629942],[24091.0,45.517,-73.589],[7713.0,45.490452,-73.560172],[5769.0,45.537159,-73.597476],[4991.0,45.501367,-73.565837],[1544.0,45.460023,-73.644442],[5326.0,45.544867,-73.614841],[6072.0,45.489545,-73.564139],[6111.0,45.54257,-73.592144],[10618.0,45.548186,-73.59201],[2139.0,45.415769,-73.637924],[7460.0,45.507078,-73.561395],[2935.0,45.459986,-73.543956],[2551.0,45.518215,-73.603251],[38063.0,45.527616,-73.589115],[17928.0,45.532592,-73.585195],[1327.0,45.570506,-73.651781],[29.0,45.609626,-73.522739],[14042.0,45.510016,-73.570654],[2255.0,45.544859,-73.666814],[6813.0,45.530871,-73.598346],[11628.0,45.521751,-73.585268],[328.0,45.539824,-73.508752],[3999.0,45.550769,-73.620911],[16316.0,45.527795,-73.571939],[2943.0,45.516079,-73.607798],[5897.0,45.544513,-73.571564],[4110.0,45.466569,-73.623295],[10632.0,45.514784,-73.565338],[5054.0,45.49606,-73.57348],[4635.0,45.561594,-73.585643],[12098.0,45.51086,-73.54983],[5480.0,45.54918,-73.593858],[2560.0,45.552649,-73.581185],[4927.0,45.547397,-73.607664],[8908.0,45.514784,-73.565338],[7354.0,45.493034,-73.583836],[5583.0,45.519373,-73.551958],[4313.0,45.542983,-73.606381],[5607.0,45.544819,-73.62151],[480.0,45.588303,-73.577837],[2752.0,45.4688,-73.6198],[22884.0,45.51086,-73.54983],[6417.0,45.512734,-73.561141],[4983.0,45.553053,-73.532428],[15240.0,45.501715,-73.574131],[65.0,45.533661,-73.552197],[1845.0,45.486279,-73.573599],[5157.0,45.53133,-73.59155],[3930.0,45.540627,-73.630089],[2168.0,45.569789,-73.548081],[14393.0,45.524375,-73.59181],[5487.0,45.457894,-73.567528],[5878.0,45.50449,-73.5592],[17281.0,45.4918,-73.584006],[4564.0,45.534563,-73.554786],[8876.0,45.5215,-73.5851],[923.0,45.559705,-73.551554],[9012.0,45.506251,-73.571391],[19079.0,45.519092,-73.55785],[9231.0,45.498919,-73.557254],[3416.0,45.549714,-73.563848],[1007.0,45.568225,-73.579436],[7564.0,45.505721,-73.629459],[4168.0,45.547997,-73.58523],[8705.0,45.547179,-73.551739],[7490.0,45.511132,-73.567907],[2026.0,45.541369,-73.583974],[7593.0,45.523584,-73.560941],[8010.0,45.519373,-73.551958],[7265.0,45.525021,-73.610737],[18638.0,45.521495,-73.596758],[12734.0,45.495761,-73.576826],[10193.0,45.504347,-73.568949],[7519.0,45.519581,-73.560116],[22312.0,45.47393,-73.604735],[1634.0,45.561594,-73.585643],[21156.0,45.516427,-73.558112],[4704.0,45.480264,-73.580994],[7011.0,45.508356,-73.580498],[631.0,45.536344,-73.512778],[999.0,45.434434,-73.586694],[5335.0,45.556262,-73.537441],[9848.0,45.51889,-73.56353],[5349.0,45.538308,-73.654884],[5385.0,45.493034,-73.583836],[11323.0,45.521161,-73.585616],[17655.0,45.508144,-73.574772],[2636.0,45.539984,-73.638135],[7893.0,45.506251,-73.571391],[13869.0,45.492504,-73.581088],[12643.0,45.491754,-73.583972],[1023.0,45.537226,-73.495067],[4030.0,45.464877,-73.626597],[3298.0,45.53877,-73.588679],[439.0,45.433557,-73.684043],[5924.0,45.531178,-73.632136],[24572.0,45.51059,-73.57547],[12505.0,45.520574,-73.556144],[9655.0,45.501967,-73.555153],[5931.0,45.526246,-73.613747],[12375.0,45.492897,-73.580294],[3924.0,45.429471,-73.593099],[10154.0,45.519157,-73.55786],[29917.0,45.53229,-73.57544],[13154.0,45.503475,-73.572103],[20687.0,45.525628,-73.581775],[12332.0,45.500779,-73.558826],[8341.0,45.532674,-73.571776],[3378.0,45.531938,-73.617842],[8023.0,45.527727,-73.586284],[16709.0,45.529693,-73.567401],[13656.0,45.535341,-73.603659],[2936.0,45.539294,-73.60042],[7359.0,45.532674,-73.571776],[12591.0,45.537649,-73.579785],[5583.0,45.527363,-73.607723],[5371.0,45.510042,-73.624835],[482.0,45.506176,-73.711186],[6090.0,45.540881,-73.584157],[6685.0,45.539487,-73.623556],[2725.0,45.45002,-73.600306],[4772.0,45.56833,-73.566663],[10283.0,45.538021,-73.583524],[10226.0,45.534136,-73.595478],[1947.0,45.573471,-73.574975],[5785.0,45.505377,-73.56762],[18390.0,45.49871,-73.56671],[2453.0,45.456365,-73.57614],[2038.0,45.582847,-73.542476],[1199.0,45.556065,-73.653502],[3786.0,45.47968,-73.56385],[9685.0,45.520295,-73.614824],[9988.0,45.459488,-73.572092],[3288.0,45.4703,-73.6162],[3658.0,45.545459,-73.556669],[15378.0,45.53707,-73.61529],[5484.0,45.539487,-73.623556],[5997.0,45.534453,-73.559652],[32185.0,45.51735,-73.56906],[11132.0,45.535189,-73.617459],[9262.0,45.501449,-73.633051],[328.0,45.537226,-73.495067],[656.0,45.503666,-73.569576],[4891.0,45.518128,-73.561851],[11275.0,45.51494,-73.57821],[13317.0,45.523507,-73.576413],[4813.0,45.531178,-73.632136],[3081.0,45.556262,-73.537441],[8946.0,45.517459,-73.597927],[18136.0,45.49947,-73.57591],[10063.0,45.530046,-73.58677],[14898.0,45.547405,-73.544674],[8001.0,45.528724,-73.583859],[1189.0,45.490472,-73.638532],[6337.0,45.481491,-73.564754],[5667.0,45.475743,-73.564601],[6025.0,45.532331,-73.620187],[5670.0,45.482826,-73.590843],[5480.0,45.528449,-73.551085],[3366.0,45.549121,-73.600874],[13051.0,45.51494,-73.57821],[3308.0,45.549121,-73.600874],[14130.0,45.4952,-73.56328],[3197.0,45.506373,-73.524577],[729.0,45.566869,-73.641017],[3707.0,45.514379,-73.610871],[1798.0,45.456049,-73.571877],[22.0,45.505973,-73.629593],[9134.0,45.533348,-73.605834],[10478.0,45.503063,-73.573986],[2162.0,45.453246,-73.576815],[15149.0,45.536785,-73.614888],[4579.0,45.503552,-73.568755],[1339.0,45.461685,-73.584039],[15027.0,45.512801,-73.531116],[22802.0,45.504242,-73.55347],[12043.0,45.49871,-73.56671],[6654.0,45.550684,-73.615293],[7436.0,45.499326,-73.571768],[1704.0,45.523413,-73.623526],[4619.0,45.484859,-73.571585],[8130.0,45.508356,-73.580498],[5158.0,45.500043,-73.560412],[24890.0,45.504242,-73.553469],[4148.0,45.4703,-73.6162],[3131.0,45.523319,-73.520127],[9924.0,45.537259,-73.57917],[1410.0,45.434434,-73.586694],[11981.0,45.4952,-73.56328],[10972.0,45.536505,-73.568039],[4930.0,45.477923,-73.559038],[4862.0,45.552016,-73.581855],[4134.0,45.551584,-73.561916],[2965.0,45.547408,-73.607669],[20291.0,45.52689,-73.57264],[2651.0,45.457591,-73.567559],[12518.0,45.507402,-73.578444],[17066.0,45.530221,-73.576997],[5436.0,45.543452,-73.60101],[7749.0,45.531164,-73.583695],[3328.0,45.5195,-73.560269],[4168.0,45.544513,-73.571564],[9330.0,45.53696,-73.61199],[170.0,45.61443,-73.612233],[20146.0,45.51059,-73.57547],[14976.0,45.535985,-73.622544],[301.0,45.56169,-73.610512],[5063.0,45.536299,-73.607773],[12013.0,45.507078,-73.561395],[7244.0,45.541435,-73.554314],[8803.0,45.524634,-73.578755],[570.0,45.590892,-73.640617],[10615.0,45.532281,-73.610991],[11638.0,45.5452,-73.576451],[17235.0,45.49871,-73.56671],[9742.0,45.529219,-73.574777],[10518.0,45.55896,-73.548687],[5609.0,45.497165,-73.55933],[12127.0,45.49606,-73.57348],[5532.0,45.493596,-73.561964],[6854.0,45.531818,-73.565317],[7501.0,45.53703,-73.58584],[11011.0,45.495789,-73.582596],[2844.0,45.493569,-73.568576],[96.0,45.446061,-73.6508],[2480.0,45.552123,-73.630103],[4350.0,45.4891,-73.57656],[1184.0,45.532415,-73.564708],[9437.0,45.50723,-73.615085],[28833.0,45.515616,-73.575808],[44009.0,45.527616,-73.589115],[688.0,45.500105,-73.560199],[2936.0,45.549126,-73.600912],[7485.0,45.553457,-73.587842],[18951.0,45.52689,-73.57264],[5907.0,45.543771,-73.628509],[6841.0,45.543285,-73.616286],[5228.0,45.553215,-73.58752],[5560.0,45.526566,-73.585428],[4881.0,45.564944,-73.577639],[19700.0,45.525172,-73.599375],[8501.0,45.544376,-73.595645],[4848.0,45.5427,-73.59212],[4710.0,45.534415,-73.612316],[6867.0,45.544377,-73.581018],[7160.0,45.5262,-73.549266],[5561.0,45.514946,-73.556233],[6193.0,45.490687,-73.562538],[3447.0,45.561594,-73.585643],[2411.0,45.55197,-73.593775],[7618.0,45.55816,-73.5484],[15644.0,45.529802,-73.57029],[3319.0,45.463254,-73.575879],[5331.0,45.463001,-73.571569],[191.0,45.63065,-73.494995],[11843.0,45.502813,-73.576024],[10635.0,45.512871,-73.558466],[19067.0,45.49683,-73.57913],[17855.0,45.51086,-73.54983],[15298.0,45.522821,-73.601296],[10846.0,45.52211,-73.5962],[5816.0,45.457509,-73.639485],[11282.0,45.510086,-73.611429],[11833.0,45.561783,-73.546611],[61.0,45.43746,-73.693532],[8197.0,45.529135,-73.559234],[6678.0,45.49642,-73.57616],[4282.0,45.529033,-73.54641],[10460.0,45.484018,-73.563437],[4588.0,45.551584,-73.561916],[6747.0,45.547179,-73.551739],[6710.0,45.532331,-73.620187],[5804.0,45.526206,-73.54925],[10511.0,45.499745,-73.579034],[882.0,45.526856,-73.626672],[6407.0,45.47968,-73.56385],[3053.0,45.480508,-73.58661],[14324.0,45.516818,-73.554188],[6580.0,45.514335,-73.556463],[826.0,45.456365,-73.57614],[10353.0,45.474224,-73.603721],[11705.0,45.513406,-73.562594],[8549.0,45.544599,-73.588223],[3422.0,45.483928,-73.577312],[3361.0,45.552123,-73.630103],[2818.0,45.437914,-73.58274],[1865.0,45.577136,-73.546289],[11250.0,45.505403,-73.621166],[4158.0,45.542972,-73.617939],[17948.0,45.504095,-73.568696],[54.0,45.439779,-73.681371],[9420.0,45.497595,-73.555237],[4494.0,45.46982,-73.616961],[7784.0,45.544599,-73.588223],[12258.0,45.527041,-73.593471],[603.0,45.484896,-73.560119],[10927.0,45.522878,-73.577495],[22140.0,45.525164,-73.599306],[10482.0,45.488877,-73.583537],[18573.0,45.535341,-73.603659],[292.0,45.649546,-73.491916],[13508.0,45.527748,-73.597439],[2922.0,45.471162,-73.627206],[10917.0,45.54111,-73.54755],[5762.0,45.533924,-73.562425],[2463.0,45.48184,-73.595379],[7445.0,45.4891,-73.57656],[13697.0,45.506587,-73.573299],[16787.0,45.522278,-73.577591],[3877.0,45.551584,-73.561916],[9117.0,45.510086,-73.611429],[11588.0,45.517643,-73.588928],[8670.0,45.5452,-73.576451],[7279.0,45.51339,-73.56255],[7746.0,45.541484,-73.612441],[22994.0,45.517354,-73.582129],[21242.0,45.498673,-73.552563],[11994.0,45.547161,-73.543773],[2475.0,45.456355,-73.597557],[2824.0,45.552649,-73.581185],[3972.0,45.461685,-73.584039],[12973.0,45.51829,-73.59235],[14111.0,45.504095,-73.568696],[8248.0,45.538958,-73.582432],[1114.0,45.438544,-73.620216],[3663.0,45.528449,-73.551085],[3371.0,45.556905,-73.58925],[162.0,45.483629,-73.579703],[9163.0,45.53569,-73.565923],[8744.0,45.512813,-73.577017],[511.0,45.52841,-73.517166],[17930.0,45.516664,-73.57722],[10457.0,45.533924,-73.562425],[11671.0,45.525669,-73.593016],[4908.0,45.527363,-73.607723],[26545.0,45.480208,-73.577599],[20923.0,45.537041,-73.602026],[4880.0,45.539857,-73.620383],[6837.0,45.519376,-73.551951],[4330.0,45.461685,-73.584039],[14634.0,45.471669,-73.582644],[5460.0,45.542983,-73.606381],[6575.0,45.542283,-73.607025],[10266.0,45.514531,-73.568471],[8273.0,45.467666,-73.593918],[4741.0,45.539271,-73.568372],[3909.0,45.502602,-73.527503],[3192.0,45.559482,-73.535496],[1105.0,45.52246,-73.691564],[2102.0,45.494564,-73.638702],[3835.0,45.557895,-73.576529],[4479.0,45.53217,-73.55859],[6832.0,45.506445,-73.556352],[1557.0,45.553262,-73.638615],[12670.0,45.515175,-73.581095],[1356.0,45.4981,-73.652942],[12846.0,45.520188,-73.590559],[448.0,45.53092,-73.57674],[1329.0,45.597221,-73.56279],[11120.0,45.535189,-73.617459],[9195.0,45.507662,-73.57886],[21531.0,45.521342,-73.589419],[845.0,45.526904,-73.624331],[6215.0,45.537159,-73.597476],[14587.0,45.532955,-73.584194],[3120.0,45.550669,-73.549036],[2574.0,45.497504,-73.552845],[2332.0,45.547218,-73.631103],[9060.0,45.504993,-73.571703],[9745.0,45.537159,-73.597476],[19365.0,45.53543,-73.5822],[4818.0,45.543541,-73.600982],[7539.0,45.530505,-73.614708],[9916.0,45.491687,-73.576767],[21143.0,45.520604,-73.575984],[4341.0,45.526058,-73.612969],[2605.0,45.559178,-73.61413],[15334.0,45.522847,-73.583612],[3591.0,45.456498,-73.582144],[6441.0,45.52087,-73.55922],[3621.0,45.540881,-73.584157],[2043.0,45.542119,-73.622547],[5666.0,45.49497,-73.558042],[9267.0,45.477429,-73.600169],[11382.0,45.509864,-73.576105],[7400.0,45.547408,-73.607669],[1764.0,45.544101,-73.667386],[3960.0,45.547227,-73.56987],[8183.0,45.529802,-73.591766],[8051.0,45.498312,-73.567256],[365.0,45.540823,-73.58421],[8008.0,45.540944,-73.547142],[4162.0,45.480846,-73.583862],[5655.0,45.532331,-73.620187],[9164.0,45.533409,-73.570657],[6351.0,45.549911,-73.558263],[7497.0,45.536261,-73.607545],[4171.0,45.531938,-73.617842],[3136.0,45.543376,-73.632267],[5126.0,45.507144,-73.555119],[4353.0,45.539857,-73.620383],[9480.0,45.524025,-73.600422],[5737.0,45.549911,-73.558263],[7150.0,45.524037,-73.590143],[2623.0,45.550737,-73.615222],[8193.0,45.524566,-73.572079],[879.0,45.576944,-73.540453],[3541.0,45.487693,-73.56904],[10748.0,45.529326,-73.574806],[17442.0,45.5171,-73.57921],[162.0,45.543962,-73.610768],[3670.0,45.553571,-73.560494],[2234.0,45.548931,-73.616837],[9167.0,45.50015,-73.56928],[19263.0,45.517,-73.589],[22120.0,45.513338,-73.57295],[4729.0,45.549911,-73.558263],[23558.0,45.51059,-73.57547],[3822.0,45.553262,-73.638615],[4946.0,45.470303,-73.589848],[7155.0,45.527933,-73.601223],[4433.0,45.527117,-73.597736],[10766.0,45.51791,-73.567143],[5456.0,45.542119,-73.622547],[3023.0,45.53057,-73.54913],[5648.0,45.501974,-73.555157],[7549.0,45.517459,-73.597927],[36519.0,45.49659,-73.57851],[9615.0,45.537877,-73.618323],[3214.0,45.521565,-73.535332],[5277.0,45.538589,-73.586302],[5002.0,45.52765,-73.6123],[3212.0,45.47968,-73.56385],[7507.0,45.550316,-73.573358],[7793.0,45.531164,-73.583695],[8850.0,45.52457,-73.59588],[436.0,45.58077,-73.544766],[416.0,45.439145,-73.591737],[6145.0,45.515858,-73.559891],[1704.0,45.579325,-73.570504],[7419.0,45.549605,-73.541939],[2718.0,45.513413,-73.538253],[54351.0,45.49659,-73.57851],[12603.0,45.52317,-73.569013],[4476.0,45.567766,-73.579688],[6183.0,45.546115,-73.557093],[17924.0,45.488855,-73.583593],[5909.0,45.5452,-73.576451],[5744.0,45.541549,-73.565012],[15527.0,45.495415,-73.622341],[17304.0,45.498312,-73.567256],[20335.0,45.521495,-73.596758],[9526.0,45.523035,-73.565395],[9372.0,45.50723,-73.615085],[27357.0,45.518593,-73.581566],[2700.0,45.55197,-73.593775],[1626.0,45.549704,-73.605512],[2561.0,45.554961,-73.610737],[15762.0,45.534134,-73.573492],[7554.0,45.5452,-73.576451],[2363.0,45.54977,-73.554853],[13228.0,45.52708,-73.573278],[14133.0,45.517354,-73.582129],[1909.0,45.549833,-73.640633],[6779.0,45.535828,-73.57869],[19290.0,45.527795,-73.571939],[20801.0,45.520125,-73.595214],[2462.0,45.492222,-73.569348],[3814.0,45.549121,-73.600874],[5887.0,45.530317,-73.554867],[5812.0,45.467369,-73.570769],[1982.0,45.515775,-73.600609],[18205.0,45.518967,-73.583616],[7232.0,45.53318,-73.61544],[4529.0,45.526246,-73.613747],[8996.0,45.499677,-73.578849],[6155.0,45.502326,-73.565138],[8962.0,45.539236,-73.585946],[5542.0,45.494514,-73.583368],[1981.0,45.564353,-73.571244],[3870.0,45.542868,-73.538224],[12106.0,45.546907,-73.551557],[8332.0,45.542119,-73.622547],[6810.0,45.477249,-73.587238],[6867.0,45.529002,-73.546199],[3197.0,45.482779,-73.584525],[3712.0,45.547218,-73.631103],[5043.0,45.547362,-73.579221],[2847.0,45.461685,-73.584039],[5668.0,45.534358,-73.612231],[11551.0,45.49123,-73.580782],[7065.0,45.539932,-73.599343],[1996.0,45.576963,-73.546836],[20136.0,45.517062,-73.559781],[3029.0,45.559828,-73.658325],[42836.0,45.50038,-73.57507],[13090.0,45.49947,-73.57591],[10665.0,45.525048,-73.560036],[3989.0,45.55001,-73.59576],[9676.0,45.517333,-73.574436],[4295.0,45.491384,-73.586013],[21633.0,45.51561,-73.57569],[11799.0,45.533547,-73.566867],[8215.0,45.532977,-73.581222],[4587.0,45.456355,-73.597557],[6117.0,45.550316,-73.573358],[5064.0,45.444408,-73.575568],[1232.0,45.544958,-73.572886],[3997.0,45.549755,-73.605566],[7717.0,45.472668,-73.58539],[6700.0,45.537026,-73.593166],[5767.0,45.521069,-73.563757],[15047.0,45.49606,-73.57348],[3671.0,45.544867,-73.614841],[13740.0,45.49947,-73.57591],[6787.0,45.532926,-73.56378],[6575.0,45.532281,-73.610991],[7427.0,45.499326,-73.571768],[5706.0,45.481491,-73.564754],[12903.0,45.531783,-73.597793],[11943.0,45.505403,-73.621166],[20799.0,45.51496,-73.58503],[16985.0,45.52479,-73.56545],[29076.0,45.51059,-73.57547],[16569.0,45.537041,-73.602026],[15348.0,45.524037,-73.590143],[2830.0,45.480073,-73.585479],[16570.0,45.53565,-73.57172],[4427.0,45.553408,-73.569531],[2489.0,45.456351,-73.582292],[18971.0,45.521495,-73.596758],[3976.0,45.456355,-73.597557],[3651.0,45.481491,-73.564754],[1186.0,45.573733,-73.539446],[9690.0,45.523584,-73.560941],[4089.0,45.468927,-73.61987],[17028.0,45.519216,-73.57718],[4308.0,45.55001,-73.59576],[4598.0,45.526246,-73.613747],[11216.0,45.523525,-73.587509],[9629.0,45.532977,-73.581222],[3855.0,45.514379,-73.610871],[7240.0,45.461078,-73.567307],[14554.0,45.53848,-73.60556],[11026.0,45.517333,-73.574436],[3514.0,45.5499,-73.583084],[7154.0,45.520188,-73.590559],[16697.0,45.512014,-73.532547],[4445.0,45.480914,-73.56846],[6039.0,45.530351,-73.624392],[7534.0,45.523492,-73.56073],[10551.0,45.545776,-73.562175],[17124.0,45.51908,-73.5727],[9427.0,45.526027,-73.577859],[7733.0,45.483006,-73.562187],[4521.0,45.5306,-73.614616],[20049.0,45.510072,-73.570752],[16967.0,45.554947,-73.555295],[20792.0,45.518593,-73.581566],[27072.0,45.49659,-73.57851],[542.0,45.595352,-73.55859],[5882.0,45.539932,-73.599343],[10014.0,45.555004,-73.555355],[17171.0,45.522278,-73.577591],[3277.0,45.531565,-73.616946],[39.0,45.562668,-73.692362],[28924.0,45.51066,-73.56497],[5850.0,45.477107,-73.621438],[5436.0,45.543424,-73.632462],[7924.0,45.526006,-73.548864],[6743.0,45.504993,-73.571703],[8547.0,45.541484,-73.612441],[13862.0,45.522426,-73.574184],[1387.0,45.545542,-73.634748],[6443.0,45.534739,-73.620769],[1717.0,45.485889,-73.577514],[29159.0,45.480208,-73.577599],[7956.0,45.543651,-73.60565],[5329.0,45.542993,-73.606372],[4374.0,45.549686,-73.591155],[4469.0,45.475963,-73.607234],[2153.0,45.545542,-73.634748],[9868.0,45.515604,-73.572135],[17798.0,45.51908,-73.5727],[13731.0,45.517062,-73.559781],[18933.0,45.531917,-73.597983],[12458.0,45.524628,-73.595811],[2447.0,45.529686,-73.601728],[5073.0,45.501441,-73.560144],[8949.0,45.541435,-73.554314],[14416.0,45.533924,-73.562425],[5034.0,45.529155,-73.559411],[3120.0,45.486319,-73.570233],[17689.0,45.515065,-73.55923],[13373.0,45.525048,-73.560036],[5523.0,45.466569,-73.623295],[195.0,45.542645,-73.622445],[8544.0,45.512813,-73.577017],[5003.0,45.54816,-73.592026],[4174.0,45.54467,-73.621581],[28632.0,45.53229,-73.57544],[3292.0,45.499757,-73.629298],[14099.0,45.531327,-73.597888],[3930.0,45.552016,-73.581855],[2024.0,45.557921,-73.529147],[5148.0,45.504777,-73.561189],[7877.0,45.509655,-73.554009],[11126.0,45.50781,-73.57208],[6272.0,45.537628,-73.553394],[10340.0,45.505377,-73.56762],[2185.0,45.506373,-73.524577],[3843.0,45.462362,-73.59665],[2900.0,45.551582,-73.599156],[4111.0,45.542794,-73.618105],[444.0,45.640003,-73.490113],[3479.0,45.535134,-73.554864],[18147.0,45.4918,-73.584006],[2337.0,45.46714,-73.54259],[4103.0,45.559199,-73.599658],[5188.0,45.523276,-73.558218],[7674.0,45.512813,-73.577017],[7509.0,45.534882,-73.609412],[1856.0,45.563921,-73.548912],[3515.0,45.53057,-73.54913],[13647.0,45.522558,-73.574409],[5611.0,45.462742,-73.565845],[2661.0,45.484489,-73.628834],[4526.0,45.542794,-73.618105],[5957.0,45.502326,-73.565138],[2916.0,45.465058,-73.566775],[7276.0,45.539932,-73.599343],[11045.0,45.510625,-73.566903],[19074.0,45.53543,-73.5822],[4267.0,45.477107,-73.621438],[8412.0,45.503063,-73.573986],[4630.0,45.524942,-73.610606],[13260.0,45.524518,-73.571977],[10737.0,45.531958,-73.607001],[23660.0,45.519088,-73.569509],[4499.0,45.494499,-73.574173],[12121.0,45.493029,-73.564818],[3885.0,45.542579,-73.636557],[12881.0,45.515052,-73.559221],[432.0,45.533682,-73.515261],[11370.0,45.521485,-73.58498],[4424.0,45.518309,-73.603286],[6270.0,45.527933,-73.601223],[929.0,45.549833,-73.640633],[4322.0,45.467369,-73.570769],[8287.0,45.478228,-73.569651],[24814.0,45.51059,-73.57547],[11688.0,45.535341,-73.603659],[3584.0,45.534864,-73.555455],[7682.0,45.535248,-73.617564],[1391.0,45.529686,-73.601728],[13624.0,45.523856,-73.600127],[7298.0,45.55896,-73.548687],[4315.0,45.531774,-73.572446],[10006.0,45.53073,-73.58153],[5205.0,45.477923,-73.559038],[4705.0,45.444547,-73.575091],[8028.0,45.506899,-73.623526],[5388.0,45.529219,-73.574777],[8987.0,45.530351,-73.624392],[87.0,45.599086,-73.636948],[2779.0,45.469046,-73.618575],[198.0,45.539663,-73.634475],[7474.0,45.547108,-73.569367],[6620.0,45.53696,-73.61199],[11279.0,45.500779,-73.558826],[7964.0,45.471743,-73.613924],[9261.0,45.53611,-73.576287],[3881.0,45.551937,-73.593928],[7266.0,45.520688,-73.608428],[3945.0,45.552289,-73.613007],[5061.0,45.551774,-73.603478],[27356.0,45.51496,-73.58503],[4878.0,45.542055,-73.597793],[1546.0,45.562299,-73.549443],[6163.0,45.506421,-73.556466],[8245.0,45.48498,-73.560333],[10336.0,45.53341,-73.601124],[2787.0,45.553571,-73.560494],[5564.0,45.533348,-73.605834],[9927.0,45.546978,-73.575515],[779.0,45.576,-73.646765],[2259.0,45.466965,-73.631664],[20375.0,45.502813,-73.576024],[2240.0,45.574984,-73.538849],[293.0,45.478671,-73.565705],[7584.0,45.535709,-73.619624],[14029.0,45.520178,-73.579424],[13305.0,45.497718,-73.56818],[4565.0,45.456498,-73.582144],[3671.0,45.546939,-73.602616],[12638.0,45.529693,-73.567401],[4635.0,45.538589,-73.586302],[9307.0,45.502311,-73.573364],[20772.0,45.534185,-73.573589],[4986.0,45.494527,-73.556753],[11365.0,45.49642,-73.57616],[17396.0,45.52479,-73.56545],[3795.0,45.534453,-73.559652],[595.0,45.527117,-73.597736],[1677.0,45.494805,-73.638772],[14288.0,45.497605,-73.55535],[15968.0,45.49368,-73.579146],[4180.0,45.537813,-73.624872],[14224.0,45.487522,-73.565003],[1214.0,45.558457,-73.65884],[290.0,45.640003,-73.490113],[12672.0,45.525048,-73.560036],[2810.0,45.496302,-73.629822],[8706.0,45.522907,-73.568364],[3194.0,45.4703,-73.6162],[14110.0,45.522278,-73.577591],[11129.0,45.533496,-73.566931],[1369.0,45.553948,-73.611575],[15443.0,45.521461,-73.570614],[155.0,45.570727,-73.737086],[10483.0,45.515604,-73.572135],[19221.0,45.512801,-73.531116],[811.0,45.537226,-73.495067],[7331.0,45.514087,-73.5523],[8362.0,45.498639,-73.574227],[2547.0,45.446534,-73.603541],[655.0,45.557022,-73.659902],[4499.0,45.529033,-73.546348],[36052.0,45.53229,-73.57544],[661.0,45.538627,-73.552839],[15079.0,45.520688,-73.608428],[1883.0,45.447554,-73.577263],[7090.0,45.553571,-73.560494],[9393.0,45.54813,-73.591935],[5576.0,45.521015,-73.563745],[6865.0,45.52697,-73.558021],[2878.0,45.550769,-73.57711],[4164.0,45.500278,-73.559877],[15657.0,45.52479,-73.56545],[20624.0,45.5079,-73.563112],[5181.0,45.459707,-73.571438],[9374.0,45.533348,-73.605834],[1668.0,45.459593,-73.544377],[25773.0,45.527432,-73.579917],[5686.0,45.514114,-73.552645],[14306.0,45.537041,-73.602026],[4119.0,45.550769,-73.620911],[4320.0,45.543541,-73.600982],[11867.0,45.537041,-73.602026],[1806.0,45.542119,-73.622547],[9647.0,45.526557,-73.598276],[27253.0,45.50761,-73.551836],[2924.0,45.4703,-73.6162],[10431.0,45.530351,-73.624392],[5944.0,45.53867,-73.56936],[5339.0,45.532408,-73.564776],[7750.0,45.504561,-73.56279],[9953.0,45.547304,-73.543827],[2626.0,45.549963,-73.534434],[23122.0,45.516091,-73.570129],[88.0,45.649546,-73.491915],[1056.0,45.519049,-73.604981],[8148.0,45.553219,-73.539782],[4255.0,45.485801,-73.595797],[6320.0,45.484018,-73.563437],[10873.0,45.49123,-73.580782],[6035.0,45.52974,-73.59527],[9154.0,45.508205,-73.568814],[12462.0,45.513406,-73.562594],[39.0,45.602728,-73.55793],[5034.0,45.54458,-73.588069],[2350.0,45.492222,-73.569348],[4056.0,45.550476,-73.590446],[9522.0,45.524025,-73.600422],[4113.0,45.544867,-73.614841],[1237.0,45.544101,-73.667386],[3619.0,45.516777,-73.577784],[4829.0,45.547227,-73.56987],[9432.0,45.512832,-73.576888],[4183.0,45.501715,-73.574131],[5374.0,45.489727,-73.564063],[12372.0,45.521161,-73.585616],[18486.0,45.493718,-73.579186],[23804.0,45.498673,-73.552563],[4489.0,45.539833,-73.575559],[7385.0,45.53576,-73.57867],[15874.0,45.529559,-73.587602],[7683.0,45.538958,-73.582432],[29759.0,45.50761,-73.551836],[3115.0,45.539984,-73.638135],[4127.0,45.544513,-73.571564],[4008.0,45.501449,-73.633051],[12481.0,45.514767,-73.565491],[3225.0,45.546939,-73.602616],[4123.0,45.553262,-73.638615],[5948.0,45.470303,-73.589848],[2682.0,45.546094,-73.599279],[11058.0,45.497055,-73.618652],[4529.0,45.498765,-73.554477],[20402.0,45.488855,-73.583593],[3432.0,45.496302,-73.629822],[4894.0,45.526206,-73.54925],[13410.0,45.530871,-73.598345],[11182.0,45.519157,-73.55786],[19379.0,45.525628,-73.581775],[5280.0,45.495372,-73.568589],[5904.0,45.512832,-73.576888],[5034.0,45.504803,-73.5612],[5311.0,45.53318,-73.61544],[11657.0,45.49975,-73.55566],[6546.0,45.457894,-73.567528],[48310.0,45.524673,-73.58255],[6899.0,45.489609,-73.582445],[19249.0,45.490452,-73.560172],[24806.0,45.53229,-73.57544],[14940.0,45.527791,-73.572056],[1904.0,45.444433,-73.575788],[5663.0,45.532408,-73.564776],[5951.0,45.500105,-73.560199],[16916.0,45.523856,-73.600127],[487.0,45.544681,-73.610161],[19817.0,45.524286,-73.604973],[4944.0,45.527363,-73.607723],[1778.0,45.484489,-73.628834],[9260.0,45.49975,-73.55566],[7139.0,45.535189,-73.617459],[3416.0,45.514145,-73.620302],[14888.0,45.527748,-73.597439],[4552.0,45.504803,-73.5612],[475.0,45.536408,-73.512776],[8485.0,45.48204,-73.574863],[10408.0,45.550316,-73.573358],[3986.0,45.450851,-73.594027],[4696.0,45.4829,-73.591],[18296.0,45.521492,-73.570472],[9738.0,45.520688,-73.608428],[3523.0,45.538799,-73.552496],[3772.0,45.515634,-73.606711],[7111.0,45.549911,-73.558263],[1837.0,45.577136,-73.546289],[516.0,45.551886,-73.74444],[4099.0,45.527839,-73.594821],[11384.0,45.532977,-73.581222],[8739.0,45.494872,-73.571124],[3161.0,45.547408,-73.607669],[12669.0,45.508567,-73.577698],[13571.0,45.478796,-73.576424],[2398.0,45.485889,-73.577514],[5330.0,45.475743,-73.564601],[930.0,45.520796,-73.603271],[1946.0,45.55001,-73.59576],[4539.0,45.47968,-73.56385],[2057.0,45.480073,-73.585479],[2722.0,45.533661,-73.552197],[5348.0,45.474565,-73.624036],[3790.0,45.491415,-73.569955],[12229.0,45.52741,-73.603935],[94.0,45.614519,-73.612245],[6818.0,45.501974,-73.555157],[14106.0,45.506371,-73.564201],[3387.0,45.552923,-73.565752],[2861.0,45.550769,-73.57711],[7822.0,45.540881,-73.584157],[2168.0,45.553491,-73.54829],[3779.0,45.527513,-73.598791],[4520.0,45.576963,-73.546836],[8077.0,45.472668,-73.58539],[3670.0,45.479483,-73.61979],[5230.0,45.533195,-73.601173],[10217.0,45.509864,-73.576105],[426.0,45.504347,-73.568949],[2368.0,45.515617,-73.614991],[2857.0,45.486219,-73.573747],[9929.0,45.520295,-73.614824],[10754.0,45.486971,-73.589293],[369.0,45.5332,-73.5156],[4625.0,45.532674,-73.571776],[24990.0,45.53536,-73.603635],[10498.0,45.478228,-73.569651],[3660.0,45.547283,-73.625377],[8766.0,45.51069,-73.57805],[10622.0,45.533706,-73.552179],[10362.0,45.514108,-73.57523],[1878.0,45.560762,-73.650561],[13388.0,45.498673,-73.552563],[10727.0,45.477249,-73.587238],[6543.0,45.522318,-73.606931],[1021.0,45.506843,-73.681757],[4084.0,45.543699,-73.553673],[346.0,45.544471,-73.567251],[478.0,45.576944,-73.540453],[2038.0,45.437914,-73.58274],[2628.0,45.547218,-73.631103],[18686.0,45.5079,-73.563112],[3311.0,45.51339,-73.56255],[14220.0,45.52516,-73.599247],[4594.0,45.539292,-73.541031],[3854.0,45.456086,-73.581937],[3064.0,45.491415,-73.569955],[23151.0,45.53092,-73.57674],[23994.0,45.532514,-73.584811],[13035.0,45.500606,-73.565448],[888.0,45.568225,-73.579436],[2161.0,45.469027,-73.609537],[10375.0,45.5341,-73.58852],[2198.0,45.514905,-73.619682],[2139.0,45.558063,-73.529408],[8552.0,45.527727,-73.586284],[2686.0,45.52589,-73.650034],[5533.0,45.523276,-73.558218],[4217.0,45.484859,-73.571585],[1844.0,45.50919,-73.660321],[4765.0,45.551584,-73.561916],[6501.0,45.547179,-73.551739],[9173.0,45.521094,-73.562251],[3580.0,45.547625,-73.612025],[4571.0,45.481491,-73.564754],[3817.0,45.529686,-73.601728],[3639.0,45.500392,-73.578168],[16811.0,45.474224,-73.603721],[2847.0,45.559884,-73.633568],[13957.0,45.532955,-73.584194],[1767.0,45.55001,-73.59576],[7857.0,45.545776,-73.562175],[5578.0,45.529135,-73.559234],[3632.0,45.544377,-73.581018],[6507.0,45.519581,-73.560116],[18746.0,45.537041,-73.602026],[17692.0,45.492825,-73.557978],[9441.0,45.517333,-73.574436],[21863.0,45.50038,-73.57507],[6478.0,45.485385,-73.628142],[7922.0,45.497055,-73.618653],[3366.0,45.477353,-73.615372],[15079.0,45.536423,-73.570998],[8620.0,45.51164,-73.562184],[6472.0,45.529219,-73.574777],[6379.0,45.538984,-73.582445],[5285.0,45.546408,-73.638428],[13147.0,45.52511,-73.560206],[11556.0,45.497595,-73.555237],[6834.0,45.525021,-73.610737],[2389.0,45.512933,-73.63389],[3331.0,45.477313,-73.615327],[187.0,45.539824,-73.508752],[10989.0,45.53227,-73.56828],[1193.0,45.582757,-73.542443],[17329.0,45.521495,-73.596758],[9019.0,45.53611,-73.576287],[5872.0,45.547997,-73.58523],[4735.0,45.557192,-73.569847],[8386.0,45.538021,-73.583524],[8975.0,45.512797,-73.561462],[2481.0,45.51299,-73.682498],[3622.0,45.444408,-73.575568],[1426.0,45.558063,-73.529408],[3190.0,45.466569,-73.623295],[3230.0,45.543376,-73.632267],[9444.0,45.482098,-73.574876],[10166.0,45.530351,-73.624392],[1593.0,45.570081,-73.573047],[3105.0,45.501302,-73.633161],[1561.0,45.553262,-73.638615],[2525.0,45.4743,-73.6237],[3290.0,45.547227,-73.56987],[2875.0,45.532227,-73.560618],[6964.0,45.494579,-73.558756],[5936.0,45.54467,-73.621581],[43.0,45.503666,-73.569576],[5481.0,45.557921,-73.529147],[9614.0,45.536738,-73.565167],[5450.0,45.539829,-73.620235],[7672.0,45.501402,-73.571814],[1165.0,45.579349,-73.581681],[5041.0,45.534727,-73.614793],[8581.0,45.525048,-73.560036],[8370.0,45.487723,-73.569083],[3779.0,45.526128,-73.546002],[12553.0,45.510086,-73.611429],[8523.0,45.51383,-73.560432],[9290.0,45.502786,-73.559134],[13581.0,45.520179,-73.579424],[8738.0,45.523035,-73.565395],[7271.0,45.519153,-73.616469],[7768.0,45.506421,-73.556466],[5971.0,45.527839,-73.594821],[12669.0,45.486971,-73.589293],[5322.0,45.543763,-73.56772],[6188.0,45.500035,-73.618105],[3874.0,45.52798,-73.623883],[20471.0,45.519146,-73.569241],[9507.0,45.535189,-73.617459],[17500.0,45.51066,-73.56497],[10537.0,45.5341,-73.58852],[14136.0,45.49975,-73.55566],[22342.0,45.516091,-73.570129],[3562.0,45.500234,-73.571126],[1586.0,45.524394,-73.611423],[4652.0,45.539461,-73.576056],[11448.0,45.500675,-73.565354],[14247.0,45.522413,-73.58413],[16099.0,45.534136,-73.595478],[7671.0,45.467666,-73.593918],[2000.0,45.523854,-73.519677],[2313.0,45.540686,-73.653376],[6412.0,45.532281,-73.610991],[10741.0,45.509685,-73.562317],[20085.0,45.503475,-73.572103],[3973.0,45.477353,-73.615372],[5333.0,45.551584,-73.561916],[1823.0,45.531117,-73.632233],[15694.0,45.516091,-73.570129],[1142.0,45.579325,-73.570504],[5236.0,45.523217,-73.558187],[4124.0,45.485801,-73.595797],[18827.0,45.527795,-73.571939],[1458.0,45.567352,-73.653787],[6637.0,45.539293,-73.62078],[7093.0,45.478228,-73.569651],[4297.0,45.50233,-73.566497],[1825.0,45.569789,-73.548081],[5436.0,45.543651,-73.605651],[7739.0,45.502786,-73.559134],[5731.0,45.523194,-73.555843],[4852.0,45.530351,-73.624392],[7401.0,45.53576,-73.57867],[4232.0,45.47968,-73.56385],[4034.0,45.534311,-73.549969],[4627.0,45.523575,-73.623443],[4512.0,45.505312,-73.560891],[4551.0,45.559199,-73.599658],[2483.0,45.547218,-73.631103],[6655.0,45.541435,-73.554314],[3452.0,45.550525,-73.582982],[11379.0,45.500674,-73.561082],[99.0,45.443351,-73.716905],[3923.0,45.54467,-73.621581],[8496.0,45.529993,-73.604214],[3430.0,45.496302,-73.629822],[6503.0,45.4816,-73.6],[4976.0,45.537159,-73.597476],[9677.0,45.48916,-73.567771],[1835.0,45.5536,-73.6616],[7747.0,45.532013,-73.580541],[5332.0,45.481184,-73.579349],[4428.0,45.53923,-73.541082],[8277.0,45.510086,-73.611429],[863.0,45.56169,-73.610512],[5917.0,45.546539,-73.598665],[9954.0,45.524566,-73.572079],[7807.0,45.538887,-73.623907],[4217.0,45.549714,-73.563848],[13918.0,45.486971,-73.589293],[56380.0,45.49659,-73.57851],[19919.0,45.53543,-73.5822],[8523.0,45.511673,-73.562042],[5390.0,45.538738,-73.628022],[4040.0,45.550476,-73.590446],[1.0,45.56842,-73.746873],[4135.0,45.480846,-73.583862],[14001.0,45.53565,-73.57172],[7203.0,45.537964,-73.562643],[75.0,45.525914,-73.689112],[14946.0,45.537088,-73.593218],[10745.0,45.52457,-73.59588],[9181.0,45.53848,-73.60556],[17168.0,45.516779,-73.577403],[5818.0,45.493034,-73.583836],[2575.0,45.535295,-73.55116],[7920.0,45.54257,-73.592144],[9162.0,45.529802,-73.591766],[7392.0,45.530871,-73.598346],[3421.0,45.516079,-73.607798],[15882.0,45.524296,-73.604847],[2220.0,45.489865,-73.564731],[8884.0,45.514396,-73.561766],[47759.0,45.527616,-73.589115],[16087.0,45.514565,-73.568407],[15194.0,45.522233,-73.602082],[9417.0,45.54111,-73.54755],[3699.0,45.545425,-73.556527],[4598.0,45.541406,-73.630071],[5670.0,45.467369,-73.570769],[6445.0,45.536261,-73.607545],[5532.0,45.5367,-73.56081],[15386.0,45.532955,-73.584194],[589.0,45.556708,-73.670634],[9719.0,45.515604,-73.572135],[6340.0,45.506445,-73.556352],[5317.0,45.499191,-73.569421],[9117.0,45.525669,-73.593016],[5983.0,45.461078,-73.567307],[16585.0,45.520278,-73.568216],[10147.0,45.523584,-73.560941],[1051.0,45.601913,-73.524419],[3799.0,45.501449,-73.633051],[29473.0,45.506448,-73.576349],[13274.0,45.510016,-73.570654],[4236.0,45.552842,-73.565295],[4234.0,45.466914,-73.631704],[2712.0,45.549704,-73.605512],[5727.0,45.510042,-73.624835],[9381.0,45.536738,-73.565167],[7848.0,45.501967,-73.555153],[1108.0,45.531944,-73.713605],[6868.0,45.54206,-73.597873],[6600.0,45.538589,-73.586302],[2217.0,45.563872,-73.655568],[13894.0,45.520188,-73.590559],[3763.0,45.529686,-73.601728],[9533.0,45.486971,-73.589293],[21254.0,45.489476,-73.584566],[13909.0,45.502054,-73.573465],[2359.0,45.530871,-73.598346],[1468.0,45.500451,-73.661428],[5485.0,45.531164,-73.583695],[36256.0,45.51941,-73.58685],[4391.0,45.551937,-73.593928],[14178.0,45.53248,-73.58507],[5564.0,45.547172,-73.618521],[4808.0,45.5468,-73.588893],[8361.0,45.527839,-73.594821],[3389.0,45.570081,-73.573047],[7097.0,45.5387,-73.586222],[110.0,45.513178,-73.713061],[6110.0,45.539259,-73.577459],[16083.0,45.534134,-73.573492],[5617.0,45.482826,-73.590843],[5454.0,45.54475,-73.5949],[16887.0,45.519139,-73.557866],[8149.0,45.542119,-73.622547],[4458.0,45.46714,-73.54259],[16476.0,45.5359,-73.622442],[4226.0,45.513741,-73.538382],[4490.0,45.50233,-73.566497],[6078.0,45.51166,-73.562136],[22561.0,45.516427,-73.558112],[793.0,45.560742,-73.65041],[1614.0,45.550698,-73.60953],[3640.0,45.471162,-73.627206],[3179.0,45.564811,-73.595804],[5388.0,45.531177,-73.632136],[2914.0,45.489113,-73.647013],[3182.0,45.495962,-73.573254],[415.0,45.560742,-73.65041],[13827.0,45.524963,-73.575559],[11023.0,45.50176,-73.57128],[24315.0,45.480208,-73.577599],[9731.0,45.493029,-73.564818],[5702.0,45.547304,-73.543827],[3674.0,45.457597,-73.590529],[5558.0,45.504501,-73.617652],[3564.0,45.501399,-73.571786],[9452.0,45.510086,-73.611429],[8499.0,45.506421,-73.556466],[4578.0,45.429471,-73.593099],[4849.0,45.5195,-73.560269],[3277.0,45.544828,-73.610458],[6452.0,45.507144,-73.555119],[184.0,45.5294,-73.5178],[17016.0,45.502386,-73.573338],[5855.0,45.498151,-73.577609],[8067.0,45.512832,-73.576888],[15242.0,45.524037,-73.590143],[11507.0,45.511007,-73.567602],[4654.0,45.541198,-73.590943],[1376.0,45.587531,-73.531907],[706.0,45.552298,-73.707383],[15020.0,45.533924,-73.562425],[39185.0,45.527616,-73.589115],[11003.0,45.53707,-73.61529],[296.0,45.5294,-73.5178],[2302.0,45.553579,-73.662083],[2421.0,45.448262,-73.577856],[2667.0,45.554961,-73.610737],[2786.0,45.553215,-73.58752],[11854.0,45.513986,-73.534294],[2283.0,45.555205,-73.555823],[3673.0,45.543541,-73.600982],[13899.0,45.51829,-73.59235],[6017.0,45.544136,-73.545609],[13998.0,45.50501,-73.57069],[10050.0,45.501165,-73.620548],[3626.0,45.486919,-73.569429],[5537.0,45.450851,-73.594027],[2810.0,45.540881,-73.584157],[12477.0,45.511007,-73.567602],[27659.0,45.51496,-73.58503],[2246.0,45.511119,-73.567974],[17725.0,45.53536,-73.603635],[17483.0,45.517062,-73.559781],[2223.0,45.499757,-73.629298],[10064.0,45.545776,-73.562175],[1961.0,45.550737,-73.615222],[12975.0,45.50176,-73.57128],[6352.0,45.470366,-73.589816],[5568.0,45.472668,-73.58539],[8114.0,45.52951,-73.61068],[20906.0,45.522278,-73.577591],[3687.0,45.466914,-73.631704],[15238.0,45.493029,-73.564818],[2555.0,45.552923,-73.565752],[19770.0,45.534185,-73.573589],[6456.0,45.557192,-73.569847],[2062.0,45.53057,-73.54913],[3917.0,45.499757,-73.629298],[3628.0,45.475743,-73.564601],[3781.0,45.549963,-73.534434],[1169.0,45.429711,-73.633883],[2814.0,45.547648,-73.665566],[2514.0,45.456351,-73.582292],[9625.0,45.527041,-73.593471],[3838.0,45.547362,-73.579221],[39413.0,45.515299,-73.561273],[7909.0,45.549686,-73.591155],[11791.0,45.498639,-73.574227],[13776.0,45.522866,-73.568377],[2086.0,45.549704,-73.605512],[4851.0,45.527933,-73.601223],[9034.0,45.544376,-73.595645],[968.0,45.508978,-73.698564],[5523.0,45.564937,-73.555591],[3608.0,45.493034,-73.583836],[18613.0,45.519216,-73.57718],[3176.0,45.493569,-73.568576],[4770.0,45.540881,-73.584157],[7865.0,45.522907,-73.568364],[7309.0,45.532331,-73.620187],[1780.0,45.563921,-73.548912],[7381.0,45.519581,-73.560116],[3046.0,45.523319,-73.520127],[2465.0,45.52687,-73.626723],[8258.0,45.497595,-73.555237],[6389.0,45.544867,-73.614841],[11806.0,45.49123,-73.580782],[2292.0,45.471162,-73.627206],[8619.0,45.53696,-73.61199],[5248.0,45.456885,-73.572607],[4837.0,45.54475,-73.5949],[11437.0,45.512587,-73.573739],[8770.0,45.533661,-73.552197],[3931.0,45.552443,-73.603002],[2511.0,45.539663,-73.634475],[4026.0,45.549714,-73.563848],[4770.0,45.547397,-73.607664],[11495.0,45.505011,-73.572028],[13095.0,45.533924,-73.562425],[6880.0,45.547227,-73.56987],[881.0,45.523413,-73.623526],[17306.0,45.526712,-73.584838],[4362.0,45.559842,-73.615447],[1247.0,45.592803,-73.510466],[2025.0,45.457597,-73.590529],[589.0,45.607616,-73.509822],[12044.0,45.497411,-73.578271],[7206.0,45.549911,-73.558263],[13600.0,45.53848,-73.60556],[7151.0,45.491687,-73.576767],[4909.0,45.461685,-73.584039],[2953.0,45.552923,-73.565752],[12441.0,45.49642,-73.57616],[313.0,45.611231,-73.531242],[6601.0,45.550684,-73.615293],[12986.0,45.500779,-73.558826],[6628.0,45.561594,-73.585643],[3734.0,45.537964,-73.562643],[7398.0,45.530365,-73.584393],[3581.0,45.553027,-73.532382],[2257.0,45.554961,-73.610737],[6578.0,45.489609,-73.582445],[1921.0,45.496302,-73.629822],[912.0,45.537226,-73.495067],[66.0,45.56003,-73.681336],[21847.0,45.516876,-73.57946],[756.0,45.435996,-73.680111],[14728.0,45.52697,-73.558021],[5610.0,45.544846,-73.610536],[37.0,45.550223,-73.705467],[12570.0,45.497697,-73.568646],[237.0,45.495709,-73.576952],[16702.0,45.517921,-73.567168],[20483.0,45.523026,-73.60184],[4409.0,45.54816,-73.592026],[254.0,45.5332,-73.5156],[10767.0,45.477429,-73.600169],[8478.0,45.527738,-73.576249],[9794.0,45.507662,-73.57886],[9065.0,45.512614,-73.573726],[3837.0,45.531834,-73.553478],[195.0,45.550784,-73.5341],[11375.0,45.530816,-73.581626],[7.0,45.640739,-73.500826],[2512.0,45.570081,-73.573047],[4592.0,45.509685,-73.562317],[2949.0,45.546408,-73.638428],[3538.0,45.575707,-73.561562],[29762.0,45.527616,-73.589115],[6911.0,45.508567,-73.577698],[15915.0,45.514565,-73.568407],[11596.0,45.533706,-73.552179],[1662.0,45.456823,-73.546804],[52.0,45.63138,-73.506857],[2111.0,45.494198,-73.563748],[15039.0,45.520178,-73.579424],[23828.0,45.521116,-73.594784],[2723.0,45.549868,-73.647657],[5213.0,45.537553,-73.589939],[5290.0,45.525643,-73.581748],[11907.0,45.512734,-73.561141],[6584.0,45.500043,-73.560412],[4046.0,45.467369,-73.570769],[9543.0,45.54167,-73.607604],[5536.0,45.49486,-73.57108],[4657.0,45.556262,-73.537441],[29413.0,45.506448,-73.576349],[1013.0,45.437914,-73.58274],[9749.0,45.529219,-73.574777],[20242.0,45.535854,-73.572055],[16547.0,45.497515,-73.552571],[1491.0,45.557513,-73.601795],[5477.0,45.482826,-73.590843],[7785.0,45.520524,-73.567697],[2831.0,45.52687,-73.626723],[5004.0,45.540396,-73.616818],[4253.0,45.54257,-73.592144],[11386.0,45.53569,-73.565923],[5214.0,45.483006,-73.562187],[579.0,45.539824,-73.508752],[2115.0,45.444043,-73.613752],[4830.0,45.539833,-73.575559],[7283.0,45.533348,-73.605834],[3279.0,45.526798,-73.624213],[4236.0,45.471926,-73.58247],[9431.0,45.549598,-73.541874],[877.0,45.52246,-73.691564],[2854.0,45.546406,-73.598421],[1074.0,45.579325,-73.570504],[4405.0,45.551119,-73.540764],[4075.0,45.529033,-73.54641],[13914.0,45.53227,-73.56828],[21520.0,45.503475,-73.572103],[44428.0,45.524673,-73.58255],[4699.0,45.503063,-73.573986],[10800.0,45.527849,-73.591861],[16392.0,45.49947,-73.57591],[7702.0,45.509685,-73.562317],[10009.0,45.529559,-73.587602],[12662.0,45.529802,-73.57029],[5678.0,45.481491,-73.564754],[5288.0,45.54467,-73.621581],[4001.0,45.549963,-73.534434],[9402.0,45.501402,-73.571814],[1969.0,45.491513,-73.633649],[22286.0,45.526712,-73.584838],[2795.0,45.515617,-73.614991],[1639.0,45.512559,-73.677011],[3235.0,45.547581,-73.665234],[2132.0,45.52114,-73.54926],[11253.0,45.537114,-73.571003],[9665.0,45.507662,-73.57886],[4408.0,45.554152,-73.547494],[14318.0,45.527009,-73.585766],[2178.0,45.556905,-73.58925],[31045.0,45.504242,-73.553469],[6574.0,45.497605,-73.55535],[2937.0,45.549868,-73.647657],[17452.0,45.506775,-73.567742],[3720.0,45.547218,-73.631103],[1579.0,45.496302,-73.629822],[14931.0,45.518977,-73.583632],[2171.0,45.515617,-73.614991],[4557.0,45.456884,-73.572607],[12300.0,45.50235,-73.566583],[3289.0,45.486919,-73.569429],[28.0,45.56842,-73.746873],[15943.0,45.52479,-73.56545],[2330.0,45.559482,-73.535496],[734.0,45.426124,-73.657416],[1200.0,45.570081,-73.573047],[4881.0,45.547997,-73.58523],[2232.0,45.549126,-73.600912],[8815.0,45.491687,-73.576767],[8218.0,45.541729,-73.565494],[8916.0,45.510351,-73.556508],[7073.0,45.505263,-73.57048],[19784.0,45.528758,-73.57868],[2552.0,45.505973,-73.629593],[254.0,45.503982,-73.62284],[109.0,45.472561,-73.569989],[4968.0,45.52114,-73.54926],[5776.0,45.482826,-73.590843],[11905.0,45.52697,-73.60261],[7686.0,45.518128,-73.561851],[13462.0,45.51829,-73.59235],[3839.0,45.482779,-73.584525],[16765.0,45.493718,-73.579186],[5290.0,45.518341,-73.616647],[10129.0,45.530046,-73.58677],[6237.0,45.507144,-73.555119],[1433.0,45.460729,-73.634073],[1760.0,45.587778,-73.569354],[7368.0,45.540881,-73.584157],[2005.0,45.485889,-73.577514],[15183.0,45.53707,-73.61529],[3859.0,45.546696,-73.588738],[4461.0,45.46714,-73.54259],[17230.0,45.520604,-73.575984],[12315.0,45.503215,-73.569979],[3228.0,45.5566,-73.5373],[33241.0,45.50708,-73.56917],[14097.0,45.517354,-73.582129],[5590.0,45.523217,-73.558187],[5010.0,45.539324,-73.595878],[15221.0,45.533314,-73.583737],[6842.0,45.531818,-73.565317],[5665.0,45.553571,-73.560494],[3508.0,45.468927,-73.61987],[2466.0,45.528,-73.619236],[17180.0,45.523856,-73.600127],[4899.0,45.537813,-73.624872],[1795.0,45.526904,-73.624331],[12757.0,45.52697,-73.558021],[15893.0,45.502145,-73.573427],[14990.0,45.527731,-73.557538],[2408.0,45.466569,-73.623295],[14781.0,45.52729,-73.56463],[5201.0,45.549911,-73.558263],[3863.0,45.541766,-73.626128],[24.0,45.500035,-73.618105],[23845.0,45.51086,-73.54983],[858.0,45.534311,-73.549969],[15177.0,45.531917,-73.597983],[10833.0,45.500231,-73.569394],[11672.0,45.520688,-73.608428],[4323.0,45.482779,-73.584525],[11421.0,45.507975,-73.563072],[7056.0,45.553194,-73.539814],[635.0,45.539807,-73.687266],[5912.0,45.477249,-73.587238],[8546.0,45.533496,-73.566931],[13154.0,45.527758,-73.576185],[5446.0,45.52114,-73.54926],[4523.0,45.446934,-73.572199],[1863.0,45.435665,-73.605652],[17372.0,45.49803,-73.552665],[4407.0,45.550723,-73.609571],[3925.0,45.45073,-73.572575],[15686.0,45.527791,-73.572056],[5368.0,45.490874,-73.562389],[7922.0,45.550316,-73.573358],[647.0,45.435794,-73.68019],[12941.0,45.505403,-73.621166],[4087.0,45.553139,-73.602711],[20138.0,45.515299,-73.561273],[3674.0,45.491513,-73.633649],[10010.0,45.539737,-73.614256],[4652.0,45.509655,-73.554009],[2082.0,45.457597,-73.590529],[22578.0,45.518593,-73.581566],[6977.0,45.512813,-73.577017],[6602.0,45.529155,-73.559411],[5277.0,45.483928,-73.577312],[6977.0,45.54006,-73.60897],[24448.0,45.51941,-73.58685],[14916.0,45.50781,-73.57208],[4871.0,45.54918,-73.593858],[7561.0,45.534882,-73.609412],[6846.0,45.480914,-73.56846],[7134.0,45.53867,-73.56936],[1596.0,45.558457,-73.65884],[3587.0,45.531203,-73.632147],[10396.0,45.512495,-73.550878],[1322.0,45.580315,-73.535314],[2911.0,45.482779,-73.584525],[3648.0,45.534453,-73.559652],[4776.0,45.538021,-73.583524],[7643.0,45.522586,-73.612658],[8232.0,45.544376,-73.595645],[10187.0,45.529971,-73.563053],[12730.0,45.527513,-73.598791],[11411.0,45.515901,-73.559948],[3656.0,45.431471,-73.671205],[28224.0,45.51735,-73.56906],[7428.0,45.520271,-73.614849],[5164.0,45.52114,-73.54926],[13492.0,45.53248,-73.58507],[6780.0,45.537964,-73.562643],[3524.0,45.5499,-73.583084],[3833.0,45.481367,-73.564319],[3908.0,45.489186,-73.5719],[5866.0,45.505377,-73.56762],[1413.0,45.5536,-73.6616],[5245.0,45.544377,-73.581018],[10388.0,45.500685,-73.572156],[2883.0,45.482852,-73.630344],[1383.0,45.504014,-73.704427],[5454.0,45.516783,-73.611423],[3593.0,45.477353,-73.615372],[325.0,45.537226,-73.495067],[6085.0,45.50235,-73.566583],[1472.0,45.52445,-73.605174],[890.0,45.494044,-73.654495],[26077.0,45.529408,-73.578154],[6294.0,45.489951,-73.567091],[7455.0,45.534563,-73.554787],[4061.0,45.540627,-73.630089],[17011.0,45.524505,-73.594142],[3932.0,45.557192,-73.569847],[6709.0,45.541247,-73.591044],[1289.0,45.432894,-73.596853],[10626.0,45.52697,-73.558021],[12665.0,45.521461,-73.570614],[6036.0,45.539224,-73.573837],[5652.0,45.553194,-73.539814],[7703.0,45.52931,-73.588657],[812.0,45.495323,-73.573887],[20436.0,45.502054,-73.573465],[2067.0,45.466965,-73.631664],[3676.0,45.463254,-73.575879],[10954.0,45.543712,-73.628284],[6791.0,45.530505,-73.614708],[11245.0,45.484018,-73.563437],[6315.0,45.461078,-73.567307],[3466.0,45.470544,-73.63168],[3299.0,45.456358,-73.64013],[3977.0,45.543424,-73.632462],[12068.0,45.521461,-73.570614],[5925.0,45.46714,-73.54259],[2744.0,45.4703,-73.6162],[15821.0,45.522233,-73.602082],[6645.0,45.459707,-73.571438],[5257.0,45.548004,-73.58139],[9689.0,45.51069,-73.57805],[8330.0,45.515065,-73.55923],[9085.0,45.500832,-73.57246],[21174.0,45.5171,-73.57921],[8182.0,45.519373,-73.551958],[5018.0,45.495581,-73.553711],[2659.0,45.559828,-73.658325],[2026.0,45.491513,-73.633649],[3578.0,45.538799,-73.552496],[1416.0,45.494805,-73.638772],[17477.0,45.49368,-73.579146],[10834.0,45.516818,-73.554188],[11366.0,45.527426,-73.579918],[629.0,45.437914,-73.58274],[3946.0,45.54078,-73.630401],[2561.0,45.477313,-73.615327],[11203.0,45.530871,-73.598346],[8226.0,45.54111,-73.54755],[8349.0,45.53318,-73.61544],[8716.0,45.50297,-73.57505],[8049.0,45.53576,-73.57867],[17502.0,45.515038,-73.559201],[5533.0,45.461685,-73.584039],[4088.0,45.498578,-73.560242],[2875.0,45.539632,-73.634448],[5221.0,45.540154,-73.623531],[13726.0,45.520578,-73.567733],[17862.0,45.521496,-73.596827],[19416.0,45.507629,-73.551876],[876.0,45.560742,-73.65041],[2114.0,45.482778,-73.584621],[5519.0,45.549695,-73.554647],[9852.0,45.470696,-73.565422],[9621.0,45.527041,-73.593471],[7392.0,45.541484,-73.612441],[2679.0,45.448262,-73.577856],[976.0,45.563947,-73.655573],[2842.0,45.523575,-73.623443],[6071.0,45.4891,-73.57656],[3725.0,45.531367,-73.562624],[8993.0,45.509418,-73.561358],[11286.0,45.527738,-73.576249],[10453.0,45.529971,-73.563053],[3567.0,45.542794,-73.618105],[553.0,45.434898,-73.670819],[3651.0,45.531938,-73.617842],[28567.0,45.518593,-73.581566],[7677.0,45.53703,-73.58584],[4505.0,45.525933,-73.595253],[4894.0,45.537877,-73.618323],[5618.0,45.489186,-73.5719],[7394.0,45.553898,-73.571387],[22320.0,45.513303,-73.572961],[5692.0,45.5306,-73.614616],[67.0,45.520056,-73.677952],[9423.0,45.526027,-73.577859],[724.0,45.556708,-73.670634],[5500.0,45.48204,-73.574863],[6225.0,45.47993,-73.555921],[2364.0,45.554931,-73.610711],[2694.0,45.486279,-73.573599],[2578.0,45.490628,-73.628878],[4426.0,45.537964,-73.562643],[5274.0,45.544867,-73.614841],[61.0,45.58116,-73.641243],[16024.0,45.528746,-73.578654],[20752.0,45.52689,-73.57264],[7562.0,45.464893,-73.566998],[13724.0,45.487531,-73.56507],[2591.0,45.477107,-73.621438],[45600.0,45.524673,-73.58255],[4389.0,45.551937,-73.593928],[3465.0,45.485889,-73.577514],[1569.0,45.452858,-73.589249],[7277.0,45.505305,-73.581002],[18739.0,45.515092,-73.581142],[13586.0,45.492897,-73.580294],[8614.0,45.482098,-73.574876],[4167.0,45.486452,-73.595234],[3916.0,45.5367,-73.56081],[15943.0,45.527616,-73.589115],[2059.0,45.545425,-73.556527],[2713.0,45.514905,-73.619682],[16892.0,45.49871,-73.56671],[5851.0,45.50235,-73.566583],[5253.0,45.470696,-73.565422],[6879.0,45.505305,-73.581002],[2844.0,45.521069,-73.563757],[1775.0,45.515617,-73.614991],[17492.0,45.525931,-73.598839],[9967.0,45.509418,-73.561358],[10144.0,45.501441,-73.560144],[8101.0,45.536123,-73.622445],[2558.0,45.556905,-73.58925],[4325.0,45.457509,-73.639485],[16236.0,45.50971,-73.57354],[1685.0,45.535295,-73.55116],[8672.0,45.543583,-73.6284],[13096.0,45.523544,-73.587332],[3307.0,45.559842,-73.615447],[2296.0,45.5566,-73.5373],[14378.0,45.507402,-73.578444],[18686.0,45.519088,-73.569509],[5967.0,45.543651,-73.605651],[3502.0,45.553262,-73.638615],[2203.0,45.556905,-73.58925],[6869.0,45.462362,-73.59665],[4521.0,45.547283,-73.625377],[57.0,45.606235,-73.628784],[13850.0,45.51908,-73.5727],[9.0,45.525914,-73.689112],[1147.0,45.449901,-73.600251],[17063.0,45.524287,-73.604973],[7793.0,45.518128,-73.561851],[6798.0,45.497697,-73.568646],[5978.0,45.51339,-73.56255],[21799.0,45.53229,-73.57544],[6267.0,45.537877,-73.618323],[7238.0,45.529802,-73.591766],[1901.0,45.508439,-73.672965],[8908.0,45.541447,-73.565],[4412.0,45.539259,-73.577338],[6592.0,45.484018,-73.563437],[11976.0,45.516818,-73.554188],[9308.0,45.515172,-73.581002],[8531.0,45.511638,-73.574635],[2658.0,45.498468,-73.568434],[32159.0,45.51484,-73.584779],[4828.0,45.523217,-73.558187],[2824.0,45.55325,-73.54822],[43799.0,45.524673,-73.58255],[7689.0,45.533195,-73.601173],[5724.0,45.551173,-73.541011],[17215.0,45.526252,-73.589758],[4360.0,45.477923,-73.559038],[2721.0,45.486219,-73.573747],[7984.0,45.515172,-73.581002],[10755.0,45.472668,-73.58539],[2312.0,45.515617,-73.614991],[3867.0,45.530871,-73.598346],[26016.0,45.523615,-73.592457],[4751.0,45.542055,-73.597793],[16827.0,45.530347,-73.566988],[21704.0,45.53092,-73.57674],[15243.0,45.51252,-73.57062],[9638.0,45.54172,-73.565177],[10864.0,45.487723,-73.569083],[14394.0,45.529559,-73.587602],[13530.0,45.525177,-73.595446],[4376.0,45.546939,-73.602616],[2526.0,45.539632,-73.634448],[4290.0,45.529033,-73.54641],[4713.0,45.493569,-73.568576],[2999.0,45.43074,-73.591911],[1907.0,45.546124,-73.646647],[23726.0,45.512541,-73.570677],[5729.0,45.54046,-73.540303],[1627.0,45.534557,-73.55075],[35415.0,45.50038,-73.57507],[12172.0,45.497697,-73.568646],[6463.0,45.533661,-73.552197],[1994.0,45.538799,-73.552496],[4306.0,45.539833,-73.575559],[7905.0,45.537114,-73.569139],[1333.0,45.567352,-73.653787],[14106.0,45.536785,-73.614888],[5919.0,45.547329,-73.625329],[2849.0,45.544594,-73.537795],[3144.0,45.478432,-73.624899],[13864.0,45.524025,-73.600422],[8820.0,45.49642,-73.57616],[10579.0,45.54006,-73.60897],[23445.0,45.531955,-73.59816],[6626.0,45.550692,-73.656367],[1474.0,45.497024,-73.559223],[8.0,45.500234,-73.571126],[9813.0,45.467666,-73.593918],[7940.0,45.472486,-73.58355],[3310.0,45.501302,-73.633161],[3681.0,45.549121,-73.600874],[9085.0,45.51889,-73.56353],[14673.0,45.525643,-73.581748],[3640.0,45.552923,-73.565752],[7251.0,45.470696,-73.565422],[12632.0,45.546978,-73.575515],[4252.0,45.47993,-73.555921],[16799.0,45.51496,-73.58503],[6936.0,45.525021,-73.610737],[21018.0,45.5079,-73.563112],[3121.0,45.513741,-73.538382],[27831.0,45.521116,-73.594784],[2488.0,45.546036,-73.646599],[5098.0,45.541729,-73.565494],[486.0,45.430543,-73.669024],[14350.0,45.527791,-73.572056],[11715.0,45.488855,-73.583593],[3759.0,45.505721,-73.629459],[4192.0,45.575054,-73.538932],[1976.0,45.512894,-73.561248],[1816.0,45.567352,-73.653787],[12137.0,45.519396,-73.583227],[1754.0,45.5536,-73.6616],[8763.0,45.459488,-73.572092],[16778.0,45.521495,-73.596758],[1973.0,45.577136,-73.546289],[9258.0,45.520188,-73.590559],[4858.0,45.543763,-73.56772],[5937.0,45.551119,-73.540764],[15698.0,45.52353,-73.55199],[3564.0,45.456355,-73.597557],[6825.0,45.5478,-73.584867],[8371.0,45.477429,-73.600169],[9923.0,45.507097,-73.568442],[5439.0,45.525021,-73.610737],[3592.0,45.527363,-73.607723],[21208.0,45.524286,-73.604973],[15984.0,45.52697,-73.60261],[6128.0,45.558281,-73.583159],[2058.0,45.53057,-73.54913],[7115.0,45.481491,-73.564754],[2574.0,45.502828,-73.527793],[11185.0,45.527231,-73.586726],[4316.0,45.485385,-73.628142],[13988.0,45.521051,-73.571137],[4036.0,45.543347,-73.632353],[7873.0,45.532978,-73.581222],[16865.0,45.517354,-73.582129],[7499.0,45.487693,-73.56904],[2979.0,45.486919,-73.569429],[7714.0,45.545776,-73.562175],[4368.0,45.4771,-73.6212],[8646.0,45.516792,-73.564239],[3950.0,45.550723,-73.609571],[8527.0,45.525983,-73.577808],[37465.0,45.51941,-73.58685],[11745.0,45.525109,-73.560206],[5303.0,45.531367,-73.562624],[2100.0,45.562509,-73.61656],[8185.0,45.54006,-73.60897],[3765.0,45.525115,-73.586006],[3467.0,45.478432,-73.624899],[13641.0,45.516873,-73.554175],[9563.0,45.521156,-73.585589],[12798.0,45.523171,-73.573431],[13849.0,45.500674,-73.561082],[500.0,45.562668,-73.692362],[3130.0,45.535753,-73.557941],[5679.0,45.538544,-73.576133],[4022.0,45.541766,-73.626128],[981.0,45.426124,-73.657416],[1959.0,45.494805,-73.638772],[5203.0,45.477249,-73.587238],[2368.0,45.558376,-73.658844],[2326.0,45.550056,-73.534576],[5127.0,45.471743,-73.613924],[4097.0,45.547283,-73.625377],[6742.0,45.532372,-73.606432],[7247.0,45.5265,-73.545753],[15279.0,45.532436,-73.606655],[5106.0,45.500606,-73.565448],[8679.0,45.502786,-73.559134],[4004.0,45.470303,-73.589848],[2619.0,45.554931,-73.610711],[8683.0,45.535834,-73.587021],[349.0,45.641557,-73.500375],[30372.0,45.50038,-73.57507],[15508.0,45.53848,-73.60556],[5375.0,45.453016,-73.571915],[9265.0,45.546978,-73.575515],[18470.0,45.535506,-73.603882],[16157.0,45.522894,-73.601816],[15388.0,45.51908,-73.5727],[16997.0,45.510016,-73.570654],[8534.0,45.510142,-73.624752],[3589.0,45.547295,-73.578999],[10254.0,45.513405,-73.562594],[15341.0,45.533924,-73.562425],[2763.0,45.5566,-73.5373],[1026.0,45.447916,-73.583819],[2450.0,45.523854,-73.519677],[3080.0,45.53057,-73.54913],[12313.0,45.519139,-73.557866],[2151.0,45.562509,-73.61656],[6876.0,45.52089,-73.559221],[3805.0,45.535295,-73.55116],[2891.0,45.515384,-73.606408],[18875.0,45.537041,-73.602026],[5359.0,45.55896,-73.548687],[14472.0,45.550306,-73.573353],[374.0,45.536408,-73.512776],[2550.0,45.557921,-73.529147],[28010.0,45.51484,-73.584779],[6999.0,45.544819,-73.62151],[18448.0,45.509932,-73.563807],[9467.0,45.51086,-73.54983],[1839.0,45.503858,-73.571094],[7466.0,45.539224,-73.573837],[2895.0,45.521565,-73.535332],[7534.0,45.531985,-73.580675],[6688.0,45.54006,-73.60897],[1692.0,45.563872,-73.655568],[1032.0,45.58077,-73.544766],[18835.0,45.521342,-73.589419],[26118.0,45.508205,-73.568814],[6333.0,45.537114,-73.569139],[3292.0,45.471743,-73.613924],[5705.0,45.487693,-73.56904],[4757.0,45.546939,-73.602616],[10438.0,45.4952,-73.56328],[6838.0,45.522225,-73.606687],[12853.0,45.521161,-73.585616],[6232.0,45.528449,-73.551085],[7327.0,45.530027,-73.563176],[12215.0,45.508567,-73.577698],[4001.0,45.541766,-73.626128],[7633.0,45.56833,-73.566663],[2620.0,45.56169,-73.610512],[6612.0,45.5367,-73.56081],[138.0,45.47188,-73.53798],[4776.0,45.547997,-73.58523],[14961.0,45.49803,-73.552665],[18869.0,45.51059,-73.57547],[3456.0,45.550723,-73.609571],[12768.0,45.54006,-73.60897],[4255.0,45.480495,-73.577827],[103.0,45.579406,-73.602589],[20719.0,45.50501,-73.57069],[5525.0,45.540951,-73.626375],[1063.0,45.504561,-73.56279],[2074.0,45.572789,-73.540313],[2898.0,45.459986,-73.543956],[15657.0,45.492838,-73.55642],[1938.0,45.434434,-73.586694],[5838.0,45.5367,-73.56081],[6072.0,45.502828,-73.527793],[9098.0,45.511638,-73.574635],[7374.0,45.533409,-73.570657],[8414.0,45.472668,-73.58539],[6250.0,45.48204,-73.574863],[7792.0,45.541729,-73.565494],[12313.0,45.527758,-73.576185],[13184.0,45.534136,-73.595478],[13741.0,45.496265,-73.621544],[1126.0,45.497047,-73.643216],[4098.0,45.495894,-73.560097],[11194.0,45.53227,-73.56828],[4656.0,45.502525,-73.580337],[6161.0,45.541729,-73.565494],[14568.0,45.523171,-73.573431],[1883.0,45.456351,-73.582292],[6380.0,45.51383,-73.560432],[2948.0,45.557895,-73.576529],[2925.0,45.493986,-73.563279],[8854.0,45.501399,-73.571786],[10024.0,45.529993,-73.604214],[21747.0,45.534185,-73.573589],[5567.0,45.5452,-73.576451],[59.0,45.517583,-73.726614],[5048.0,45.535295,-73.55116],[19298.0,45.527795,-73.571939],[6639.0,45.507144,-73.555119],[6721.0,45.480495,-73.577827],[16923.0,45.509813,-73.563895],[3642.0,45.557895,-73.576529],[2651.0,45.523854,-73.519677],[19359.0,45.53536,-73.603635],[4452.0,45.535295,-73.55116],[5212.0,45.510142,-73.624752],[9845.0,45.484947,-73.56036],[3427.0,45.550692,-73.656367],[6228.0,45.533348,-73.605834],[7629.0,45.544311,-73.545549],[5343.0,45.54078,-73.630401],[2379.0,45.537877,-73.618323],[6443.0,45.523507,-73.576413],[4028.0,45.543347,-73.632353],[8724.0,45.50345,-73.568404],[3817.0,45.447044,-73.602178],[6588.0,45.544599,-73.588223],[5843.0,45.503005,-73.573857],[14485.0,45.486971,-73.589293],[144.0,45.440347,-73.702313],[1341.0,45.558481,-73.583304],[13343.0,45.500685,-73.572156],[326.0,45.491415,-73.569955],[10909.0,45.497092,-73.575549],[11200.0,45.499326,-73.571768],[2026.0,45.480073,-73.585479],[2424.0,45.559482,-73.535496],[17750.0,45.508141,-73.57493],[9567.0,45.52697,-73.60261],[949.0,45.605902,-73.53437],[14884.0,45.525172,-73.599375],[825.0,45.560742,-73.65041],[13966.0,45.493029,-73.564818],[11371.0,45.550306,-73.573353],[13532.0,45.514767,-73.565491],[3528.0,45.55001,-73.59576],[5719.0,45.534882,-73.609412],[10671.0,45.500832,-73.572459],[12820.0,45.535247,-73.617712],[9300.0,45.525933,-73.595253],[3639.0,45.531367,-73.562624],[15806.0,45.478228,-73.569651],[15247.0,45.51829,-73.59235],[2100.0,45.512994,-73.682499],[7277.0,45.536123,-73.622445],[7110.0,45.541247,-73.591045],[22332.0,45.531327,-73.597888],[16180.0,45.504095,-73.568696],[8466.0,45.51166,-73.562136],[1727.0,45.456355,-73.597557],[6154.0,45.534563,-73.554786],[6584.0,45.53927,-73.620879],[2987.0,45.550668,-73.549502],[1156.0,45.500451,-73.661428],[8791.0,45.51069,-73.57805],[8156.0,45.5028,-73.55917],[6205.0,45.475743,-73.564601],[6534.0,45.52087,-73.55922],[17632.0,45.478228,-73.569651],[6914.0,45.52087,-73.55922],[12545.0,45.526543,-73.598233],[25040.0,45.51086,-73.54983],[2112.0,45.460729,-73.634073],[87.0,45.528631,-73.677312],[12222.0,45.519895,-73.596827],[8026.0,45.527933,-73.601223],[6444.0,45.482941,-73.579742],[3163.0,45.558281,-73.583159],[7128.0,45.491226,-73.58763],[7968.0,45.524037,-73.590143],[10813.0,45.509188,-73.55458],[8138.0,45.519581,-73.560116],[12408.0,45.536153,-73.622421],[3085.0,45.544513,-73.571564],[11674.0,45.50297,-73.57505],[4928.0,45.536273,-73.578197],[6456.0,45.484896,-73.560119],[8807.0,45.495955,-73.62186],[13189.0,45.50781,-73.57208],[11291.0,45.51829,-73.59235],[7044.0,45.522225,-73.606687],[1441.0,45.520923,-73.559177],[7859.0,45.5215,-73.5851],[17913.0,45.522278,-73.577591],[19362.0,45.51908,-73.5727],[8415.0,45.529802,-73.591766],[4837.0,45.554152,-73.547494],[6027.0,45.541198,-73.590943],[8883.0,45.522907,-73.568364],[10473.0,45.520271,-73.614849],[4287.0,45.472668,-73.58539],[11732.0,45.52931,-73.588657],[2717.0,45.501302,-73.633161],[2154.0,45.485889,-73.577514],[345.0,45.52841,-73.517166],[8127.0,45.51164,-73.562184],[5266.0,45.495581,-73.553711],[13699.0,45.520604,-73.575984],[9055.0,45.50345,-73.568404],[2607.0,45.422451,-73.601194],[4603.0,45.527748,-73.597439],[313.0,45.504635,-73.704432],[17054.0,45.473868,-73.604538],[1596.0,45.437914,-73.58274],[11501.0,45.553879,-73.551761],[4223.0,45.539984,-73.638135],[4791.0,45.539259,-73.577338],[11851.0,45.50297,-73.57505],[3683.0,45.483928,-73.577312],[4100.0,45.541955,-73.629942],[1719.0,45.460729,-73.634073],[7506.0,45.522586,-73.612658],[10882.0,45.533349,-73.601136],[667.0,45.592596,-73.566175],[12331.0,45.471926,-73.58247],[12104.0,45.519157,-73.55786],[3327.0,45.499757,-73.629298],[3806.0,45.540627,-73.630089],[6742.0,45.523304,-73.558],[14117.0,45.5452,-73.576451],[21260.0,45.517354,-73.582129],[2392.0,45.502602,-73.527503],[2025.0,45.456365,-73.57614],[508.0,45.528446,-73.618702],[1735.0,45.580315,-73.535314],[6048.0,45.533851,-73.578912],[8460.0,45.550316,-73.573358],[71.0,45.5864,-73.595218],[11765.0,45.501715,-73.574131],[7622.0,45.497011,-73.618575],[12643.0,45.482941,-73.579742],[8771.0,45.493569,-73.568576],[16439.0,45.534185,-73.573589],[12558.0,45.500779,-73.558826],[20097.0,45.53543,-73.5822],[8849.0,45.541586,-73.612481],[3621.0,45.559842,-73.615447],[2968.0,45.456365,-73.57614],[10508.0,45.514385,-73.561732],[2629.0,45.544149,-73.66752],[6120.0,45.551912,-73.579211],[3175.0,45.471743,-73.613924],[4971.0,45.477923,-73.559038],[3189.0,45.537527,-73.575124],[2156.0,45.573541,-73.574923],[8960.0,45.534727,-73.614793],[10644.0,45.50015,-73.56928],[8672.0,45.54475,-73.5949],[3893.0,45.548931,-73.616837],[464.0,45.525931,-73.598839],[6789.0,45.54206,-73.597873],[3507.0,45.551585,-73.599165],[11000.0,45.522413,-73.58413],[12743.0,45.524505,-73.594142],[22400.0,45.52353,-73.55199],[2497.0,45.542868,-73.538224],[4127.0,45.538308,-73.654884],[4229.0,45.5265,-73.545753],[6783.0,45.5028,-73.55917],[782.0,45.537226,-73.495067],[1742.0,45.461685,-73.584039],[11953.0,45.502292,-73.573252],[5088.0,45.485048,-73.571963],[8990.0,45.507662,-73.57886],[10031.0,45.531401,-73.612674],[4345.0,45.484859,-73.571585],[5745.0,45.544377,-73.581018],[9271.0,45.527932,-73.623805],[6639.0,45.462742,-73.565845],[6390.0,45.540944,-73.547142],[7469.0,45.528449,-73.551085],[17743.0,45.51086,-73.54983],[14457.0,45.524628,-73.595811],[21429.0,45.517354,-73.582129],[5811.0,45.530027,-73.563176],[4153.0,45.527363,-73.607723],[2862.0,45.516937,-73.640483],[28106.0,45.515616,-73.575808],[139.0,45.522342,-73.721679],[13110.0,45.527758,-73.576185],[20028.0,45.521342,-73.589419],[3338.0,45.538738,-73.628021],[5557.0,45.463001,-73.571569],[574.0,45.539824,-73.508752],[12365.0,45.525669,-73.593016],[11657.0,45.537259,-73.57917],[12577.0,45.521081,-73.562275],[1181.0,45.433557,-73.684043],[20493.0,45.517,-73.589],[8731.0,45.519396,-73.583227],[6560.0,45.543651,-73.605651],[6077.0,45.471743,-73.613924],[19150.0,45.51533,-73.559148],[11964.0,45.495789,-73.582596],[4434.0,45.48204,-73.574863],[3791.0,45.5195,-73.560269],[29057.0,45.50708,-73.56917],[24450.0,45.53092,-73.57674],[12783.0,45.509759,-73.576107],[2099.0,45.575707,-73.561562],[3952.0,45.456365,-73.57614],[11354.0,45.512587,-73.573739],[18199.0,45.516427,-73.558112],[12595.0,45.492825,-73.557978],[4710.0,45.53133,-73.59155],[1408.0,45.579325,-73.570504],[7495.0,45.547408,-73.607669],[1703.0,45.595875,-73.523528],[13100.0,45.50723,-73.615085],[3422.0,45.4891,-73.57656],[27113.0,45.515299,-73.561273],[8543.0,45.498639,-73.574227],[6545.0,45.543763,-73.56772],[5645.0,45.531834,-73.553478],[7810.0,45.49486,-73.57108],[5003.0,45.533661,-73.552197],[11183.0,45.51889,-73.56353],[2086.0,45.41631,-73.612916],[40525.0,45.50708,-73.56917],[5628.0,45.54918,-73.593858],[9359.0,45.53073,-73.58153],[6531.0,45.533195,-73.601173],[7042.0,45.536178,-73.57641],[3199.0,45.4981,-73.652942],[4236.0,45.513917,-73.560765],[4402.0,45.545604,-73.63474],[5930.0,45.47993,-73.555921],[12132.0,45.520688,-73.608428],[16115.0,45.501929,-73.571451],[13985.0,45.508567,-73.577698],[21.0,45.438367,-73.715467],[4789.0,45.526206,-73.54925],[6990.0,45.535614,-73.566014],[1147.0,45.457597,-73.590529],[9161.0,45.523194,-73.555843],[3203.0,45.518215,-73.603251],[9572.0,45.510086,-73.611429],[35144.0,45.51941,-73.58685],[12048.0,45.514108,-73.57523],[3588.0,45.532013,-73.580541],[7723.0,45.503982,-73.62284],[12612.0,45.522413,-73.58413],[17015.0,45.52729,-73.56463],[4856.0,45.477923,-73.559038],[385.0,45.533703,-73.515283],[32055.0,45.50038,-73.57507],[3187.0,45.550613,-73.582883],[4582.0,45.534453,-73.559652],[980.0,45.450851,-73.594027],[1350.0,45.505973,-73.629593],[6976.0,45.495709,-73.576952],[3409.0,45.527363,-73.607723],[8599.0,45.533924,-73.562425],[9849.0,45.53318,-73.61544],[908.0,45.445272,-73.624484],[11735.0,45.534727,-73.614793],[1789.0,45.496302,-73.629822],[10941.0,45.507078,-73.561395],[4480.0,45.542485,-73.636342],[3422.0,45.513741,-73.538382],[4746.0,45.553194,-73.539814],[6636.0,45.498578,-73.560242],[7779.0,45.538799,-73.5887],[4320.0,45.534358,-73.612231],[1806.0,45.550899,-73.62108],[19993.0,45.519088,-73.569509],[5731.0,45.477923,-73.559038],[1574.0,45.475743,-73.564601],[2653.0,45.456358,-73.64013],[6633.0,45.497974,-73.556141],[7127.0,45.489951,-73.567091],[11538.0,45.529971,-73.563053],[7206.0,45.512734,-73.561141],[1549.0,45.417147,-73.644225],[3554.0,45.475193,-73.580943],[9802.0,45.545776,-73.562175],[4224.0,45.542579,-73.636557],[2003.0,45.557895,-73.576529],[1365.0,45.472503,-73.539285],[5320.0,45.47968,-73.56385],[1493.0,45.555205,-73.555823],[14346.0,45.50501,-73.57069],[2069.0,45.512797,-73.561463],[29483.0,45.51941,-73.58685],[4014.0,45.564353,-73.571244],[5694.0,45.53217,-73.55859],[2019.0,45.589175,-73.539656],[11098.0,45.514403,-73.575082],[8362.0,45.501974,-73.555157],[5587.0,45.502326,-73.565138],[7535.0,45.532977,-73.581222],[13329.0,45.497411,-73.578271],[1982.0,45.551582,-73.599156],[16867.0,45.534136,-73.595478],[12715.0,45.492897,-73.580294],[18938.0,45.493718,-73.579186],[4535.0,45.464877,-73.626597],[5574.0,45.542972,-73.617939],[9788.0,45.540138,-73.614028],[9562.0,45.547161,-73.543773],[7353.0,45.497974,-73.556141],[3086.0,45.457597,-73.590529],[13237.0,45.497718,-73.56818],[5763.0,45.551192,-73.540872],[4734.0,45.510042,-73.624835],[4734.0,45.553215,-73.58752],[28.0,45.551886,-73.74444],[6527.0,45.5478,-73.584867],[4924.0,45.52974,-73.59527],[7337.0,45.509229,-73.55447],[4777.0,45.541198,-73.590943],[2873.0,45.444433,-73.575788],[3204.0,45.557192,-73.569847],[6988.0,45.527041,-73.593471],[1562.0,45.544594,-73.537795],[5864.0,45.493034,-73.583836],[4173.0,45.531611,-73.61335],[10087.0,45.510351,-73.556508],[6061.0,45.544846,-73.610536],[25674.0,45.51066,-73.56497],[4438.0,45.553571,-73.560494],[19165.0,45.536785,-73.614888],[4591.0,45.514087,-73.5523],[27494.0,45.527432,-73.579917],[1612.0,45.573733,-73.539446],[7176.0,45.54717,-73.618041],[11149.0,45.501715,-73.574131],[534.0,45.573798,-73.550345],[908.0,45.58116,-73.641243],[6638.0,45.53318,-73.61544],[20402.0,45.52689,-73.57264],[18410.0,45.532265,-73.611063],[1848.0,45.568225,-73.579436],[5641.0,45.533348,-73.605834],[5132.0,45.536299,-73.607773],[1115.0,45.508979,-73.698564],[3166.0,45.550056,-73.534576],[16271.0,45.51908,-73.5727],[10409.0,45.537114,-73.571003],[2004.0,45.571798,-73.54155],[15838.0,45.51674,-73.577703],[5384.0,45.501715,-73.57413],[4177.0,45.53133,-73.59155],[19482.0,45.492825,-73.557978],[3998.0,45.546939,-73.602616],[2515.0,45.557895,-73.576529],[2947.0,45.55325,-73.54822],[2955.0,45.466965,-73.631664],[2958.0,45.550684,-73.615293],[688.0,45.472503,-73.539285],[2551.0,45.552649,-73.581185],[3537.0,45.506373,-73.524577],[4654.0,45.49486,-73.57108],[7081.0,45.52089,-73.559221],[8191.0,45.529134,-73.559234],[4015.0,45.459707,-73.571438],[18007.0,45.525931,-73.598839],[10049.0,45.503805,-73.568179],[8972.0,45.555004,-73.555355],[6143.0,45.4891,-73.57656],[5862.0,45.520188,-73.590559],[8259.0,45.532926,-73.56378],[9563.0,45.535614,-73.566014],[8889.0,45.532008,-73.580444],[5097.0,45.53867,-73.56936],[7146.0,45.53703,-73.58584],[892.0,45.595352,-73.55859],[9856.0,45.5452,-73.576451],[12528.0,45.50206,-73.56295],[11648.0,45.531401,-73.612674],[9414.0,45.487531,-73.56507],[4710.0,45.485385,-73.628142],[2408.0,45.472503,-73.539285],[8428.0,45.508205,-73.568814],[11115.0,45.51829,-73.59235],[12546.0,45.53543,-73.5822],[20298.0,45.527201,-73.564729],[24618.0,45.489476,-73.584566],[9394.0,45.479483,-73.61979],[1941.0,45.523575,-73.623443],[2876.0,45.470544,-73.63168],[2416.0,45.567766,-73.579688],[4362.0,45.466569,-73.623295],[22784.0,45.521116,-73.594784],[2105.0,45.550692,-73.656367],[939.0,45.576,-73.646765],[2972.0,45.453319,-73.576775],[1459.0,45.453246,-73.576815],[22909.0,45.51533,-73.559148],[4303.0,45.514379,-73.610871],[7819.0,45.523035,-73.565406],[2262.0,45.539632,-73.634448],[15530.0,45.508144,-73.574772],[19734.0,45.518967,-73.583616],[4043.0,45.545459,-73.556669],[10772.0,45.4952,-73.56328],[2956.0,45.564944,-73.577639],[6150.0,45.494872,-73.571124],[20790.0,45.4918,-73.584006],[13295.0,45.531674,-73.565413],[1336.0,45.485631,-73.628765],[4128.0,45.54918,-73.593858],[1691.0,45.501302,-73.633161],[18205.0,45.527009,-73.585766],[11649.0,45.459488,-73.572092],[11202.0,45.521094,-73.562251],[7790.0,45.536261,-73.607546],[18051.0,45.53092,-73.57674],[4311.0,45.489545,-73.564139],[6905.0,45.531818,-73.565317],[6548.0,45.527727,-73.586284],[18074.0,45.516876,-73.57946],[7566.0,45.525021,-73.610737],[11594.0,45.484018,-73.563437],[12438.0,45.509685,-73.562317],[8600.0,45.536582,-73.616225],[10376.0,45.544284,-73.54534],[11784.0,45.546907,-73.551557],[6574.0,45.479483,-73.61979],[11279.0,45.506876,-73.623592],[2549.0,45.553579,-73.662083],[7378.0,45.499326,-73.571768],[5738.0,45.489609,-73.582445],[3433.0,45.480134,-73.585908],[11050.0,45.518253,-73.588338],[8216.0,45.534882,-73.609412],[14913.0,45.527616,-73.589115],[1766.0,45.491513,-73.633649],[14216.0,45.486971,-73.589293],[12032.0,45.524566,-73.572079],[12061.0,45.498673,-73.552563],[5983.0,45.52974,-73.59527],[7747.0,45.536261,-73.607545],[14468.0,45.4952,-73.56328],[2970.0,45.545542,-73.634748],[4934.0,45.537813,-73.624872],[12143.0,45.506194,-73.569968],[2300.0,45.53057,-73.54913],[15625.0,45.528746,-73.578654],[5817.0,45.53217,-73.55859],[4658.0,45.545955,-73.626992],[1316.0,45.57179,-73.564315],[11882.0,45.515288,-73.581249],[11863.0,45.493718,-73.579186],[3271.0,45.564353,-73.571244],[16758.0,45.529559,-73.587602],[4932.0,45.526235,-73.54895],[469.0,45.516181,-73.608543],[4284.0,45.557192,-73.569847],[2732.0,45.534795,-73.609546],[8951.0,45.497697,-73.568646],[4191.0,45.514087,-73.5523],[13988.0,45.50206,-73.56295],[4263.0,45.551585,-73.599165],[2334.0,45.460729,-73.634073],[6933.0,45.532408,-73.564776],[2620.0,45.557771,-73.584138],[4195.0,45.540396,-73.616818],[2415.0,45.492227,-73.584782],[16726.0,45.508144,-73.574772],[14453.0,45.486971,-73.589293],[3333.0,45.4743,-73.6237],[6649.0,45.531164,-73.583695],[3401.0,45.553215,-73.58752],[2958.0,45.471882,-73.640022],[3274.0,45.450837,-73.572446],[6836.0,45.540944,-73.547142],[18497.0,45.532994,-73.584414],[1013.0,45.556562,-73.670115],[3673.0,45.518309,-73.603286],[19069.0,45.527009,-73.585766],[24647.0,45.515616,-73.575808],[5957.0,45.532408,-73.564776],[1536.0,45.450851,-73.594027],[9141.0,45.553879,-73.551761],[9407.0,45.51754,-73.597832],[12832.0,45.50206,-73.56295],[3479.0,45.515384,-73.606408],[11922.0,45.547405,-73.544675],[3279.0,45.550784,-73.5341],[12929.0,45.524505,-73.594142],[18225.0,45.529802,-73.57029],[2350.0,45.55197,-73.593775],[3423.0,45.548859,-73.616611],[3937.0,45.470696,-73.565422],[17.0,45.439824,-73.694878],[6437.0,45.555241,-73.550822],[4831.0,45.544846,-73.610536],[11900.0,45.517333,-73.574436],[10281.0,45.54178,-73.555714],[7802.0,45.528366,-73.551171],[5183.0,45.495323,-73.573887],[2695.0,45.538799,-73.552496],[4978.0,45.537964,-73.562643],[1580.0,45.547648,-73.665566],[6563.0,45.496563,-73.564917],[28148.0,45.50038,-73.57507],[686.0,45.514995,-73.559169],[8070.0,45.498639,-73.574227],[23651.0,45.513303,-73.572961],[6791.0,45.470303,-73.589848],[9115.0,45.470696,-73.565422],[2201.0,45.456358,-73.64013],[4588.0,45.461078,-73.567307],[8788.0,45.547304,-73.543827],[23465.0,45.512541,-73.570677],[8514.0,45.532132,-73.55876],[4775.0,45.456085,-73.581937],[10230.0,45.539932,-73.599343],[3668.0,45.456049,-73.571877],[7973.0,45.537159,-73.597476],[3068.0,45.523319,-73.520127],[6558.0,45.546661,-73.588684],[12254.0,45.520018,-73.579463],[2041.0,45.519505,-73.598529],[7571.0,45.484896,-73.560119],[4756.0,45.521069,-73.563757],[3263.0,45.546661,-73.588684],[22235.0,45.51066,-73.56497],[15213.0,45.53229,-73.57544],[6170.0,45.5028,-73.55917],[1614.0,45.530221,-73.576997],[8504.0,45.495789,-73.582596],[1937.0,45.561396,-73.601544],[6539.0,45.483006,-73.562187],[5313.0,45.54816,-73.592026],[8324.0,45.537891,-73.618215],[4106.0,45.538738,-73.628021],[1835.0,45.596796,-73.535179],[3259.0,45.464877,-73.626597],[11881.0,45.505359,-73.567741],[3070.0,45.547227,-73.56987],[12873.0,45.5219,-73.55534],[20498.0,45.513303,-73.572961],[19148.0,45.519157,-73.55786],[4902.0,45.478968,-73.619599],[5335.0,45.515616,-73.575808],[7138.0,45.531939,-73.553439],[16625.0,45.518967,-73.583616],[12925.0,45.530816,-73.581626],[7242.0,45.489609,-73.582445],[4115.0,45.482779,-73.584525],[4486.0,45.515634,-73.606711],[5720.0,45.493034,-73.583836],[2626.0,45.553914,-73.571419],[10356.0,45.524683,-73.578897],[4670.0,45.429471,-73.593099],[8285.0,45.509328,-73.554347],[9407.0,45.53227,-73.56828],[14700.0,45.537114,-73.571003],[7399.0,45.553879,-73.551761],[4966.0,45.527839,-73.594821],[3941.0,45.549833,-73.640633],[4889.0,45.505063,-73.570681],[7868.0,45.522878,-73.577495],[7204.0,45.50623,-73.55976],[11941.0,45.50206,-73.56295],[15789.0,45.52479,-73.56545],[5048.0,45.46714,-73.54259],[6219.0,45.534298,-73.612172],[16721.0,45.492504,-73.581088],[8836.0,45.521156,-73.585589],[3692.0,45.470303,-73.589848],[2656.0,45.539679,-73.634588],[19413.0,45.524286,-73.604973],[2046.0,45.564353,-73.571244],[5063.0,45.474138,-73.603597],[2231.0,45.433251,-73.693962],[7868.0,45.53696,-73.61199],[3865.0,45.534377,-73.612278],[8196.0,45.515172,-73.581002],[7935.0,45.520018,-73.579463],[3747.0,45.550476,-73.590446],[9735.0,45.52951,-73.61068],[3588.0,45.551585,-73.599165],[3126.0,45.515617,-73.614991],[1939.0,45.552649,-73.581185],[31549.0,45.532514,-73.584811],[9508.0,45.546978,-73.575515],[2016.0,45.554564,-73.596157],[1346.0,45.587531,-73.531907],[404.0,45.494514,-73.583368],[4422.0,45.536856,-73.579373],[12330.0,45.504407,-73.572543],[25305.0,45.507629,-73.551876],[5478.0,45.521565,-73.57043],[1099.0,45.456365,-73.57614],[10768.0,45.471669,-73.582644],[14693.0,45.5452,-73.576451],[4488.0,45.544377,-73.581018],[1333.0,45.552123,-73.630103],[8588.0,45.531401,-73.612674],[1699.0,45.549933,-73.647645],[2117.0,45.559482,-73.535496],[6279.0,45.519376,-73.551951],[17633.0,45.513338,-73.57295],[11132.0,45.480208,-73.577599],[8263.0,45.506421,-73.556466],[11411.0,45.50623,-73.55976],[3371.0,45.516783,-73.611423],[19777.0,45.52353,-73.55199],[28627.0,45.515299,-73.561273],[10813.0,45.524683,-73.578897],[20603.0,45.520458,-73.567575],[9785.0,45.491754,-73.583972],[13682.0,45.537026,-73.593166],[2462.0,45.559199,-73.599658],[2930.0,45.550698,-73.60953],[10472.0,45.516873,-73.554175],[3534.0,45.557487,-73.588749],[9679.0,45.521485,-73.58498],[12070.0,45.543712,-73.628284],[11154.0,45.512601,-73.57374],[7570.0,45.494514,-73.583368],[2666.0,45.55325,-73.54822],[11104.0,45.547558,-73.575472],[1186.0,45.570081,-73.573047],[9992.0,45.52457,-73.59588],[25937.0,45.50206,-73.56295],[10006.0,45.521156,-73.585589],[43811.0,45.524673,-73.58255],[1290.0,45.58077,-73.544766],[7232.0,45.462846,-73.565922],[14522.0,45.527513,-73.598791],[5046.0,45.554133,-73.560226],[20322.0,45.528758,-73.57868],[2781.0,45.569789,-73.548081],[7905.0,45.538403,-73.561964],[7652.0,45.484896,-73.560119],[8036.0,45.549457,-73.541649],[5976.0,45.457509,-73.639485],[12884.0,45.537114,-73.571003],[2924.0,45.556751,-73.667162],[1915.0,45.43284,-73.693492],[4291.0,45.479483,-73.61979],[1023.0,45.459593,-73.544377],[7621.0,45.50931,-73.554431],[19770.0,45.532265,-73.611063],[3105.0,45.544136,-73.545609],[688.0,45.556708,-73.670634],[8752.0,45.489951,-73.567091],[84.0,45.438056,-73.672739],[14598.0,45.53248,-73.58507],[4046.0,45.475142,-73.623505],[4642.0,45.52974,-73.59527],[21763.0,45.518593,-73.581566],[8792.0,45.50015,-73.56928],[7143.0,45.522586,-73.612658],[8415.0,45.54178,-73.555714],[12776.0,45.531917,-73.597983],[2919.0,45.462362,-73.59665],[3776.0,45.557921,-73.529147],[8849.0,45.519895,-73.596827],[2461.0,45.559199,-73.599658],[1190.0,45.526856,-73.626672],[10900.0,45.503738,-73.568106],[4518.0,45.55224,-73.544127],[2084.0,45.570657,-73.651842],[13268.0,45.484081,-73.560918],[19574.0,45.52729,-73.56463],[6337.0,45.556262,-73.537441],[1885.0,45.472503,-73.539285],[4133.0,45.553914,-73.571419],[7001.0,45.523194,-73.555843],[8091.0,45.502292,-73.573252],[19898.0,45.524505,-73.594142],[9123.0,45.533548,-73.566867],[8904.0,45.54006,-73.60897],[9434.0,45.522233,-73.602082],[7640.0,45.541435,-73.554314],[2084.0,45.482693,-73.634813],[11688.0,45.509569,-73.564137],[7465.0,45.50723,-73.615085],[9160.0,45.525048,-73.560036],[3624.0,45.546406,-73.598421],[7547.0,45.529802,-73.591766],[8909.0,45.509864,-73.576105],[8152.0,45.516897,-73.563949],[2464.0,45.494564,-73.638702],[3002.0,45.486319,-73.570233],[8087.0,45.462726,-73.565959],[3105.0,45.589175,-73.539656],[11303.0,45.498639,-73.574227],[15130.0,45.534134,-73.573492],[6825.0,45.497011,-73.618575],[10905.0,45.5341,-73.58852],[1729.0,45.523854,-73.519677],[2625.0,45.548931,-73.616837],[24386.0,45.497515,-73.552571],[7259.0,45.5265,-73.545753],[2498.0,45.513741,-73.538382],[23969.0,45.51561,-73.57569],[4527.0,45.542983,-73.606381],[5602.0,45.51164,-73.562184],[10071.0,45.524566,-73.572079],[10022.0,45.489951,-73.567091],[17339.0,45.522365,-73.584146],[15422.0,45.49975,-73.55566],[1808.0,45.544594,-73.537795],[12276.0,45.529971,-73.563053],[15472.0,45.528746,-73.578654],[16959.0,45.50781,-73.57208],[17884.0,45.517,-73.589],[3296.0,45.486452,-73.595234],[5543.0,45.478889,-73.581989],[2602.0,45.453319,-73.576775],[4090.0,45.515007,-73.619788],[3398.0,45.486319,-73.570233],[4138.0,45.541955,-73.629942],[2986.0,45.549126,-73.600912],[11412.0,45.538984,-73.582445],[2466.0,45.518215,-73.603251],[1566.0,45.556591,-73.652676],[10039.0,45.548186,-73.59201],[18990.0,45.5079,-73.563112],[5696.0,45.52029,-73.568119],[20066.0,45.527009,-73.585766],[12355.0,45.501715,-73.574131],[9853.0,45.536123,-73.622445],[5904.0,45.51164,-73.562184],[4938.0,45.545581,-73.6349],[6411.0,45.559482,-73.535496],[8104.0,45.531401,-73.612674],[460.0,45.6175,-73.606011],[10798.0,45.500234,-73.571126],[6417.0,45.495894,-73.560097],[91.0,45.436568,-73.70837],[4177.0,45.557192,-73.569847],[12788.0,45.500674,-73.561082],[17591.0,45.537088,-73.593219],[5347.0,45.505312,-73.560891],[15470.0,45.52697,-73.60261],[3708.0,45.479483,-73.61979],[4661.0,45.559482,-73.535496],[7580.0,45.542119,-73.622547],[68.0,45.593985,-73.637672],[5673.0,45.553215,-73.58752],[1759.0,45.549933,-73.647645],[7378.0,45.517459,-73.597927],[3059.0,45.540686,-73.653376],[11777.0,45.524518,-73.571977],[14515.0,45.502813,-73.576024],[14863.0,45.521293,-73.589365],[4981.0,45.561594,-73.585643],[24274.0,45.51066,-73.56497],[5487.0,45.528088,-73.585863],[1163.0,45.5536,-73.6616],[17226.0,45.494981,-73.577733],[762.0,45.536378,-73.512642],[7408.0,45.525983,-73.577808],[14758.0,45.504095,-73.568696],[2588.0,45.477923,-73.559038],[1008.0,45.58116,-73.641243],[4492.0,45.4829,-73.591],[12284.0,45.520604,-73.575984],[2256.0,45.579349,-73.581681],[4001.0,45.468927,-73.61987],[119.0,45.576259,-73.538312],[25123.0,45.51059,-73.57547],[6913.0,45.531818,-73.565317],[3309.0,45.559199,-73.599658],[8229.0,45.557192,-73.569847],[5989.0,45.546797,-73.602664],[11431.0,45.501041,-73.577178],[8114.0,45.53867,-73.56936],[1688.0,45.563921,-73.548912],[2687.0,45.422451,-73.601194],[8734.0,45.553219,-73.539782],[5956.0,45.522586,-73.612658],[33056.0,45.51735,-73.56906],[4218.0,45.46714,-73.54259],[24252.0,45.513303,-73.572961],[23907.0,45.510072,-73.570752],[6199.0,45.508356,-73.580498],[13732.0,45.51069,-73.57805],[9937.0,45.502786,-73.559134],[1086.0,45.444433,-73.575788],[11862.0,45.501402,-73.571814],[38.0,45.558109,-73.719597],[17624.0,45.52353,-73.55199],[7464.0,45.538461,-73.577679],[5219.0,45.504591,-73.559188],[8789.0,45.50235,-73.566583],[9064.0,45.541586,-73.612481],[15281.0,45.516818,-73.554188],[16835.0,45.554214,-73.55156],[2097.0,45.554862,-73.532717],[9722.0,45.528724,-73.583859],[31745.0,45.506448,-73.576349],[4456.0,45.542485,-73.636342],[692.0,45.559496,-73.698918],[13555.0,45.50971,-73.57354],[7047.0,45.493034,-73.583836],[7173.0,45.539932,-73.599343],[1757.0,45.564944,-73.577639],[5076.0,45.54523,-73.53725],[1697.0,45.560762,-73.650561],[1284.0,45.45843,-73.59606],[10755.0,45.510351,-73.556508],[14588.0,45.498151,-73.577609],[12499.0,45.504245,-73.553545],[1838.0,45.459233,-73.637405],[4315.0,45.553491,-73.54829],[242.0,45.524277,-73.58929],[4251.0,45.539224,-73.573837],[25113.0,45.504242,-73.553469],[409.0,45.514335,-73.556464],[9872.0,45.531939,-73.553439],[5685.0,45.531939,-73.553439],[2936.0,45.553262,-73.638615],[29810.0,45.51735,-73.56906],[10799.0,45.514726,-73.565486],[7277.0,45.53867,-73.56936],[6437.0,45.49606,-73.57348],[13153.0,45.50623,-73.55976],[25566.0,45.506448,-73.576349],[10515.0,45.499677,-73.578849],[23841.0,45.52353,-73.55199],[9626.0,45.527748,-73.597439],[6725.0,45.477429,-73.600169],[9956.0,45.519515,-73.598517],[11863.0,45.516873,-73.554175],[2752.0,45.51405,-73.647256],[6200.0,45.480495,-73.577827],[15883.0,45.49947,-73.57591],[14870.0,45.524037,-73.590143],[10897.0,45.496457,-73.620155],[16641.0,45.521564,-73.570367],[12188.0,45.523525,-73.587509],[2168.0,45.537628,-73.553394],[3274.0,45.523575,-73.623443],[25258.0,45.50761,-73.551836],[6682.0,45.529686,-73.601728],[17426.0,45.52479,-73.56545],[4317.0,45.510042,-73.624835],[8721.0,45.50449,-73.5592],[9523.0,45.509229,-73.55447],[1976.0,45.53057,-73.54913],[5067.0,45.5499,-73.583084],[9246.0,45.507078,-73.561395],[2421.0,45.521769,-73.534859],[4254.0,45.486919,-73.569429],[4560.0,45.54078,-73.630401],[21275.0,45.519146,-73.569241],[5560.0,45.54458,-73.588069],[5247.0,45.52765,-73.6123],[7994.0,45.549911,-73.558263],[10403.0,45.532008,-73.580444],[6419.0,45.507838,-73.563136],[5291.0,45.500234,-73.571126],[9757.0,45.522278,-73.577591],[1978.0,45.567352,-73.653787],[6980.0,45.487601,-73.565185],[2114.0,45.531164,-73.583695],[10466.0,45.477429,-73.600169],[11631.0,45.554214,-73.55156],[6968.0,45.505382,-73.567634],[162.0,45.533496,-73.566931],[4186.0,45.518309,-73.603286],[9100.0,45.512614,-73.573726],[4090.0,45.45073,-73.572575],[4980.0,45.519132,-73.55789],[9175.0,45.499745,-73.579034],[17590.0,45.520541,-73.567751],[11781.0,45.51569,-73.560966],[38875.0,45.50708,-73.56917],[14490.0,45.474224,-73.603721],[2511.0,45.43155,-73.67112],[8902.0,45.495581,-73.553711],[14005.0,45.526712,-73.584838],[21137.0,45.53543,-73.5822],[6915.0,45.523172,-73.618097],[4849.0,45.453016,-73.571915],[2692.0,45.554931,-73.610711],[2476.0,45.545425,-73.556527],[12350.0,45.513409,-73.533554],[4396.0,45.539857,-73.620383],[496.0,45.53308,-73.694597],[3494.0,45.5427,-73.59212],[12832.0,45.51791,-73.567143],[1378.0,45.512283,-73.561826],[10588.0,45.539032,-73.614318],[551.0,45.526765,-73.730423],[13122.0,45.520178,-73.579424],[6779.0,45.5306,-73.614616],[2.0,45.503215,-73.569979],[4625.0,45.494514,-73.583368],[4287.0,45.521565,-73.535332],[953.0,45.53308,-73.694597],[12616.0,45.513917,-73.560765],[5107.0,45.52114,-73.54926],[3075.0,45.564353,-73.571244],[12941.0,45.505314,-73.567554],[1869.0,45.559,-73.658414],[9746.0,45.477249,-73.587238],[7533.0,45.546408,-73.638428],[5897.0,45.552364,-73.602868],[6943.0,45.503727,-73.561704],[22100.0,45.512541,-73.570677],[4478.0,45.44691,-73.60363],[5873.0,45.532013,-73.580541],[301.0,45.520019,-73.618907],[5184.0,45.511119,-73.567974],[5528.0,45.4816,-73.6],[2208.0,45.48047,-73.611216],[1054.0,45.539726,-73.677165],[21024.0,45.516664,-73.57722],[12197.0,45.492463,-73.560784],[1802.0,45.538799,-73.552496],[4351.0,45.483928,-73.577312],[1099.0,45.483221,-73.580764],[16796.0,45.503475,-73.572103],[9276.0,45.490042,-73.567111],[13015.0,45.529936,-73.604096],[14291.0,45.512014,-73.532547],[10050.0,45.499237,-73.569442],[4116.0,45.542485,-73.636342],[8333.0,45.531401,-73.612674],[10281.0,45.53073,-73.58153],[8341.0,45.51889,-73.56353],[1752.0,45.561594,-73.585643],[1239.0,45.523413,-73.623526],[3232.0,45.559872,-73.615541],[4575.0,45.4816,-73.6],[1853.0,45.515617,-73.614991],[9697.0,45.532674,-73.571776],[3324.0,45.549755,-73.605566],[18983.0,45.52689,-73.57264],[2664.0,45.513413,-73.538253],[3296.0,45.550769,-73.57711],[19161.0,45.500951,-73.574578],[21386.0,45.506511,-73.576669],[12555.0,45.51069,-73.57805],[13458.0,45.500606,-73.565448],[7839.0,45.525669,-73.593016],[322.0,45.529512,-73.517691],[9235.0,45.499677,-73.578849],[8668.0,45.496583,-73.576502],[7151.0,45.467369,-73.570769],[11071.0,45.523584,-73.560941],[6865.0,45.531475,-73.592421],[8734.0,45.503727,-73.561704],[5080.0,45.53703,-73.58584],[2685.0,45.48983,-73.589678],[5179.0,45.502326,-73.565138],[189.0,45.588082,-73.632223],[6434.0,45.50931,-73.554431],[7997.0,45.539932,-73.599343],[12250.0,45.537114,-73.571003],[2456.0,45.450851,-73.594027],[2666.0,45.4743,-73.6237],[7150.0,45.522586,-73.612658],[11428.0,45.51069,-73.57805],[2236.0,45.463257,-73.575621],[141.0,45.549783,-73.70631],[26150.0,45.53092,-73.57674],[4122.0,45.499757,-73.629298],[5920.0,45.536178,-73.57641],[6385.0,45.557895,-73.576529],[15045.0,45.50176,-73.57128],[6520.0,45.53217,-73.55859],[4625.0,45.50076,-73.570483],[3071.0,45.549833,-73.640633],[7048.0,45.518263,-73.595775],[8595.0,45.530351,-73.624392],[15747.0,45.520578,-73.567733],[4973.0,45.523276,-73.558218],[10316.0,45.50781,-73.57208],[13132.0,45.502294,-73.573219],[7201.0,45.481291,-73.60033],[4008.0,45.53923,-73.541082],[6856.0,45.491322,-73.58781],[4068.0,45.544828,-73.610458],[234.0,45.494499,-73.574173],[15327.0,45.522365,-73.584146],[10157.0,45.523525,-73.587509],[2830.0,45.543699,-73.553673],[11665.0,45.526543,-73.598233],[14553.0,45.529693,-73.567401],[21242.0,45.522278,-73.577591],[14991.0,45.522426,-73.574184],[4915.0,45.52746,-73.624175],[3808.0,45.460729,-73.634073],[7195.0,45.481291,-73.60033],[5365.0,45.475743,-73.564601],[4803.0,45.546539,-73.598665],[2829.0,45.550056,-73.534576],[10034.0,45.544311,-73.545549],[8966.0,45.539032,-73.614318],[1956.0,45.4688,-73.6198],[2985.0,45.521565,-73.535332],[11339.0,45.523507,-73.576413],[4830.0,45.52974,-73.59527],[2307.0,45.486279,-73.573599],[7845.0,45.539385,-73.541],[16002.0,45.492504,-73.581088],[1401.0,45.472503,-73.539285],[5888.0,45.5367,-73.56081],[13060.0,45.529802,-73.57029],[8898.0,45.477605,-73.573775],[592.0,45.514734,-73.691449],[9339.0,45.546978,-73.575515],[427.0,45.533682,-73.515261],[3452.0,45.549695,-73.554647],[1514.0,45.569789,-73.548081],[1401.0,45.563947,-73.655573],[24924.0,45.52353,-73.55199],[9026.0,45.511638,-73.574635],[1606.0,45.485889,-73.577514],[10414.0,45.487723,-73.569083],[568.0,45.526235,-73.54895],[22279.0,45.529559,-73.587602],[2681.0,45.520222,-73.629856],[21201.0,45.498112,-73.577615],[7994.0,45.521685,-73.574942],[3339.0,45.553408,-73.569531],[11745.0,45.505377,-73.56762],[9492.0,45.527738,-73.576249],[20095.0,45.478796,-73.576424],[20861.0,45.49803,-73.552665],[2583.0,45.508439,-73.672965],[11421.0,45.53045,-73.587638],[609.0,45.539824,-73.508752],[6162.0,45.540396,-73.616818],[6483.0,45.500208,-73.571138],[8113.0,45.544599,-73.588223],[12440.0,45.559018,-73.548481],[3567.0,45.552123,-73.630103],[1474.0,45.568225,-73.579436],[5335.0,45.505386,-73.570432],[13101.0,45.507717,-73.57167],[4476.0,45.552123,-73.630103],[4097.0,45.550768,-73.549427],[9331.0,45.529326,-73.574806],[3480.0,45.564944,-73.577639],[3345.0,45.505973,-73.629593],[975.0,45.506445,-73.556352],[2369.0,45.512861,-73.561595],[4346.0,45.551584,-73.561916],[7154.0,45.508356,-73.580498],[1284.0,45.560742,-73.65041],[11450.0,45.50723,-73.615085],[12300.0,45.51069,-73.57805],[5177.0,45.50233,-73.566497],[8062.0,45.537026,-73.593166],[3004.0,45.558481,-73.583304],[46.0,45.588356,-73.631712],[12962.0,45.54006,-73.60897],[9401.0,45.533409,-73.570657],[982.0,45.433124,-73.681405],[16778.0,45.514565,-73.568407],[10586.0,45.523492,-73.56073],[6807.0,45.481291,-73.60033],[8684.0,45.519957,-73.554817],[3734.0,45.549714,-73.563848],[1923.0,45.494564,-73.638702],[8609.0,45.500035,-73.618105],[2197.0,45.572789,-73.540313],[3783.0,45.553215,-73.569496],[6263.0,45.504263,-73.571523],[19683.0,45.503738,-73.568106],[6214.0,45.478888,-73.581989],[11875.0,45.47393,-73.604735],[1991.0,45.494564,-73.638702],[4778.0,45.491722,-73.57689],[8373.0,45.527231,-73.586726],[3307.0,45.596797,-73.535179],[10879.0,45.505403,-73.621166],[14209.0,45.526252,-73.589758],[12768.0,45.508144,-73.574772],[6061.0,45.5262,-73.549266],[10019.0,45.529219,-73.574777],[4033.0,45.46982,-73.616961],[6489.0,45.541247,-73.591045],[13558.0,45.526557,-73.598276],[12314.0,45.520188,-73.590559],[11049.0,45.507402,-73.578444],[14986.0,45.518253,-73.588338],[5681.0,45.52114,-73.54926],[9109.0,45.524634,-73.578755],[4901.0,45.485048,-73.571963],[8925.0,45.524634,-73.578755],[4508.0,45.49486,-73.57108],[1291.0,45.434434,-73.586694],[7291.0,45.506251,-73.571391],[4068.0,45.539833,-73.575559],[3955.0,45.547362,-73.579221],[5845.0,45.531412,-73.591749],[2995.0,45.506373,-73.524577],[884.0,45.56169,-73.610512],[13888.0,45.523544,-73.587332],[33353.0,45.49659,-73.57851],[6492.0,45.537114,-73.571003],[4007.0,45.4891,-73.57656],[11030.0,45.474448,-73.604031],[30565.0,45.489476,-73.584566],[3588.0,45.545604,-73.63474],[6871.0,45.55287,-73.565631],[2529.0,45.561059,-73.603967],[8397.0,45.538984,-73.582445],[5165.0,45.507144,-73.555119],[1810.0,45.512933,-73.63389],[26113.0,45.49803,-73.552665],[2711.0,45.556905,-73.58925],[17604.0,45.521293,-73.589365],[14007.0,45.500674,-73.561082],[12599.0,45.52056,-73.58582],[12372.0,45.498639,-73.574227],[9635.0,45.541586,-73.612481],[3337.0,45.546406,-73.598421],[3325.0,45.526059,-73.612969],[2975.0,45.456355,-73.597557],[3405.0,45.437914,-73.58274],[4868.0,45.542972,-73.617939],[9705.0,45.484018,-73.563437],[12158.0,45.530816,-73.581626],[648.0,45.519492,-73.685104],[6995.0,45.539932,-73.599343],[9817.0,45.50297,-73.57505],[11605.0,45.53611,-73.576287],[9518.0,45.53341,-73.601124],[19172.0,45.519216,-73.57718],[791.0,45.524277,-73.58929],[11131.0,45.51889,-73.56353],[21490.0,45.50501,-73.57069],[7312.0,45.50449,-73.5592],[1736.0,45.592803,-73.510466],[7553.0,45.497974,-73.556141],[37679.0,45.497515,-73.552571],[12458.0,45.517401,-73.574605],[2180.0,45.558481,-73.583304],[602.0,45.536907,-73.511914],[13510.0,45.493029,-73.564818],[2687.0,45.553215,-73.58752],[2101.0,45.484489,-73.628834],[3068.0,45.515384,-73.606408],[12348.0,45.524505,-73.594142],[53.0,45.442397,-73.71765],[10771.0,45.52163,-73.562348],[3300.0,45.551585,-73.599165],[2883.0,45.463001,-73.571569],[17537.0,45.4918,-73.584006],[12401.0,45.529693,-73.567401],[4742.0,45.520604,-73.575984],[2411.0,45.56244,-73.595333],[2510.0,45.531834,-73.553478],[17765.0,45.517,-73.589],[6677.0,45.53867,-73.56936],[7106.0,45.531401,-73.612674],[12002.0,45.510086,-73.611429],[796.0,45.590892,-73.640617],[15198.0,45.50971,-73.57354],[15502.0,45.524296,-73.604847],[5953.0,45.529002,-73.546199],[1169.0,45.491513,-73.633649],[10637.0,45.500675,-73.565354],[3747.0,45.575707,-73.561562],[5477.0,45.502629,-73.52775],[5472.0,45.530505,-73.614708],[13433.0,45.524037,-73.590143],[395.0,45.58601,-73.521838],[2882.0,45.521769,-73.534859],[1152.0,45.561396,-73.601544],[9878.0,45.543583,-73.6284],[5500.0,45.53703,-73.58584],[4203.0,45.547362,-73.579221],[9650.0,45.543583,-73.6284],[4720.0,45.526566,-73.585428],[4358.0,45.471162,-73.627206],[11526.0,45.520634,-73.563906],[3018.0,45.514379,-73.610871],[14077.0,45.520578,-73.567733],[8722.0,45.53227,-73.56828],[10877.0,45.534727,-73.614793],[2815.0,45.539632,-73.634448],[3947.0,45.569789,-73.548081],[1489.0,45.563947,-73.655573],[8000.0,45.487991,-73.569632],[9112.0,45.506251,-73.571391],[10023.0,45.513834,-73.560446],[1480.0,45.563947,-73.655573],[11197.0,45.523171,-73.573431],[5952.0,45.494514,-73.583368],[4219.0,45.486241,-73.595728],[3082.0,45.486452,-73.595234],[3088.0,45.448262,-73.577856],[4387.0,45.466569,-73.623295],[1943.0,45.527727,-73.586284],[18355.0,45.517999,-73.568184],[6183.0,45.543452,-73.60101],[11997.0,45.49947,-73.57591],[4802.0,45.494499,-73.574173],[13521.0,45.523171,-73.573431],[1664.0,45.526904,-73.624331],[3865.0,45.460729,-73.634073],[3397.0,45.520018,-73.579463],[7529.0,45.527727,-73.586284],[2174.0,45.45002,-73.600306],[6672.0,45.499326,-73.571768],[4018.0,45.549868,-73.647657],[7.0,45.531007,-73.598799],[9535.0,45.50723,-73.615085],[11408.0,45.505011,-73.572028],[3605.0,45.4771,-73.6212],[4856.0,45.550692,-73.656367],[5033.0,45.493569,-73.568576],[1846.0,45.521769,-73.534859],[1526.0,45.563679,-73.571605],[2407.0,45.446534,-73.603541],[8400.0,45.549686,-73.591155],[10953.0,45.501041,-73.577178],[15342.0,45.52479,-73.56545],[2757.0,45.514379,-73.610871],[7443.0,45.529993,-73.604214],[7168.0,45.523035,-73.565395],[318.0,45.5534,-73.662255],[896.0,45.571627,-73.592048],[5519.0,45.544867,-73.614841],[2691.0,45.456049,-73.571877],[4421.0,45.556751,-73.667162],[23626.0,45.51941,-73.58685],[22675.0,45.512541,-73.570677],[5431.0,45.545955,-73.626992],[16957.0,45.529802,-73.57029],[1893.0,45.575707,-73.561562],[29736.0,45.50761,-73.551836],[3754.0,45.550769,-73.57711],[2391.0,45.551582,-73.599156],[16660.0,45.50367,-73.61848],[1847.0,45.5536,-73.6616],[9776.0,45.537026,-73.593166],[5127.0,45.541,-73.558263],[9797.0,45.514565,-73.568407],[6421.0,45.532281,-73.610991],[7784.0,45.517333,-73.574436],[9800.0,45.514784,-73.565338],[20015.0,45.510072,-73.570752],[37570.0,45.50038,-73.57507],[7536.0,45.531164,-73.583695],[1063.0,45.476344,-73.606259],[2293.0,45.477313,-73.615327],[13705.0,45.523544,-73.587332],[3796.0,45.527363,-73.607723],[10515.0,45.504407,-73.572543],[11037.0,45.5341,-73.58852],[22232.0,45.515299,-73.561273],[2677.0,45.486219,-73.573747],[2083.0,45.5613,-73.60143],[976.0,45.462362,-73.59665],[17697.0,45.519088,-73.569509],[10580.0,45.532977,-73.581222],[4222.0,45.544513,-73.571564],[760.0,45.506843,-73.681757],[4849.0,45.536299,-73.607773],[14239.0,45.514887,-73.578276],[8057.0,45.5215,-73.5851],[7925.0,45.539236,-73.585946],[973.0,45.558109,-73.719598],[10602.0,45.524037,-73.590143],[6439.0,45.496851,-73.62332],[16927.0,45.524296,-73.604847],[168.0,45.651406,-73.500413],[2871.0,45.550692,-73.656367],[797.0,45.587778,-73.569354],[13251.0,45.516777,-73.577784],[3641.0,45.552443,-73.603002],[12648.0,45.500685,-73.572156],[8284.0,45.497595,-73.555237],[2453.0,45.51378,-73.632463],[13451.0,45.544284,-73.54534],[15552.0,45.509361,-73.573864],[1969.0,45.494044,-73.654495],[2904.0,45.5367,-73.56081],[1093.0,45.569176,-73.636293],[11316.0,45.520295,-73.614824],[4131.0,45.466936,-73.631531],[17773.0,45.524628,-73.595811],[3765.0,45.505721,-73.629459],[5198.0,45.543763,-73.56772],[13275.0,45.53045,-73.587638],[4243.0,45.507144,-73.555119],[8070.0,45.49123,-73.580782],[9616.0,45.525983,-73.577808],[22698.0,45.51908,-73.5727],[4585.0,45.506373,-73.524577],[3940.0,45.472668,-73.58539],[9176.0,45.49975,-73.55566],[1756.0,45.555241,-73.550822],[6873.0,45.47993,-73.555921],[6574.0,45.502326,-73.565138],[4040.0,45.501195,-73.620625],[6390.0,45.467666,-73.593918],[2128.0,45.55001,-73.59576],[5186.0,45.537114,-73.569139],[3577.0,45.559199,-73.599658],[8638.0,45.512734,-73.561141],[14197.0,45.534136,-73.595478],[6538.0,45.541816,-73.626112],[8916.0,45.522719,-73.577204],[11419.0,45.499191,-73.569421],[13144.0,45.520688,-73.608428],[5420.0,45.497165,-73.55933],[8963.0,45.501967,-73.555153],[5173.0,45.471743,-73.613924],[8112.0,45.5219,-73.55534],[109.0,45.56844,-73.675202],[4542.0,45.450851,-73.594027],[271.0,45.498606,-73.569228],[5042.0,45.527698,-73.586253],[2911.0,45.483928,-73.577312],[2800.0,45.578332,-73.655954],[1542.0,45.506373,-73.524577],[11146.0,45.493029,-73.564818],[2386.0,45.550613,-73.582883],[15419.0,45.522403,-73.581177],[12080.0,45.525669,-73.593016],[8627.0,45.539737,-73.614256],[11079.0,45.52951,-73.61068],[10557.0,45.519237,-73.577215],[2710.0,45.467369,-73.570769],[18563.0,45.480208,-73.577599],[6533.0,45.519376,-73.551951],[8452.0,45.534071,-73.562651],[3352.0,45.513413,-73.538253],[24313.0,45.527432,-73.579917]],"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003estn_name=%{text}\u003cbr\u003estn_lat=%{customdata[1]:.3f}\u003cbr\u003estn_lon=%{customdata[2]:.3f}\u003cbr\u003erides=%{customdata[0]:,}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Ottawa \u002f Peel","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Chabot \u002f Everett","Duluth \u002f St-Denis","Jacques-Le Ber \u002f de la Pointe Nord","Lajeunesse \u002f Cr\u00e9mazie","Resther \u002f du Mont-Royal","Palm \u002f St-Remi","Duvernay \u002f Charlevoix","Chambord \u002f Jean-Talon","de Bellechasse \u002f de St-Vallier","Gauthier \u002f Papineau","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Laval \u002f du Mont-Royal","de Bordeaux \u002f Sherbrooke","Gauthier \u002f de Lorimier","Notre-Dame \u002f Jean-d'Estr\u00e9es","Drolet \u002f Beaubien","Hutchison \u002f Prince-Arthur","Parc Jean-Brillant (Swail \u002f Decelles)","Cypress \u002f Peel","Marquette \u002f Villeray","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Montcalm \u002f Ontario","des \u00c9rables \u002f Rachel","Hutchison \u002f Edouard Charles","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","St-Andr\u00e9 \u002f St-Antoine","Chomedey \u002f de Maisonneuve","McGill \u002f des R\u00e9collets","Henri-Julien \u002f de Castelnau","Fullum \u002f de Rouen","Benny \u002f de Monkland","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Guizot \u002f St-Laurent","CHSLD Benjamin-Victor-Rousselot (Dickson \u002f Sherbrooke)","de Repentigny \u002f Sherbrooke","Resther \u002f du Mont-Royal","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Clark \u002f de Li\u00e8ge","M\u00e9tro Jean-Talon (Berri \u002f Jean-Talon)","Dunlop \u002f Van Horne","de Bordeaux \u002f Jean-Talon","Monkland \u002f Girouard","30e avenue \u002f St-Zotique","De Gasp\u00e9 \u002f Villeray","Ste-Croix \u002f Tass\u00e9","St-Dominique \u002f St-Viateur","Laurier \u002f Bordeaux","LaSalle \u002f 67e avenue","LaSalle \u002f S\u00e9n\u00e9cal","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","Rushbrooke \u002f  Caisse","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Bloomfield \u002f Van Horne","Duchesneau \u002f de Marseille","Place du Commerce","de Repentigny \u002f Sherbrooke","Logan \u002f de Champlain","M\u00e9tro Cr\u00e9mazie (Cr\u00e9mazie \u002f Lajeunesse)","McGill \u002f Place d'Youville","Parthenais \u002f Laurier","de Gasp\u00e9 \u002f Marmier","Dorion \u002f Gauthier","St-Charles \u002f St-Sylvestre","Georges-Baril \u002f Fleury","Square Victoria (Viger \u002f du Square-Victoria)","Laurier \u002f de Bordeaux","3e avenue \u002f Dandurand","Cavendish \u002f Poirier","Duluth \u002f St-Laurent","Fabre \u002f Beaubien","Boyer \u002f du Mont-Royal","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Laurier \u002f de Bordeaux","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","Laporte \u002f St-Antoine","des Ormeaux \u002f Notre-Dame","Place Longueuil","Elgar \u002f de l'\u00cele-des-S\u0153urs","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","M\u00e9tro Sauv\u00e9 (Berri \u002f Sauv\u00e9)","Fullum \u002f Gilford","St-Andr\u00e9 \u002f St-Gr\u00e9goire","University \u002f des Pins","Shearer \u002f Centre","Laval \u002f Duluth","Leman \u002f de Chateaubriand","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","Milton \u002f Universit\u00e9","Fabre \u002f Beaubien","Queen \u002f Ottawa","de la Roche \u002f  de Bellechasse","16e avenue \u002f Jean-Talon","Monk \u002f Jacques-Hertel","Bossuet \u002f Pierre-de-Coubertin","Gare d'autocars de Montr\u00e9al (Berri \u002f Ontario)","Larivi\u00e8re \u002f de Lorimier","Ottawa \u002f William","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Crescent \u002f Ren\u00e9-L\u00e9vesque","Cartier \u002f Marie-Anne","Guilbault \u002f Clark","Plessis \u002f Ren\u00e9-L\u00e9vesque","des \u00c9cores \u002f Jean-Talon","4e avenue \u002f de Verdun","Alexandra \u002f Waverly","Lajeunesse \u002f Cr\u00e9mazie","Mackay \u002fde Maisonneuve (Sud)","Murray \u002f William","de Maisonneuve \u002f Robert-Bourassa","Ann \u002f William","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Rachel \u002f Br\u00e9beuf","de la Roche \u002f B\u00e9langer","Faillon \u002f St-Denis","Hillside \u002f Ste-Catherine","St-Andr\u00e9 \u002f Ontario","de Br\u00e9beuf \u002f St-Gr\u00e9goire","Boyer \u002f du Mont-Royal","Square Phillips 2","de l'Esplanade \u002f du Mont-Royal","Jeanne Mance \u002f du Mont-Royal","Drolet \u002f Villeray","M\u00e9tro Viau (Pierre-de-Coubertin \u002f Sicard)","5e avenue \u002f de Verdun","Parc de Dieppe","Ropery \u002f Augustin-Cantin","Gerry-Boulet \u002f St-Gr\u00e9goire","West Broadway \u002f Sherbrooke","Marie-Anne \u002f St-Hubert","de Marseille \u002f Viau","Fran\u00e7ois-Perrault \u002f L.-O.-David","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Lincoln \u002f Lambert Closse","Laval \u002f Duluth","Mentana \u002f Marie-Anne","C\u00f4te St-Antoine \u002f Royal","Beaudry \u002f Ontario","Laurier \u002f 15e avenue","Lucien L'Allier \u002f St-Jacques","St-Jacques \u002f Gauvin","Henri-Julien \u002f de Bellechasse","Cadillac \u002f Sherbrooke","Garnier \u002f St-Joseph","Milton \u002f Universit\u00e9","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Parc St-Paul (Marc-Sauvalle \u002f Le Caron)","C\u00e9gep Andr\u00e9-Laurendeau","des Bassins \u002f Richmond","Wilderton  \u002f Van Horne","de la Commune \u002f McGill","Bordeaux \u002fGilford","d'Oxford \u002f de Monkland","Drolet \u002f Laurier","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","des R\u00e9collets \u002f McGill","Alexandre-DeS\u00e8ve \u002f Ste-Catherine","De La Roche \u002f Everett","Marquette \u002f Villeray","Prince-Arthur \u002f St-Urbain","St-Andr\u00e9 \u002f St-Antoine","Ste-Catherine \u002f D\u00e9z\u00e9ry","Dorion \u002f Ontario","Crescent \u002f de Maisonneuve","Square Ahmerst \u002f Wolfe","35e avenue \u002f Beaubien","Drolet \u002f Gounod","de Bullion \u002f du Mont-Royal","Cartier \u002f Marie-Anne","M\u00e9tro Universit\u00e9 de Montr\u00e9al (\u00c9douard-Montpetit \u002f Louis-Collin)","Queen \u002f Wellington","35e avenue \u002f Beaubien","St-Charles \u002f Charlotte","St-Mathieu \u002fSte-Catherine","de Hampton \u002f de Monkland","Rouen \u002f Fullum","de Gasp\u00e9 \u002f Dante","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Boyer \u002f St-Zotique","15e avenue \u002f Rosemont","Marquette \u002f Rachel","Bennett \u002f Ste-Catherine","de Bordeaux \u002f Marie-Anne","Viger \u002f Chenneville","Chabot \u002f de Bellechasse","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","d'Outremont \u002f Ogilvy","Bishop \u002f Ste-Catherine","Mansfield \u002f Ren\u00e9-L\u00e9vesque","St-Denis \u002f de Maisonneuve","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","Park Row O \u002f Sherbrooke","Gilford \u002f de Lanaudi\u00e8re","Guizot \u002f St-Denis","Jeanne-d'Arc \u002f Ontario","d'Orl\u00e9ans \u002f Hochelaga","Atwater \u002f Sherbrooke","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Chauveau \u002f de l'Assomption","8e avenue \u002f Saint-Joseph Lachine","Laval \u002f Duluth","Ropery \u002f Augustin-Cantin","Place d'Youville \u002f McGill","4e avenue \u002f Masson","Rachel \u002f Br\u00e9beuf","Centre \u00c9PIC (St-Zotique \u002f 40e avenue)","Jeanne-Mance \u002f Ren\u00e9-L\u00e9vesque","St-Cuthbert \u002f St-Urbain","Mozart \u002f St-Laurent","St-Dominique \u002f Rachel","Aylwin \u002f Ontario","Duluth \u002f St-Denis","de Gasp\u00e9 \u002f Marmier","Fullum \u002f St-Joseph","Plessis \u002f Ontario","de Vend\u00f4me \u002f de Maisonneuve","M\u00e9tro Villa-Maria (D\u00e9carie \u002f de Monkland)","Villeneuve \u002f St-Laurent","Lionel-Groulx \u002f George-Vanier","Waverly \u002f St-Zotique","de Grosbois \u002f Duchesneau","Chambord \u002f Laurier","March\u00e9 Atwater","Bannantyne \u002f de l'\u00c9glise","Papineau \u002f \u00c9mile-Journault","Rose-de-Lima \u002f Notre-Dame","Dollard \u002f Van Horne","de Gasp\u00e9 \u002f de Li\u00e8ge","Boyer \u002f Rosemont","Notre-Dame \u002f de la Montagne","Omer-Lavall\u00e9e \u002f Midway","Boyer \u002f Jean-Talon","Greene \u002f Workman","Notre-Dame \u002f St-Gabriel","Drolet \u002f Faillon","Villeneuve \u002f St-Urbain","Milton \u002f du Parc","Joseph-Manceau \u002f Ren\u00e9-L\u00e9vesque","Marie-Anne \u002f St-Hubert","St-Dominique \u002f St-Zotique","Milton \u002f Durocher","Ellendale \u002f C\u00f4te-des-Neiges","de Bordeaux \u002f Masson","Parc Plage","Metcalfe \u002f de Maisonneuve","St-Charles \u002f Grant","Georges-Vanier \u002f Notre-Dame","26e avenue \u002f Beaubien","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","St-Hubert \u002f Rachel","Marquette \u002f Laurier","Davaar \u002f de la C\u00f4te-Ste-Catherine","Molson \u002f Beaubien","Marquette \u002f du Mont-Royal","Dollard \u002f Van Horne","Rousselot \u002f Jarry","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","Beatty \u002f de Verdun","Fran\u00e7ois-Boivin \u002f B\u00e9langer","Joseph Manceau \u002f Ren\u00e9-L\u00e9vesque","Georges-Vanier \u002f Notre-Dame","Durocher \u002f Van Horne","de Bordeaux \u002f Jean-Talon","Terrasse Mercure \u002f Fullum","Andr\u00e9-Laurendeau \u002f Rachel","Jeanne-Mance \u002f St-Viateur","Harvard \u002f de Monkland","de Maisonneuve \u002f Aylmer","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","St-Charles \u002f Ch\u00e2teauguay","Fabre \u002f St-Zotique","Plessis \u002f Logan","Parc Chamberland (Dorais \u002f de l'\u00c9glise)","Omer-Lavall\u00e9e \u002f Midway","Jeanne Mance \u002f du Mont-Royal","Gary-Carter \u002f St-Laurent","St-Cuthbert \u002f St-Urbain","de Bordeaux \u002fGilford","Clark \u002f Rachel","Charlevoix \u002f Lionel-Groulx","de Mentana \u002f Rachel","St-Marc \u002f Sherbrooke","Bishop \u002f de Maisonneuve","16e avenue \u002f St-Joseph","Cartier \u002f St-Joseph","B\u00e9langer \u002f St-Denis","Berlioz \u002f de l'\u00cele des Soeurs","Hutchison \u002f Prince-Arthur","Victoria Hall","de la Montagne \u002f Sherbrooke","Gauthier \u002f Papineau","St-Charles \u002f Montarville","LaSalle \u002f Godin","Terminus Newman (Airlie \u002f Lafleur)","Desjardins \u002f Ontario","M\u00e9tro Viau (Pierre-de-Coubertin \u002f Viau)","Sanguinet \u002f de Maisonneuve","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","Lacombe \u002f de la C\u00f4te-des-Neiges","Graham \u002f Brookfield","Villeneuve \u002f St-Laurent","St-Patrick \u002f Briand","Jogues \u002f Allard","Bloomfield \u002f Bernard","Ste-Catherine \u002f Dezery","15e avenue \u002f Masson","Casino de Montr\u00e9al","Bishop \u002f Ste-Catherine","Jean-Brillant \u002f McKenna","Alexis-Nihon \u002f St-Louis","Louis H\u00e9mon \u002f Rosemont","Drolet \u002f Jarry","Parc Jeanne-Mance (du Mont-Royal \u002f du Parc)","Basin \u002f du S\u00e9minaire","de la Roche \u002f  de Bellechasse","Belmont \u002f Robert-Bourassa","Parc Loyola (Somerled \u002f Doherty)","de la Roche \u002f Everett","Ottawa \u002f William","Chabot \u002f de Bellechasse","Louis-H\u00e9bert \u002f Beaubien","LaSalle \u002f 80e avenue","St-Urbain \u002f de la Gaucheti\u00e8re","Berlioz \u002f de l'\u00cele des Soeurs","Parc Outremont (Bloomfield \u002f Elmwood)","M\u00e9tro Laurier (Rivard \u002f Laurier)","de Br\u00e9beuf \u002f Laurier","Fleury \u002f de St-Firmin","Duchesneau \u002f Marseille","Ste-Famille \u002f Sherbrooke","Meunier \u002f Fleury","Marmier \u002f St-Denis","de Bullion \u002f du Mont-Royal","St-Charles \u002f Charlotte","Rousselot \u002f Jarry","Calixa-Lavall\u00e9e \u002f Rachel","Stuart \u002f St-Viateur","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Grand Boulevard \u002f Sherbrooke","Sanguinet \u002f Ontario","Crescent \u002f Ren\u00e9-L\u00e9vesque","19e avenue \u002f St-Zotique","de la Commune \u002f Berri","Elsdale\u002f Louis-Hebert","10e Avenue \u002f Rosemont","Marquette \u002f Jean-Talon","Sanguinet \u002f Ontario","Lincoln \u002f du Fort","Plessis \u002f Ren\u00e9-L\u00e9vesque","Chambord \u002f B\u00e9langer","du Rosaire \u002f St-Hubert","de C\u00f4me \u002f Jean-Talon","C\u00f4te St-Antoine \u002f Royal","de la Commune \u002f Berri","Sanguinet \u002f Ste-Catherine","Bennett \u002f Ste-Catherine","Metcalfe \u002f de Maisonneuve","Ontario \u002f du Havre","Lionel-Groulx \u002f George-Vanier","St-Andr\u00e9 \u002f St-Gr\u00e9goire","de Gasp\u00e9 \u002f Jarry","Chauveau \u002f de l'Assomption","Casgrain \u002f Laurier","2e avenue \u002f Wellington","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Chomedey \u002f de Maisonneuve","du Havre \u002f de Rouen","de Bullion \u002f du Mont-Royal","La Tour de Montr\u00e9al (Pierre-de-Coubertin \u002f Bennett)","City Councillors \u002f du President-Kennedy","Montcalm \u002f de Maisonneuve","William \u002f St-Henri","Ernest-Gendreau \u002f du Mont-Royal","30e avenue \u002f St-Zotique","Parc Kent (de Kent \u002f Hudson)","1\u00e8re avenue \u002f Rosemont","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Clark \u002f Evans","des \u00c9rables \u002f Dandurand","Plessis \u002f Ontario","Plessis \u002f Ren\u00e9-L\u00e9vesque","Hutchison \u002f Van Horne","de l'Esplanade \u002f Fairmount","Mackay \u002f Sainte-Catherine","Aylmer \u002f Ste-Catherine","Wolfe \u002f Robin","M\u00e9tro Vend\u00f4me (de Marlowe \u002f de Maisonneuve)","19e avenue \u002f St-Zotique","St-Andr\u00e9 \u002f Ste-Catherine","Rose-de-Lima \u002f Workman","University \u002f des Pins","St-Charles \u002f St-Sylvestre","LaSalle \u002f S\u00e9n\u00e9cal","Ontario \u002f Sicard","St-Andr\u00e9 \u002f Ontario","Chabanel \u002f du Parc","Lincoln \u002f du Fort","Coloniale \u002f du Mont-Royal","Milton \u002f Durocher","Clark \u002f de Li\u00e8ge","City Councillors \u002f du President-Kennedy","Towers \u002f Ste-Catherine","Chomedey \u002f de Maisonneuve","Coll\u00e8ge \u00c9douard-Montpetit (de Gentilly \u002f de Normandie)","Benny \u002f Sherbrooke","Marquette \u002f des Carri\u00e8res","St-Joseph \u002f21e avenue Lachine","Ball \u002f Querbes","Prince-Arthur \u002f du Parc","Panet \u002f de Maisonneuve","St-Nicolas \u002f Place d'Youville","Hutchison \u002f Beaubien","Ste-Catherine \u002f St-Marc","Parc des Rapides (LaSalle \u002f 6e avenue)","Montcalm \u002f de Maisonneuve","Marquette \u002f du Mont-Royal","de Maisonneuve \u002f Robert-Bourassa","Resther \u002f du Mont-Royal","McGill \u002f des R\u00e9collets","Cartier \u002f Marie-Anne","Alexandra \u002f Waverly","Resther \u002f St-Joseph","Gauthier \u002f Papineau","de Ch\u00e2teaubriand \u002f Beaubien","de Normanville \u002f Beaubien","Cartier \u002f Marie-Anne","Laurier \u002f de Bordeaux","Waverly \u002f Van Horne","Wilderton  \u002f Van Horne","Ateliers municipaux de St-Laurent (Cavendish \u002f Poirier)","Dandurand \u002f de Lorimier","Henri-Julien \u002f Villeray","Jogues \u002f Allard","Rosemont \u002f Viau","Cartier \u002f Masson","Boyer \u002f Rosemont","St-Zotique \u002f 39e avenue","St-Alexandre \u002f Ste-Catherine","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f Mansfield)","5e avenue \u002f Bannantyne","M\u00e9tro Langelier (Langelier \u002f Sherbrooke)","St-Charles \u002f Sauv\u00e9","Grand Trunk \u002f Hibernia","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","Wilson \u002f Sherbrooke","Darling \u002f Sherbrooke","Henri-Julien \u002f Jean-Talon","Henri-Julien \u002f Villeray","Hochelaga \u002f Chapleau","Square St-Louis","St-Dominique \u002f Jean-Talon","de Kent \u002f de la C\u00f4te-des-Neiges","Coll\u00e8ge \u00c9douard-Montpetit","Ste-Catherine \u002f Union","St-Andr\u00e9 \u002f Robin","St-Cuthbert \u002f St-Urbain","St-Hubert \u002f Rachel","Ball \u002f Querbes","Ontario \u002f Sicard","Querbes \u002f Laurier","Drummond \u002f de Maisonneuve","de Mentana \u002f Laurier","Valois \u002f Ontario","de Mentana \u002f Gilford","Parc Van Horne (de la Peltrie \u002f de Westbury)","Island \u002f Centre","Ryde \u002f Charlevoix","Alexandra \u002f Jean-Talon","Hillside \u002f Ste-Catherine","Logan \u002f Fullum","des \u00c9rables \u002f B\u00e9langer","St-Cuthbert \u002f St-Urbain","des \u00c9rables \u002f B\u00e9langer","Notre-Dame \u002f Peel","Casino de Montr\u00e9al","Champdor\u00e9 \u002f de Lorimier","Davaar \u002f de la C\u00f4te-Ste-Catherine","5e avenue \u002f Verdun","Hudson \u002f de Kent","Drolet \u002f Beaubien","Pr\u00e9sident-Kennedy \u002f McGill College","Argyle \u002f Bannantyne","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","Union \u002f Cathcart","Le Caron \u002f Marc-Sauvalle ","Parc Jean-Drapeau (Chemin Macdonald)","de la Commune \u002f St-Sulpice","de la Gaucheti\u00e8re \u002f Mansfield","Marquette \u002f Villeray","Cypress \u002f Peel","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Georges-Vanier \u002f Notre-Dame","University \u002f des Pins","Gauvin \u002f Notre-Dame","de la Commune \u002f St-Sulpice","Wilson \u002f Sherbrooke","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Laurier \u002f Bordeaux","LaSalle \u002f S\u00e9n\u00e9cal","Notre-Dame \u002f Peel","Fullum \u002f Marie-Anne","Fortune \u002f Wellington","Biblioth\u00e8que de Rosemont (9e avenue \u002f Rosemont)","Parc J.-Arthur-Champagne (de Chambly \u002f du Mont-Royal)","M\u00e9tro Fabre (Marquette \u002f Jean-Talon)","Rachel \u002f Br\u00e9beuf","Wellington \u002f 2e avenue","University \u002f Prince-Arthur","de Lanaudi\u00e8re \u002f du Mont-Royal","Fabre \u002f St-Zotique","de la Roche \u002f St-Joseph","Wolfe \u002f Robin","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","B\u00e9langer \u002f St-Denis","Parc Langelier - Marie-Victorin","Prince-Arthur \u002f du Parc","Gary-Carter \u002f St-Laurent","8e avenue \u002f Jarry","de St-Vallier \u002f St-Zotique","St-Urbain \u002f de la Gaucheti\u00e8re","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Marie-Anne \u002f St-Hubert","des R\u00e9collets \u002f Martial","St-Dominique \u002f St-Zotique","1\u00e8re avenue \u002f Masson","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f Mansfield)","de Lanaudi\u00e8re \u002f Marie-Anne","Pierre-de-Coubertin \u002f Aird","William \u002f Robert-Bourassa","Crescent \u002f Ren\u00e9-L\u00e9vesque","Murray \u002f William","Gauthier \u002f De Lorimier","Marquette \u002f St-Gr\u00e9goire","St-Mathieu \u002f Sherbrooke","Lucien L'Allier \u002f St-Jacques","Parc Kirkland (des \u00c9rables \u002f Boyer)","Jacques-Cassault \u002f Christophe-Colomb","M\u00e9tro George-Vanier (St-Antoine \u002f Canning)","Gauthier\u002f Des \u00c9rables","\u00c9douard-Montpetit \u002f de Stirling","Roy \u002f St-Laurent","M\u00e9tro Laurier (Rivard \u002f Laurier)","Notre-Dame \u002f Gauvin","des \u00c9rables \u002f B\u00e9langer","Beaubien \u002f 8e avenue","Rachel \u002f de Br\u00e9beuf","Lajeunesse \u002f Jarry","Boyer \u002f Everett","8e avenue \u002f Beaubien","Berri \u002f Gilford","26e avenue \u002f Beaubien","Clark \u002f St-Viateur","Chabot \u002f Beaubien","Chabot \u002f de Bellechasse","Dante \u002f de Gasp\u00e9","Parc Rosemont (Dandurand \u002f d'Iberville)","Parthenais \u002f Ste-Catherine","St-Christophe \u002f Ren\u00e9-L\u00e9vesque","Ottawa \u002f St-Thomas","19e avenue \u002f St-Zotique","1\u00e8re  avenue \u002fSt-Zotique","Bennett \u002f Pierre-de-Coubertin","Marquette \u002f Rachel","de l'\u00c9glise \u002f Bannantyne","de l'\u00c9glise \u002f de Verdun","Parc de l'H\u00f4tel-de-Ville (Dub\u00e9 \u002f Laurendeau)","McTavish \u002f Sherbrooke","St-Denis \u002f Ren\u00e9-L\u00e9vesque","Mackay \u002f de Maisonneuve","de la Commune \u002f Berri","Jeanne-Mance \u002f St-Viateur","Waverly \u002f Fairmount","U. Concordia - Campus Loyola (Sherbrooke \u002f West Broadway)","du Mont-Royal \u002f Vincent-d'Indy","M\u00e9tro Viau (Pierre-de-Coubertin \u002f Viau)","Parc Ivan-Franko (33e avenue \u002f Victoria)","Larivi\u00e8re \u002f de Lorimier","Bishop \u002f Ste-Catherine","Poupart \u002f Ste-Catherine","Augustin-Cantin \u002f Shearer","Parc J.-Arthur-Champagne (Chambly \u002f du Mont-Royal)","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Alexandra \u002f Jean-Talon","Parthenais \u002f Ste-Catherine","de la Montagne \u002f Sherbrooke","d'Outremont \u002f Ogilvy","Grand Trunk \u002f Hibernia","Laporte \u002f St-Antoine","Wolfe \u002f Ren\u00e9-L\u00e9vesque","St-Hubert \u002f Ren\u00e9-L\u00e9vesque","5e avenue \u002f Bannantyne","de Vend\u00f4me \u002f de Maisonneuve","Sanguinet \u002f de Maisonneuve","Louis-H\u00e9mon \u002f Rosemont","M\u00e9tro Lionel-Groulx (Lionel-Groulx \u002f Charlevoix)","Jacques-Casault \u002f Christophe-Colomb","LaSalle \u002f Crawford","Cadillac \u002f Sherbrooke","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Faillon \u002f St-Hubert","Square Phillips","19e avenue \u002f Saint-Antoine","Queen \u002f Wellington","de Melrose \u002f Sherbrooke","Louis H\u00e9mon \u002f Rosemont","Maguire \u002f Henri-Julien","de Montmorency \u002f Richardson","Berri \u002f Rachel","Clark \u002f St-Viateur","Tupper \u002f Atwater","de Ch\u00e2teaubriand \u002f Beaubien","du Tricentenaire \u002f Prince-Albert","de Gasp\u00e9 \u002f St-Viateur","Hampton \u002f Monkland","D\u00e9z\u00e9ry \u002f Ontario","Fullum \u002f Sherbrooke","Melville \u002f de Maisonneuve","M\u00e9tro Georges-Vanier (St-Antoine \u002f Canning)","Aylmer \u002f Sherbrooke","Rivard \u002f Rachel","Parc J.-Arthur-Champagne (de Chambly \u002f du Mont-Royal)","du Mont-Royal \u002f Vincent-d'Indy","Jeanne Mance \u002f du Mont-Royal","1\u00e8re avenue \u002f Masson","Sanguinet \u002f de Maisonneuve","Boyer \u002f Jean-Talon","Clark \u002f Rachel","de la Commune \u002f McGill","Valois \u002f Ontario","Hamilton \u002f Jolicoeur","10e Avenue \u002f Rosemont","Le Caron \u002f Marc-Sauvalle ","Villeneuve \u002f du Parc","Square Phillips","Bordeaux \u002f Masson","Lapierre \u002f Serge","Logan \u002f Fullum","12e avenue \u002f St-Zotique","M\u00e9tro Lionel-Groulx (St-Jacques \u002f Atwater)","Terrasse Mercure \u002f Fullum","Ste-Famille \u002f des Pins","Place Longueuil","St-Dominique \u002f Napol\u00e9on","Fullum \u002f Sherbrooke ","de Gasp\u00e9 \u002f Fairmount","Waverly \u002f Van Horne","March\u00e9 Atwater","Boyer \u002f Beaubien","Faillon \u002f St-Denis","Plessis \u002f Ren\u00e9-L\u00e9vesque","Le Caron \u002f Marc-Sauvalle ","L\u00e9a-Roback \u002f Sir-Georges-Etienne-Cartier","Chambord \u002f B\u00e9langer","de Normanville \u002f B\u00e9langer","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","de la C\u00f4te St-Paul \u002f St-Ambroise","Frontenac \u002f du Mont-Royal","Parc Plage","Ontario \u002f Viau","O'Brien \u002f Poirier","de Kent \u002f Victoria","18e avenue \u002f Rosemont","Rouen \u002f Fullum","St-Jacques \u002f St-Laurent","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","Duluth \u002f de l'Esplanade","Par\u00e9 \u002f Mountain Sights","Villeneuve \u002f St-Urbain","Garnier \u002f du Mont-Royal","B\u00e9langer \u002f des Galeries d'Anjou","St-Dominique \u002f Jean-Talon","University \u002f Prince-Arthur","Villeneuve \u002f St-Laurent","Champagneur \u002f Jean-Talon","de la Roche \u002f  de Bellechasse","Chambord \u002f Laurier","d'Orl\u00e9ans \u002f Hochelaga","De la Commune \u002f King","Leman \u002f de Chateaubriand","Union \u002f du Pr\u00e9sident-Kennedy","de la Roche \u002f  de Bellechasse","Marquette \u002f Laurier","Fabre \u002f St-Zotique","Waverly \u002f St-Zotique","Joseph-Manseau \u002f Ren\u00e9-L\u00e9vesque","Duluth \u002f St-Denis","Hutchison \u002f Beaubien","Paul Boutet \u002f Jarry","Henri-Julien \u002f du Mont-Royal","M\u00e9tro Jolicoeur (Drake \u002f de S\u00e8ve)","Robin \u002f de la Visitation","Dandurand \u002f de Lorimier","Lajeunesse \u002f Villeray (place Tap\u00e9o)","Ann \u002f Ottawa","Victoria \u002f de Maisonneuve","Hutchison\u002f Prince-Arthur","Marquette \u002f Jean-Talon","Tolhurst \u002f Fleury","7e avenue \u002f St-Joseph","Berri \u002f St-Gr\u00e9goire","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f de la Cath\u00e9drale)","Dandurand \u002f des \u00c9rables","D\u00e9z\u00e9ry \u002f Ontario","Bourget \u002f St-Jacques","Alexandra \u002f Jean Talon","de Bordeaux \u002f Marie-Anne","de Chambly \u002f Rachel","de St-Vallier \u002f St-Zotique","Alexandra \u002f Waverly","Guizot \u002f St-Denis","Notre-Dame \u002f St-Gabriel","Faillon \u002f St-Denis","Waverly \u002f St-Viateur","Chambly \u002f Rachel","de Bullion \u002f St-Joseph","Marquette \u002f Villeray","du Parc-La Fontaine \u002f Duluth","CHSLD \u00c9loria-Lepage (de la P\u00e9pini\u00e8re \u002f de Marseille)","des Seigneurs \u002f Notre-Dame","de Lanaudi\u00e8re \u002f Marie-Anne","Duluth \u002f St-Laurent","de Normanville \u002f Jean-Talon","Bourbonni\u00e8re \u002f du Mont-Royal","Rousselot \u002f Villeray","Mansfield \u002f Ren\u00e9-L\u00e9vesque","du Mont-Royal \u002f du Parc","Clark \u002f Prince-Arthur","Chambly \u002f Rachel","Prince-Arthur \u002f du Parc","Complexe sportif Claude-Robillard","Palm \u002f St-Remi","St-Dominique \u002f Bernard","St-Viateur \u002f Casgrain","Berri \u002f Sherbrooke","Lajeunesse \u002f Villeray (place Tap\u00e9o)","Logan \u002f d'Iberville","St-Nicolas \u002f Place d'Youville","Querbes \u002f Laurier","Mackay \u002f de Maisonneuve","Henri-Julien \u002f de Castelnau","La Ronde","Dandurand \u002f Papineau","Jeanne-Mance \u002f Beaubien","Grand Trunk \u002f Hibernia","10e avenue \u002f Masson","de la Roche \u002f St-Joseph","Maguire \u002f St-Laurent","de Repentigny \u002f Sherbrooke","Parc Reine-\u00c9lizabeth (Penniston \u002f Crawford)","Place \u00c9milie-Gamelin (St-Hubert \u002f de Maisonneuve)","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Jeanne d'Arc \u002f Ontario","Quai de la navette fluviale","Mackay \u002f de Maisonneuve","du Parc-La Fontaine \u002f Roy","29e avenue \u002f St-Zotique","Rivier \u002f Davidson","Tupper \u002f Atwater","1\u00e8re avenue \u002f Masson","Molson \u002f William-Tremblay","Jean-Brillant \u002f de la C\u00f4te-des-Neiges","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f de la Cath\u00e9drale)","de l'Esplanade \u002f Fairmount","Beaudry \u002f Sherbrooke","\u00c9douard-Montpetit \u002f Stirling","St-Dominique \u002f Rachel","1\u00e8re  avenue \u002fSt-Zotique","de Bordeaux \u002f Jean-Talon","Louis-H\u00e9mon \u002f Villeray","Chabot \u002f du Mont-Royal","1\u00e8re avenue \u002f Masson","Nicolet \u002f Sherbrooke","de Br\u00e9beuf \u002f Rachel","Clark \u002f Rachel","\u00c9mile-Journault \u002f de Chateaubriand","Cartier \u002f St-Joseph","Calixa-Lavall\u00e9e \u002f Rachel","Jeanne-Mance \u002f Laurier","Lusignan \u002f St-Jacques","des \u00c9rables \u002f B\u00e9langer","Fullum \u002f Ontario","Regina \u002f de Verdun","McCulloch \u002f de la C\u00f4te Ste-Catherine","Valli\u00e8res \u002f St-Laurent","Mozart \u002f St-Laurent","Hutchison \u002f Beaubien","de la Montagne \u002f Sherbrooke","Belmont \u002f du Beaver Hall","Cartier \u002f Dandurand","St-Marc \u002f Sherbrooke","28e avenue \u002f Rosemont","Joliette \u002f Ste-Catherine","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Lajeunesse \u002f Villeray","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","Poupart \u002f Ste-Catherine","Bel Air \u002f St-Antoine","Leman \u002f de Chateaubriand","3e avenue \u002f Dandurand","Le Caron \u002f Marc-Sauvalle ","de Gasp\u00e9 \u002f Dante","Tupper \u002f du Fort","Chambord \u002f Beaubien","M\u00e9tro Cadillac (Sherbrooke \u002f de Cadillac)","St-Andr\u00e9 \u002f de Maisonneuve","Chambord \u002f Fleury","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","Drummond \u002f de Maisonneuve","Champlain \u002f Ontario","Louis-H\u00e9bert \u002f St-Zotique","de l'H\u00f4tel-de-Ville \u002f Roy","Lincoln \u002f Lambert Closse","Roy \u002f St-Laurent","des \u00c9rables \u002f Rachel","Garnier \u002f St-Joseph","Hamilton \u002f Jolicoeur","10e avenue \u002f Masson","Natatorium (LaSalle \u002f Rolland)","Parc du P\u00e9lican (Laurier\u002f 2e avenue)","de Bordeaux \u002f Jean-Talon","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","du Square Ahmerst \u002f Wolfe","Crescent \u002f Ren\u00e9-L\u00e9vesque","De La Roche \u002f Everett","Drummond \u002f de Maisonneuve","Gauthier \u002f Parthenais","St-Dominique \u002f St-Zotique","Cypress \u002f Peel","Island \u002f Centre","M\u00e9tro Rosemont (de St-Vallier \u002f Rosemont)","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Parc Jeanne Mance (monument George-\u00c9tienne Cartier)","Calixa-Lavall\u00e9e \u002f Sherbrooke","Prince-Arthur \u002f du Parc","Boyer \u002f Beaubien","de Bullion \u002f St-Joseph","Laporte \u002f Guay","des \u00c9rables \u002f du Mont-Royal","Laurier \u002f 15e avenue","Drake \u002f de S\u00e8ve","de l'Esplanade \u002f Fairmount","Hamilton \u002f Jolicoeur","Island \u002f Centre","Bossuet \u002f Pierre-de-Coubertin","Plessis \u002f Ontario","de la C\u00f4te St-Antoine \u002f Royal","Laval \u002f Duluth","Louis-H\u00e9bert \u002f St-Zotique","Hutchison \u002f Beaubien","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Garnier \u002f St-Joseph","Davaar \u002f de la C\u00f4te-Ste-Catherine","Gordon \u002f Wellington","Boyer \u002f St-Zotique","de l'H\u00f4tel-de-Ville \u002f Roy","5e avenue \u002f Rosemont","Villeneuve \u002f St-Urbain","M\u00e9tro Jean-Drapeau","Ropery \u002f Augustin-Cantin","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Plessis \u002f Ontario","Omer-Lavall\u00e9e \u002f du Midway","Roy \u002f St-Denis","de Mentana \u002f Marie-Anne","Shearer \u002f Centre","Waverly \u002f St-Zotique","Ste-Famille \u002f Sherbrooke","Jardin Botanique (Pie-IX \u002f Sherbrooke)","St-Dominique \u002f Rachel","Mackay \u002fde Maisonneuve (Sud)","Parc d'Antioche","Chambord \u002f Beaubien","Jardin Botanique (Pie-IX \u002f Sherbrooke)","Rivard \u002f Rachel","Mozart \u002f Waverly","Parc St-Claude (7e avenue \u002f 8e rue)","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","de Monkland \u002f Girouard","Guizot \u002f St-Denis","Parthenais \u002f Ste-Catherine","Union \u002f President-Kennedy","Boyer \u002f Jean-Talon","St-Hubert \u002f Duluth","de Li\u00e8ge \u002f Lajeunesse","M\u00e9tro de Castelnau (de Castelnau \u002f Clark)","Quesnel \u002f Vinet","March\u00e9 Atwater","de Lanaudi\u00e8re \u002f B\u00e9langer","Chambord \u002f B\u00e9langer","Molson \u002f Beaubien","Vend\u00f4me \u002f Sherbrooke","de Li\u00e8ge \u002f Lajeunesse","Parc de Bullion (de Bullion \u002f Prince-Arthur)","Roy \u002f St-Denis","St-Andr\u00e9 \u002f de Maisonneuve","de St-Vallier \u002f Rosemont","Maguire \u002f St-Laurent","de Gasp\u00e9 \u002f Marmier","St-Jacques \u002f McGill","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Fullum \u002f Sherbrooke ","Larivi\u00e8re \u002f de Lorimier","Canning \u002f Notre-Dame","Ste-Catherine \u002f Labelle","de Champlain \u002f Ontario","Grand Boulevard \u002f Sherbrooke","St-G\u00e9rard \u002f Villeray","Ste-Famille \u002f des Pins","Louis-H\u00e9bert \u002f Beaubien","du Rosaire \u002f St-Hubert","Marquette \u002f du Mont-Royal","Ellendale \u002f C\u00f4te-des-Neiges","M\u00e9tro Rosemont (Rosemont \u002f de St-Vallier)","Biblioth\u00e8que de Rosemont (9e avenue \u002f Rosemont)","Ville-Marie \u002f Ste-Catherine","Viger \u002f Jeanne Mance","du Champ-de-Mars \u002f Gosford","Hutchison \u002f Sherbrooke","Wurtele \u002f de Rouen","St-Alexandre \u002f Ste-Catherine","Casino de Montr\u00e9al","Briand \u002f Carron","Louis-H\u00e9bert \u002f B\u00e9langer","Faillon \u002f St-Hubert","Place du Village-de-la-Pointe-aux-Trembles (St-Jean-Baptiste \u002f Notre-Dame)","Gascon \u002f de Rouen","Chomedey \u002f de Maisonneuve","Place du Commerce","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Alexandre-DeS\u00e8ve \u002f la Fontaine","Ste-Famille \u002f des Pins","Drolet \u002f St-Zotique","Marseille \u002f Viau","Logan \u002f d'Iberville","St-Hubert \u002f Duluth","Ross \u002f de l'\u00c9glise","Trans Island \u002f Queen-Mary","Faillon \u002f St-Hubert","Belmont \u002f Beaver Hall","Wellington \u002f Hickson","Chambord \u002f Beaubien","Clark \u002f Ontario","Marquette \u002f Laurier","de Monkland \u002f Girouard","Pr\u00e9sident-Kennedy \u002f McGill College","Hutchison \u002f Van Horne","du Parc-La Fontaine \u002f Duluth","de Gasp\u00e9 \u002f Beaubien","Berri \u002f Cherrier","Mackay \u002f Ren\u00e9-L\u00e9vesque","Notre-Dame \u002f de la Montagne","de Gasp\u00e9 \u002f de Li\u00e8ge","Ste-Catherine \u002f Labelle","St-Charles \u002f Montarville","de Bullion \u002f du Mont-Royal","Parc Outremont (Bloomfield \u002f Elmwood)","St-Dominique \u002f Bernard","\u00c9mile-Journault \u002f de Chateaubriand","Regina \u002f de Verdun","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Prince-Arthur \u002f du Parc","de Ch\u00e2teaubriand \u002f Beaubien","Rouen \u002f du Havre","St-Dominique \u002f Jean-Talon","de Gasp\u00e9 \u002f Marmier","Waverly \u002f St-Viateur","Pierre-de-Coubertin \u002f Aird","Marie-Anne \u002f Papineau","Gilford \u002f Br\u00e9beuf","Fortune\u002fwellington","Natatorium (LaSalle \u002f Rolland)","de Soissons \u002f de Darlington","de Lanaudi\u00e8re \u002f Marie-Anne","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Parc Sauv\u00e9 (Balzac \u002f de Bayonne)","de Clifton \u002f Sherbrooke","Guizot \u002f St-Laurent","7e avenue \u002f St-Joseph Rosemont","B\u00e9langer \u002f St-Denis","McGill \u002f des R\u00e9collets","Marcil \u002f Sherbrooke","de Bordeaux \u002f Gilford","1\u00e8re  avenue \u002fSt-Zotique","Bloomfield \u002f Bernard","Chabot \u002f Villeray","Louis-H\u00e9mon \u002f Jean-Talon","Parc Jeanne Mance (monument \u00e0 sir George-\u00c9tienne Cartier)","Fabre \u002f Beaubien","Centre Pierre-Charbonneau","St-Jacques \u002f St-Laurent","Richardson \u002f de Montmorency","de Bellechasse \u002f de St-Vallier","Bourbonni\u00e8re \u002f du Mont-Royal","Drolet \u002f Beaubien","4e avenue \u002f Masson","Fleury \u002f Andr\u00e9-Jobin","Benny \u002f Monkland","McTavish \u002f Sherbrooke","du Quesne \u002f Pierre-de-Coubertin","Ropery \u002f Grand Trunk","M\u00e9tro de Castelnau (de Castelnau \u002f St-Laurent)","Laval \u002f Rachel","Peel \u002f avenue des Canadiens de Montr\u00e9al","M\u00e9tro Jolicoeur (Drake \u002f de S\u00e8ve)","Cartier \u002f B\u00e9langer","Gauthier \u002f Papineau","Dandurand \u002f Papineau","de Maisonneuve \u002f Mansfield (nord)","Chabot \u002f du Mont-Royal","Ann \u002f Wellington","Bishop \u002f Ste-Catherine","Calixa-Lavall\u00e9e \u002f Sherbrooke","Hochelaga \u002f Chapleau","Saint-Viateur \u002f Casgrain","de Kent \u002f Victoria","Queen \u002f Wellington","St-Mathieu \u002f Ste-Catherine","St-Dominique \u002f Villeray","Richmond \u002f des Bassins","Georges-Baril \u002f Fleury","St-Jean-Baptiste \u002f Notre-Dame","de Champlain \u002f Ontario","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","du Parc-La Fontaine \u002f Roy","Wilson \u002f Sherbrooke","Rivard \u002f Rachel","des \u00c9rables \u002f Rachel","Des \u00c9rables \u002f Villeray","Roy \u002f St-Christophe","Parc Chopin (Dumouchel \u002f Tessier)","Parc de Bullion (de Bullion \u002f Prince-Arthur)","Metro Jean-Drapeau (Chemin Macdonald)","Coll\u00e8ge \u00c9douard-Montpetit (de Gentilly \u002f de Normandie)","St-Andr\u00e9 \u002f St-Antoine","Drummond \u002f Ste-Catherine","M\u00e9tro Angrignon","P\u00e9loquin \u002f Fleury","Poupart \u002f Ste-Catherine","Marquette \u002f du Mont-Royal","de Rouen \u002f Lesp\u00e9rance","Bloomfield \u002f Bernard","Godin \u002f Bannantyne","Bourbonni\u00e8re \u002f du Mont-Royal","Louis-H\u00e9bert \u002f Beaubien","du Square Ahmerst \u002f Wolfe","Dorion \u002f Ontario","9e avenue \u002f Dandurand","St-Henri \u002f Notre-Dame","Calixa-Lavall\u00e9e \u002f Sherbrooke","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Willibrord \u002f Verdun","Drolet \u002f Beaubien","Berlioz \u002f Boul de l'Ile des Soeurs","Boyer \u002f du Mont-Royal","St-Andr\u00e9 \u002f St-Antoine","Boyer \u002f Beaubien","Rousselot \u002f Jarry","Fabre \u002f St-Zotique","Boyer \u002f Beaubien","Lajeunesse \u002f Villeray (place Tap\u00e9o)","St-Dominique \u002f St-Viateur","de la Commune \u002f Place Jacques-Cartier","Wilson \u002f Sherbrooke","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Chapleau \u002f du Mont-Royal","des \u00c9rables \u002f Gauthier","de la Gaucheti\u00e8re \u002f de Bleury","Valois \u002f Ontario","de la Salle \u002f Ste-Catherine","Square St-Louis (du Square St-Louis \u002f Laval)","du Tricentenaire \u002f Prince-Albert","Parc Outremont (Bloomfield \u002f St-Viateur)","March\u00e9 Maisonneuve","Argyle \u002f Sherbrooke","Augustin-Cantin \u002f Shearer","Tupper \u002f du Fort","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","du Pr\u00e9sident-Kennedy \u002f Jeanne-Mance","Sanguinet \u002f de Maisonneuve","Biblioth\u00e8que Jean-Corbeil (Goncourt \u002f Place Montrichard)","Louis H\u00e9mon \u002f Rosemont","Lusignan \u002f St-Jacques","1\u00e8re avenue \u002f Beaubien","Waverly \u002f St-Viateur","De La Roche \u002f Everett","Tolhurst \u002f Fleury","Napol\u00e9on \u002f  St-Dominique","7e avenue \u002f St-Joseph","Ste-Famille \u002f des Pins","Metcalfe \u002f de Maisonneuve","Ottawa \u002f William","Coloniale \u002f du Mont-Royal","St-Mathieu \u002fSte-Catherine","de la Commune \u002f McGill","Fullum \u002f St-Joseph","Cartier \u002f St-Joseph","St-Andr\u00e9 \u002f Laurier","Bordeaux \u002f Masson","de la Commune \u002f Place Jacques-Cartier","Clark \u002f de Li\u00e8ge","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","de Kent \u002f de la C\u00f4te-des-Neiges","Sanguinet \u002f Ontario","Cartier \u002f B\u00e9langer","Complexe sportif Claude-Robillard","Palm \u002f St-Remi","Chabot \u002f St-Zotique","Swail \u002f Decelles","des Soeurs-Grises \u002f Marguerite-d'Youville","Tupper \u002f Atwater","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","Parthenais \u002f Ste-Catherine","Marmier \u002f St-Denis","Montcalm \u002f de Maisonneuve","Resther \u002f du Mont-Royal","St-Antoine \u002f de la Montagne","Ste-Famille \u002f des Pins","Viger\u002fJeanne Mance","Mozart \u002f St-Laurent","McGill \u002f Place d'Youville","2e avenue \u002f Wellington","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","Lambert-Closse \u002f Tupper","des Bassins \u002f du S\u00e9minaire","Marquette \u002f du Mont-Royal","Calixa-Lavall\u00e9e \u002f Rachel","Natatorium (LaSalle \u002f Rolland)","des \u00c9rables \u002f Gauthier","Notre-Dame \u002f Gauvin","Waverly \u002f St-Viateur","Parc Turin (Chambord \u002f Jean-Talon)","Bernard \u002f Jeanne-Mance","Waverly \u002f Van Horne","Trans Island \u002f Queen-Mary","McGill \u002f Place d'Youville","St-Dominique \u002f Jean-Talon","Pratt \u002f Van Horne","de Gasp\u00e9 \u002f St-Viateur","Viger\u002fJeanne Mance","St-Charles \u002f St-Sylvestre","Duvernay \u002f Charlevoix","10e avenue \u002f Masson","M\u00e9tro Monk (Allard \u002f Beaulieu)","Hillside \u002f Ste-Catherine","Roy \u002f St-Christophe","Bloomfield \u002f Bernard","Lesp\u00e9rance \u002f de Rouen","Stuart \u002f de la C\u00f4te-Ste-Catherine","de Chambly \u002f Rachel","Cadillac \u002f Sherbrooke","du Souvenir \u002f Chomedey","Henri-Julien \u002f Carmel","Garnier \u002f St-Joseph","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Marquette \u002f Jean-Talon","Aylmer \u002f Prince-Arthur","Greene \u002f St-Ambroise","Quesnel \u002f Vinet","Ryde \u002f Charlevoix","Durocher \u002f St-Viateur","Louis-H\u00e9bert \u002f St-Zotique","Grand Trunk \u002f d'Hibernia","Laporte \u002f Guay","M\u00e9tro Frontenac (Ontario \u002f du Havre)","Harvard \u002f de Monkland","St-Jacques \u002f Guy","de l'Arcade \u002f Clark","Langelier \u002f Marie-Victorin","St-Nicolas \u002f Place d'Youville","Jeanne-Mance \u002f Ren\u00e9-L\u00e9vesque","16e avenue \u002f St-Joseph","9e avenue \u002f Dandurand","Dandurand \u002f de Lorimier","Desjardins \u002f Hochelaga","Casgrain \u002f St-Viateur","M\u00e9tro Cadillac (de Cadillac \u002f Sherbrooke)","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","M\u00e9tro Villa Maria (D\u00e9carie \u002f Monkland)","de Bellechasse \u002f de St-Vallier","Hutchison\u002f Prince-Arthur","Aylmer\u002f Sainte-Catherine","Rockland \u002f Lajoie","Lionel-Groulx \u002f George-Vanier","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","de Maisonneuve \u002f Greene","St-Charles \u002f Ch\u00e2teauguay","Cartier \u002f Marie-Anne","M\u00e9tro Beaubien (de Chateaubriand \u002f Beaubien)","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Boyer \u002f Jarry","Hutchison \u002f des Pins","M\u00e9tro Frontenac (Ontario \u002f du Havre)","Guilbault \u002f Clark","Hamel \u002f Sauv\u00e9","de la Commune \u002f McGill","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","Durocher \u002f Bernard","Marcel-Laurin \u002f Beaudet","St-Germain \u002f Hochelaga","du Mont-Royal \u002f Laurendeau","CHSLD \u00c9loria-Lepage (de la P\u00e9pini\u00e8re \u002f de Marseille)","LaSalle \u002f Crawford","Leman \u002f de Chateaubriand","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Sanguinet \u002f de Maisonneuve","Clark \u002f St-Viateur","Ste-Catherine \u002f D\u00e9z\u00e9ry","M\u00e9tro Jolicoeur (de S\u00e8ve \u002f Drake)","St-Jacques \u002f Guy","Garnier \u002f du Mont-Royal","Laurier \u002f de Br\u00e9beuf","de la Gaucheti\u00e8re \u002f Robert-Bourassa","30e avenue \u002f St-Zotique","Parc Georges St-Pierre (Upper Lachine \u002f Oxford)","Br\u00e9beuf \u002f St-Gr\u00e9goire","Dunlop \u002f Van Horne","Ville-Marie \u002f Ste-Catherine","Resther \u002f St-Joseph","Brittany \u002f Ainsley","Alexandre-de-S\u00e8ve \u002f la Fontaine","Georges-Vanier \u002f Notre-Dame","Glengarry \u002f Graham","Parc J.-Arthur-Champagne (Chambly \u002f du Mont-Royal)","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Montcalm \u002f Ontario","Garnier \u002f Everett","Island \u002f Centre","de Gasp\u00e9 \u002f Marmier","Drummond \u002f Sherbrooke","de Vend\u00f4me \u002f de Maisonneuve","Papineau \u002f \u00c9mile-Journault","Chambord \u002f Laurier","Louis-H\u00e9bert \u002f St-Zotique","Omer-Lavall\u00e9e \u002f du Midway","Larivi\u00e8re \u002f de Lorimier","Parc Rosemont (Dandurand \u002f d'Iberville)","Wolfe \u002f Robin","Boyer \u002f Beaubien","Young \u002f Wellington","de l'H\u00f4tel-de-Ville \u002f Roy","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","M\u00e9tro Snowdon (de Westbury \u002f Queen-Mary)","Swail \u002f Decelles","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Parthenais \u002f du Mont-Royal","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","de Lanaudi\u00e8re \u002f Marie-Anne","de Bordeaux \u002f Masson","M\u00e9tro Cr\u00e9mazie (Cr\u00e9mazie \u002f Lajeunesse)","de Champlain \u002f Ontario","Queen \u002f Wellington","Hutchison \u002f Van Horne","Kirkfield \u002f de Chambois","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","St-Charles \u002f Charlotte","de Bordeaux \u002f Rachel","M\u00e9tro Langelier (Sherbrooke \u002f Langelier)","de l'Esplanade \u002f Fairmount","de Bordeaux \u002f Gilford","1\u00e8re avenue \u002f Rosemont","d'Orl\u00e9ans \u002f Masson","Cartier \u002f Masson","Sanguinet \u002f Ste-Catherine","M\u00e9tro C\u00f4te-Vertu (Gohier \u002f Edouard-Laurin)","Natatorium (LaSalle \u002f Rolland)","Ville-Marie \u002f Ste-Catherine","Grand Boulevard \u002f Sherbrooke","Guizot \u002f St-Denis","Duvernay \u002f Charlevoix","M\u00e9tro Parc  (Hutchison \u002f Ogilvy)","35e avenue \u002f Beaubien","de Kent \u002f de la C\u00f4te-des-Neiges","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","Harvard \u002f Monkland","7e avenue \u002f St-Joseph","Messier \u002f Hochelaga","Shannon \u002f Ottawa","du Rosaire \u002f St-Hubert","Sainte-Catherine \u002f Union","de Ville-Marie \u002f Ste-Catherine","Chapleau \u002f Terrasse Mercure","Faillon \u002f St-Denis","Mansfield \u002f Ste-Catherine","de Lisieux \u002f Jean-Talon","Casgrain \u002f Mozart","Champlain \u002f Ontario","des Seigneurs \u002f Notre-Dame","Fullum \u002f Jean Langlois","M\u00e9tro \u00c9douard-Montpetit (du Mont-Royal \u002f Vincent-d'Indy)","Ste-Catherine \u002f St-Denis","St-Jacques \u002f St-Pierre","Laval \u002f Rachel","Beaudry \u002f Sherbrooke","Dollard \u002f Van Horne","St-Jacques \u002f St-Laurent","Henri-Julien \u002f du Carmel","de Maisonneuve \u002f Greene","du Mont-Royal \u002f Augustin-Frigon","Louis-Colin \u002f McKenna","de l'\u00c9p\u00e9e \u002f Jean-Talon","Berri \u002f Cherrier","St-Dominique \u002f Jean-Talon","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","Br\u00e9beuf \u002f St-Gr\u00e9goire","McGill \u002f Place d'Youville","Square St-Louis (du Square St-Louis \u002f Laval)","Metcalfe \u002f Square Dorchester","Durocher \u002f Van Horne","Messier \u002f St-Joseph","de la Gaucheti\u00e8re \u002f Robert-Bourassa","Laval \u002f du Mont-Royal","Boyer \u002f Rosemont","Cote St-Paul \u002f St-Ambroise","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Chabanel \u002f de l'Esplanade","St-Dominique \u002f St-Zotique","Place de la paix ( Pl. du march\u00e9 \u002f St-Dominique )","de Maisonneuve \u002f Robert-Bourassa","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Parc J.-Arthur-Champagne (de Chambly \u002f du Mont-Royal)","Querbes \u002f Ball","Square St-Louis (du Square St-Louis \u002f Laval)","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Alexandre-de-S\u00e8ve \u002f la Fontaine","Argyle \u002f Sherbrooke","Calixa-Lavall\u00e9e \u002f Rachel","Marquette \u002f Fleury","Drolet \u002f Faillon","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Union \u002f Ren\u00e9-L\u00e9vesque","Chauveau \u002f de l'Assomption","de Lanaudi\u00e8re \u002f B\u00e9langer","St-Jacques \u002f St-Pierre","Logan \u002f de Champlain","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Cartier \u002f St-Joseph","Grand Trunk \u002f Hibernia","Bercy \u002f Ontario","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Viger \u002f Chenneville","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Leman \u002f de Chateaubriand","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","6e avenue \u002f Rosemont","St-Jacques \u002f Gauvin","Dixie \u002f 53e avenue","du Rosaire \u002f St-Hubert","Casgrain \u002f de Bellechasse","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","Victoria Hall","de la Roche \u002f  de Bellechasse","Richmond \u002f Notre-Dame","Fleury \u002f Lajeunesse","de Lanaudi\u00e8re \u002f Gilford","Greene \u002f Workman","Ste-Catherine \u002f D\u00e9z\u00e9ry","M\u00e9tro \u00c9douard-Montpetit (du Mont-Royal \u002f Vincent-d'Indy)","CHSLD St-Michel (Jarry \u002f 8e avenue)","de Bordeaux \u002f St-Zotique","du Parc-La Fontaine \u002f Duluth","de Gasp\u00e9 \u002f Villeray","Ernest-Gendreau \u002f du Mont-Royal","de Maisonneuve \u002f Greene","Mackay \u002f de Maisonneuve","Marquette \u002f Laurier","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","St-Dominique \u002f Gounod","1\u00e8re avenue \u002f Beaubien","Terminus Le Carrefour (Terry-Fox \u002f Le Carrefour)","Bourget \u002f St-Jacques","des \u00c9rables \u002f du Mont-Royal","Gascon \u002f Rachel","Parc Saint-Laurent (Dutrisac)","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","Maguire \u002f St-Laurent","Boyer \u002f St-Zotique","St-Dominique \u002f Napol\u00e9on","Lincoln \u002f du Fort","Hogan \u002f Ontario","Chabot \u002f de Bellechasse","Berri \u002f St-Gr\u00e9goire","Marmier \u002f St-Denis","Stuart \u002f St-Viateur","Bernard \u002f Jeanne-Mance","William \u002f Guy","St-Denis \u002f de Maisonneuve","M\u00e9tro Laurier (Rivard \u002f Laurier)","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","St-Viateur\u002fParc","D\u00e9z\u00e9ry \u002f Ontario","Darling \u002f Sherbrooke","Henri-Julien \u002f Jarry","Regina \u002f de Verdun","de St-Vallier \u002f St-Zotique","Sherbrooke \u002f Frontenac","Chambord \u002f Laurier","Basile-Routhier \u002f Gouin","Parc de Bullion (de Bullion \u002f Prince-Arthur)","St-Jacques \u002f St-Laurent","de la Cath\u00e9drale \u002f Ren\u00e9-L\u00e9vesque","de Gasp\u00e9 \u002f Fairmount","Gordon \u002f Wellington","St-Christophe \u002f Cherrier","Plessis \u002f Ontario","Ste-Claire \u002f Pierre-T\u00e9treault","de Kent \u002f de la C\u00f4te-des-Neiges","Milton \u002f University","Ste-Famille \u002f Sherbrooke","de Chambly \u002f St-Joseph","Benny \u002f de Monkland","de Bordeaux \u002f Jean-Talon","Wilderton  \u002f Van Horne","Chapleau \u002f Terrasse Mercure","St-Nicolas \u002f Place d'Youville","H\u00f4pital Sacr\u00e9-Coeur (Forbes)","Fabre \u002f Beaubien","Dandurand \u002f Papineau","Francis \u002f Fleury","Villeneuve \u002f St-Urbain","de Gasp\u00e9 \u002f Marmier","de Maisonneuve \u002f Greene","M\u00e9tro Atwater (Atwater \u002f Ste-Catherine)","de Maisonneuve \u002f Mansfield (ouest)","Marmier \u002f St-Denis","M\u00e9tro de la Savane (de Sorel \u002f Bougainville)","de la Roche \u002f St-Joseph","du Mont-Royal \u002f Clark","1\u00e8re  avenue \u002f St-Zotique","Br\u00e9beuf \u002f Laurier","Villeray \u002f de Normanville","Louis-H\u00e9bert \u002f Belllechasse","Henri-Julien \u002f du Carmel","35e avenue \u002f Beaubien","Dandurand \u002f Papineau","Parc du Bois-Franc (de l'\u00c9quateur \u002f des Andes)","Parthenais \u002f Laurier","Chabot \u002f du Mont-Royal","Hillside \u002f Ste-Catherine","de Bordeaux \u002f Beaubien","Montcalm \u002f de Maisonneuve","Lajeunesse \u002f Villeray","Place du Commerce","Gary-Carter \u002f St-Laurent","Quai de la navette fluviale","Union \u002f Ren\u00e9-L\u00e9vesque","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","St-Andr\u00e9 \u002f Ste-Catherine","Hamel \u002f Sauv\u00e9","Chabot \u002f Everett","de Hampton \u002f de Monkland","Parc Tino-Rossi (18e avenue \u002f Everett)","Ball \u002f Querbes","McLynn \u002f Plamondon","Crescent \u002f Ren\u00e9-L\u00e9vesque","Hamel \u002f Sauv\u00e9","de Mentana \u002f Rachel","Ste-Catherine \u002f McGill College","March\u00e9 Atwater","Notre-Dame \u002f de la Montagne","Valois \u002f Ontario","Dubois \u002f Eadie","\u00c9douard-Montpetit (Universit\u00e9 de Montr\u00e9al)","Mansfield \u002f Ste-Catherine","du Mont-Royal \u002f Vincent-d'Indy","St-Jacques \u002f St-Laurent","Parc des Rapides (LaSalle \u002f 6e avenue)","Wolfe \u002f Robin","Chambord \u002f Jean-Talon","Notre-Dame \u002f St-Gabriel","Place Longueuil","de Maisonneuve \u002f Mansfield (nord)","Crescent \u002f de Maisonneuve","Ste-Famille \u002f des Pins","de Bullion \u002f St-Joseph","Clark \u002f Evans","Cartier \u002f Rosemont","CIUSSS (du Trianon \u002f Pierre-de-Coubertin)","Legrand \u002f Cartier","Fullum \u002f Sherbrooke","M\u00e9tro Laurier (Rivard \u002f Laurier)","Henri-Julien \u002f Jean-Talon","Place Longueuil","Fleury \u002f Lajeunesse","Biblioth\u00e8que de Verdun (Brown \u002f Bannantyne)","Louis-H\u00e9mon \u002f Villeray","8e avenue \u002f Beaubien","Parc Jean-Drapeau","Jardin Botanique (Pie-IX \u002f Sherbrooke)","Fabre \u002f St-Zotique","Villeneuve \u002f du Parc","Aylwin \u002f Ontario","de Maisonneuve \u002f Aylmer","McKenna \u002f \u00c9douard-Montpetit","Notre-Dame \u002f Chatham","M\u00e9tro Monk (Allard \u002f Beaulieu)","Dandurand \u002f de Lorimier","Clark \u002f Evans","Parc Jeanne Mance (monument \u00e0 sir George-\u00c9tienne Cartier)","Evans \u002f Clark","M\u00e9tro Beaubien (de Chateaubriand \u002f Beaubien)","St-Andr\u00e9 \u002f de Maisonneuve","Ellendale \u002f C\u00f4te-des-Neiges","Omer-Lavall\u00e9e \u002f du Midway","Marquette \u002f Villeray","Ste-Catherine \u002f McGill College","Palm \u002f St-R\u00e9mi","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","St-Urbain \u002f Beaubien","Rivard \u002f Rachel","Benny \u002f de Monkland","Notre-Dame \u002f de la Montagne","16e avenue \u002f St-Joseph","Chabot \u002f du Mont-Royal","d'Orl\u00e9ans \u002f Masson","Logan \u002f d'Iberville","Ellendale \u002f de la C\u00f4te-des-Neiges","Ryde \u002f Charlevoix","de la Salle \u002f Ste-Catherine","Gloria \u002f Dollard","Clark \u002f Fleury","Drake \u002f de S\u00e8ve","Maguire \u002f Henri-Julien","3e avenue \u002f Dandurand","Berri \u002f de Maisonneuve","Molson \u002f Beaubien","Drummond \u002f Ste-Catherine","du Parc-La Fontaine \u002f Roy","de Bordeaux \u002f Jean-Talon","St-Dominique \u002f Bernard","Chabot \u002f Beaubien","Thimens \u002f Alexis-Nihon","Parc Maisonneuve (Viau \u002f Sherbrooke)","Lincoln \u002f du Fort","Laval \u002f Duluth","Lucien L'Allier \u002f St-Jacques","Dandurand \u002f de Lorimier","du Parc-La Fontaine \u002f Roy","Alexandra \u002f Jean-Talon","Marseille \u002f Viau","Wolfe \u002f Robin","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","d'Outremont \u002f Ogilvy","Queen \u002f Wellington","de la Roche \u002f Everett","Tupper \u002f du Fort","Hampton \u002f Monkland","B\u00e9langer \u002f St-Denis","4e avenue \u002f de Verdun","de Bordeaux \u002f Beaubien","Prince-Arthur \u002f St-Urbain","Ontario \u002f du Havre","des \u00c9cores \u002f Jean-Talon","Guizot \u002f St-Laurent","Ernest-Gendreau \u002f du Mont-Royal","Marquette \u002f Jean-Talon","du President-Kennedy \u002f Union","Fullum \u002f Sherbrooke ","7e avenue \u002f St-Joseph Rosemont","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Pontiac \u002f Gilford","Paul Boutet \u002f des Regrattiers","Notre-Dame \u002f Li\u00e9bert","Eadie \u002f Dubois","Bellerive \u002f Duchesneau","Bishop \u002f de Maisonneuve","de Chambly \u002f Rachel","Boyer \u002f St-Zotique","Joseph-Manceau \u002f Ren\u00e9-L\u00e9vesque","Le Caron \u002f Marc-Sauvalle ","16e avenue \u002f St-Joseph","Bishop \u002f Ste-Catherine","de Grosbois \u002f Ducheneau","Marquette \u002f Villeray","McGill \u002f des R\u00e9collets","19e avenue \u002f St-Zotique","Gascon \u002f Rachel","Christophe-Colomb \u002f St-Joseph","Bennett \u002f Ste-Catherine","Louis-H\u00e9mon \u002f Villeray","Lambert-Closse \u002f Tupper","H\u00f4pital g\u00e9n\u00e9ral juif","Coll\u00e8ge \u00c9douard-Montpetit (de Gentilly \u002f de Normandie)","M\u00e9tro Cartier (des Laurentides \u002f Cartier)","Duluth \u002f St-Laurent","Victoria \u002f 18e avenue Lachine","Dorion \u002f Ontario","Chambord \u002f Jean-Talon","Legrand \u002f Laurier","Peel \u002f avenue des Canadiens de Montr\u00e9al","Mackay \u002f Ste-Catherine","Berri \u002f Sherbrooke","Jeanne-Mance \u002f St-Viateur","Louis-H\u00e9bert \u002f Beaubien","St-Charles \u002f Ch\u00e2teauguay","Victoria \u002f de Maisonneuve","Marie-Anne \u002f de la Roche","University \u002f Prince-Arthur","Prince-Arthur \u002f St-Urbain","Poupart \u002f Ontario","Letourneaux \u002f Ste-Catherine","Gilford \u002f de Br\u00e9beuf","Ren\u00e9-L\u00e9vesque \u002f 9e avenue","35e avenue \u002f Beaubien","Place de la paix ( Pl. du march\u00e9 \u002f St-Dominique )","Lajeunesse \u002f Cr\u00e9mazie","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","M\u00e9tro Laurier (Rivard \u002f Laurier)","Aylmer \u002f Prince-Arthur","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","M\u00e9tro Frontenac (Ontario \u002f du Havre)","Elgar \u002f de l'\u00cele-des-S\u0153urs","de la Grande-All\u00e9e \u002f Ste-Catherine","Notre-Dame \u002f Murray","Laval \u002f Rachel","de l'Esplanade \u002f Laurier","Basile-Routhier \u002f Chabanel","Garnier \u002f des Carri\u00e8res","Resther \u002f du Mont-Royal","Sanguinet \u002f Ste-Catherine","Gauvin \u002f Notre-Dame","Regina \u002f Verdun","de la Roche \u002f B\u00e9langer","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Ontario \u002f Sicard","Milton \u002f University","LaSalle \u002f Crawford","de Lanaudi\u00e8re \u002f Marie-Anne","des \u00c9rables \u002f du Mont-Royal","de la Commune \u002f King","Everett \u002f 8e avenue","Hillside \u002f Ste-Catherine","St-Andr\u00e9 \u002f Cherrier","d'Outremont \u002f Ogilvy","de Castelnau \u002f Lajeunesse","Chabot \u002f de Bellechasse","Terrasse Mercure \u002f Fullum","Shearer \u002f Centre","St-Charles \u002f Charlotte","Newman \u002f Senkus","Fullum \u002f St-Joseph","Drolet \u002f Beaubien","Champagneur \u002f Jean-Talon","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Jeanne-d'Arc \u002f Ontario","Poirier \u002f O'Brien","Bordeaux \u002f St-Zotique","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Desjardins \u002f Ontario","Poupart \u002f Ste-Catherine","de Bordeaux \u002f Rachel","de Maisonneuve \u002f Robert-Bourassa","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","du Pr\u00e9sident-Kennedy \u002f McGill College","Boucher \u002f St-Denis","Drummond \u002f de Maisonneuve","Place de la paix ( Pl. du march\u00e9 \u002f St-Dominique )","St-Andr\u00e9 \u002f Laurier","Marquette \u002f Rachel","Island \u002f Centre","du Rosaire \u002f St-Hubert","de la Salle \u002f Ste-Catherine","Mansfield \u002f Ste-Catherine","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","Pontiac \u002f Gilford","Rockland \u002f Lajoie","de l'Eglise \u002f St-Germain","Clark \u002f Fleury","Papineau \u002f Ren\u00e9-L\u00e9vesque","Messier \u002f du Mont-Royal","University \u002f Prince-Arthur","de la Salle \u002f Hochelaga","Pontiac \u002f Gilford","12e avenue \u002f St-Zotique","de la Commune \u002f St-Sulpice","Queen \u002f Wellington","Basile-Routhier \u002f Chabanel","de Bleury \u002f Mayor","Leman \u002f de Chateaubriand","H\u00f4pital g\u00e9n\u00e9ral juif","Valli\u00e8res \u002f St-Laurent","Rockland \u002f Lajoie","4e avenue \u002f de Verdun","Union \u002f Ren\u00e9-L\u00e9vesque","Notre-Dame \u002f Chatham","Terminus Le Carrefour (Terry-Fox \u002f Le Carr)","Calixa-Lavall\u00e9e \u002f Sherbrooke","Ontario \u002f Viau","Parc Highlands (Highlands \u002f LaSalle)","35e avenue \u002f Beaubien","1\u00e8re avenue \u002f Rosemont","des \u00c9rables \u002f B\u00e9langer","Joseph-Manseau \u002f Ren\u00e9-L\u00e9vesque","Angus (Molson \u002f William-Tremblay)","M\u00e9tro Champ-de-Mars (Viger \u002f Sanguinet)","de Maisonneuve \u002f Aylmer (est)","de la Roche \u002f du Mont-Royal","Hudson \u002f de Kent","CHU Ste-Justine (de la C\u00f4te Ste-Catherine \u002f Hudson)","Parc Labelle (Lasalle\u002f Henri Duhamel)","Ren\u00e9-L\u00e9vesque \u002f Papineau","Hillside \u002f Ste-Catherine","Bernard \u002f Clark","St-Andr\u00e9 \u002f Robin","Villeneuve \u002f du Parc","Bel Air \u002f St-Antoine","St-Mathieu \u002fSte-Catherine","McEachran \u002f Van Horne","de Mentana \u002f Laurier","Notre-Dame \u002f St-Gabriel","Park Row O \u002f Sherbrooke","Dumesnil \u002f B\u00e9langer","Dandurand \u002f de Lorimier","Quesnel \u002f Vinet","Henri-Julien \u002f Jean-Talon","Louis-H\u00e9bert \u002f de Bellechasse","Place du Commerce","Duluth \u002f St-Denis","Robert-Bourassa \u002f Ste-Catherine","Ontario \u002f Sicard","de Maisonneuve \u002f de Bleury","Clark \u002f Rachel","Alexandre-de-S\u00e8ve \u002f la Fontaine","Parc P\u00e8re-Marquette (de Lanaudi\u00e8re \u002f de Bellechasse)","de Lanaudi\u00e8re \u002f Laurier","Gauthier \u002f De Lorimier","Bourbonni\u00e8re \u002f du Mont-Royal","de la C\u00f4te St-Antoine \u002f Royal","Durocher \u002f Beaumont","Waverly \u002f St-Viateur","St-Dominique \u002f Villeray","Champagneur \u002f Jean-Talon","Dorion \u002f Ontario","de Maisonneuve \u002f Mansfield (sud)","Ontario \u002f de Lorimier","Grand Boulevard \u002f Sherbrooke","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Chambly \u002f Rachel","Gounod \u002f St-Denis","Louis-Colin \u002f McKenna","de la Commune \u002f Berri","Bercy \u002f Ontario","de St-Vallier \u002f Rosemont","Mansfield \u002f Ren\u00e9-L\u00e9vesque","Bloomfield \u002f Bernard","Bel Air \u002f St-Antoine","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","March\u00e9 Maisonneuve","Laforest \u002f Dudemaine","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","des \u00c9rables \u002f Rachel","Marie-Anne \u002f de la Roche","Papineau \u002f Ren\u00e9-L\u00e9vesque","LaSalle \u002f Godin","C\u00e9gep Andr\u00e9-Laurendeau","De la Commune \u002f King","Chabot \u002f Everett","Beatty \u002f de Verdun","Calixa-Lavall\u00e9e \u002f Rachel","Ottawa \u002f St-Thomas","10e avenue \u002f Masson","Victoria \u002f 18e avenue Lachine","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Louis-H\u00e9bert \u002f Jean-Talon","Berri \u002f de Maisonneuve","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","Lajeunesse \u002f Jean-Talon","du Champ-de-Mars \u002f Gosford","Eadie \u002f Dubois","St-Dominique \u002f Rachel","Ste-Famille \u002f des Pins","Larivi\u00e8re \u002f de Lorimier","Charlevoix \u002f Lionel-Groulx","Boyer \u002f B\u00e9langer","du Mont-Royal \u002f Clark","Hutchison \u002f Sherbrooke","Elsdale\u002f Louis-Hebert","Drolet \u002f St-Zotique","Ropery \u002f Augustin-Cantin","Chapleau \u002f du Mont-Royal","Georges-Baril \u002f Fleury","Ball \u002f Querbes","Porte-de-Qu\u00e9bec \u002f St-Hubert","Pierre-de-Coubertin \u002f Langelier","Bel Air \u002f St-Antoine","Hochelaga \u002f Chapleau","Cartier \u002f Masson","Bloomfield \u002f Van Horne","Chabot \u002f Beaubien","de Bordeaux \u002f Sherbrooke","Casgrain \u002f St-Viateur","Place \u00c9milie-Gamelin (St-Hubert \u002f de Maisonneuve)","St-Joseph \u002f 8e avenue Lachine","Square St-Louis","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","Ren\u00e9-L\u00e9vesque \u002f Papineau","de Br\u00e9beuf \u002f Laurier","Gascon \u002f Rachel","5e avenue \u002f Rosemont","de Laprairie \u002f Centre","St-Jacques \u002f des Seigneurs","St-Alexandre \u002f Ste-Catherine","Fleury \u002f Lajeunesse","Parc Rosemont (Dandurand \u002f d'Iberville)","Metcalfe \u002f Ste-Catherine","Coolbrook \u002f Queen-Mary","Stinson \u002f Montpellier","Dollard \u002f Bernard","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Coll\u00e8ge \u00c9douard-Montpetit","Union \u002f Ren\u00e9-L\u00e9vesque","Jeanne-Mance \u002f Bernard","M\u00e9tro Namur (des Jockeys \u002f D\u00e9carie)","de Br\u00e9beuf \u002f du Mont-Royal","Guy \u002f Notre-Dame","du Havre \u002f de Rouen","de Gasp\u00e9 \u002f Jarry","Fairmount \u002f St-Dominique","d'Orl\u00e9ans \u002f Masson","Cartier \u002f Rosemont","2e avenue \u002f Centrale","Dorion \u002f Ontario","Roy \u002f St-Christophe","Fullum \u002f Gilford","March\u00e9 Maisonneuve","St-Hubert \u002f Laurier","Bishop\u002f Ren\u00e9-L\u00e9vesque","de Maisonneuve \u002f Mansfield (sud)","Benny \u002f Monkland","de l'\u00c9glise \u002f Bannantyne","Lajeunesse \u002f Jarry","Waverly \u002f St-Zotique","Augustin-Cantin \u002f Shearer","Gordon \u002f Wellington","Grand Boulevard \u002f de Terrebonne","West Broadway \u002f Sherbrooke","Guizot \u002f St-Denis","Roy \u002f St-Christophe","Place du Commerce","Wilson \u002f Sherbrooke","St-Viateur\u002fParc","Willibrord \u002f Verdun","3e avenue \u002f Holt","Hutchison \u002f des Pins","Ste-Catherine \u002f Labelle","Metcalfe \u002f St-Catherine","Duluth \u002f St-Laurent","Plessis \u002f Ren\u00e9-L\u00e9vesque","Duke \u002f Brennan","Chambord \u002f Fleury","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","Lesp\u00e9rance \u002f de Rouen","de Kent \u002f Victoria","St-Mathieu \u002f Ste-Catherine","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Boyer \u002f du Mont-Royal","LaSalle \u002f Crawford","de Gasp\u00e9 \u002f Jarry","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Marmier \u002f St-Denis","D\u00e9z\u00e9ry \u002f Ontario","Mozart \u002f St-Laurent","Mansfield \u002f Sherbrooke","Cartier \u002f St-Joseph","Ste-Catherine \u002f Labelle","Parc St-Paul (Le Caron \u002f Marc-Sauvalle)","St-Maurice \u002f Robert-Bourassa","Guizot \u002f St-Laurent","Drolet \u002f Villeray","St-Andr\u00e9 \u002f Cherrier","de l'Esplanade \u002f Fairmount","de la Commune \u002f Place Jacques-Cartier","Hamel \u002f Sauv\u00e9","Bel Air \u002f St-Antoine","Nicolet \u002f Sherbrooke","M\u00e9tro Lasalle (de Rushbrooke \u002f Caisse)","Maguire \u002f Henri-Julien","Boyer \u002f Jean-Talon","Biblioth\u00e8que de Verdun (Brown \u002f Bannantyne)","Francis \u002f Fleury","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","M\u00e9tro Georges-Vanier (St-Antoine \u002f Canning)","des \u00c9rables \u002f Sherbrooke","St-Dominique \u002f Ren\u00e9-L\u00e9vesque","Marie-Anne \u002f de la Roche","de Bordeaux \u002f Sherbrooke","Faillon \u002f St-Hubert","8e avenue \u002f Notre-Dame Lachine","Alexandra \u002f Waverly","St-Dominique \u002f Rachel","Marquette \u002f St-Gr\u00e9goire","Casgrain \u002f Maguire","Henri-Julien \u002f de Castelneau","St-Jacques \u002f des Seigneurs","15e avenue \u002f Masson","Clark \u002f Prince-Arthur","Waverly \u002f St-Zotique","Parc Beaulac (rue \u00c9lizabeth)","de Mentana \u002f Marie-Anne","Basile-Routhier \u002f Gouin","Duvernay \u002f Charlevoix","Bourgeoys \u002f Favard","Louis-H\u00e9mon \u002f Villeray","Lionel-Groulx \u002f George-Vanier","Lacombe \u002f Victoria","Gascon \u002f Rachel","de la Roche \u002f Everett","Parc Ottawa (de Belleville \u002f Fleury)","de la Roche \u002f du Mont-Royal","Rachel \u002f de Br\u00e9beuf","Hickson \u002f Wellington","Basin \u002f Richmond","de Monkland \u002f Girouard","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","1\u00e8re  avenue \u002f St-Zotique","Quesnel \u002f Vinet","Egan \u002f Laurendeau","McTavish \u002f des Pins","Duluth \u002f de l'Esplanade","Ste-Catherine \u002f St-Marc","Duvernay \u002f Charlevoix","C\u00f4te St-Antoine \u002f Clarke","Sherbrooke \u002f Frontenac","M\u00e9tro Laurier (Rivard \u002f Laurier)","Darling \u002f Sherbrooke","Dunlop \u002f Van Horne","de la Gaucheti\u00e8re \u002f Mansfield","Union \u002f Ren\u00e9-L\u00e9vesque","Rushbrooke \u002f  Caisse","McTavish \u002f des Pins","Square Ahmerst \u002f Wolfe","Rockland \u002f Lajoie","St-Viateur \u002f St-Laurent","St-Dominique \u002f Ren\u00e9-L\u00e9vesque","St-Jacques \u002f McGill","Gary-Carter \u002f St-Laurent","12e avenue \u002f St-Zotique","U. Concordia - Campus Loyola (Sherbrooke \u002f West Broadway)","Milton \u002f du Parc","Hogan \u002f Ontario","Lajeunesse \u002f Jarry","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Paul Boutet \u002f des Regrattiers","Ontario \u002f Sicard","University \u002f Prince-Arthur","Berri \u002f Cherrier","de Lanaudi\u00e8re \u002f B\u00e9langer","Complexe sportif Claude-Robillard","12e avenue \u002f St-Zotique","Briand \u002f le Caron","Boyer \u002f Jarry","Parc St-Laurent (Salk \u002f Thomas-Chapais)","Roy \u002f St-Denis","Parc St-Laurent (Dutrisac)","Jogues \u002f Allard","Bernard \u002f Jeanne-Mance","St-Andr\u00e9 \u002f Robin","Peel \u002f avenue des Canadiens de Montr\u00e9al","Sanguinet \u002f de Maisonneuve","Marquette \u002f du Mont-Royal","Henri-Julien \u002f de Castelneau","Berri \u002f St-Gr\u00e9goire","M\u00e9tro du Coll\u00e8ge (Cartier \u002f D\u00e9carie)","Molson \u002f William-Tremblay","Parthenais \u002f Laurier","Augustin-Cantin \u002f Shearer","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Duluth \u002f de l'Esplanade","Prince-Arthur \u002f Ste-Famille","Place du Canada (Peel \u002f de la Gaucheti\u00e8re)","Parc Jeanne Mance (monument sir George-Etienne Cartier)","Alexandre-de-S\u00e8ve \u002f la Fontaine","Desjardins \u002f Hochelaga","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","de Bellechasse \u002f de St-Vallier","Desjardins \u002f Ontario","Drolet \u002f Laurier","Fortune\u002fwellington","Lionel-Groulx \u002f George-Vanier","Duluth \u002f de l'Esplanade","Ste-\u00c9milie \u002f Sir-Georges-Etienne-Cartier","Rockland \u002f Lajoie","Marmier \u002f St-Denis","St-Dominique \u002f Laurier","Fabre \u002f Beaubien","Cartier \u002f Gauthier","Garnier \u002f du Mont-Royal","Milton \u002f Clark","William-Tremblay \u002f Molson","des Seigneurs \u002f Notre-Dame","St-Andr\u00e9 \u002f Laurier","St-Dominique \u002f Maguire","Cartier \u002f B\u00e9langer","Guizot \u002f St-Laurent","Poupart \u002f Ste-Catherine","Lucien L'Allier \u002f St-Jacques","LaSalle \u002f 4e avenue","Legendre \u002f Henri-Julien","Milton \u002f Clark","Darling \u002f Ste-Catherine","Bercy \u002f Ontario","de Maisonneuve \u002f Stanley","Peel \u002f avenue des Canadiens de Montr\u00e9al","Ontario \u002f du Havre","Lesp\u00e9rance \u002f Rouen","Fullum \u002f St-Joseph","Terrasse Guindon \u002f Fullum","Marquette \u002f Fleury","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","Boyer \u002f Jarry","Nicolet \u002f Ste-Catherine","Girouard \u002f de Terrebonne","Waverly \u002f St-Viateur","Bishop \u002f Ste-Catherine","Boyer \u002f B\u00e9langer","M\u00e9tro Rosemont (de St-Vallier \u002f Rosemont)","M\u00e9tro Sauv\u00e9 (Berri \u002f Sauv\u00e9)","Nazareth \u002f William","Metcalfe \u002f Square Dorchester","de la C\u00f4te St-Paul \u002f St-Ambroise","Square Sir-Georges-\u00c9tienne-Cartier \u002f St-Ambroise","de Kent \u002f de la C\u00f4te-des-Neiges","des \u00c9rables \u002f B\u00e9langer","St-Andr\u00e9 \u002f Ontario","Resther \u002f du Mont-Royal","16e avenue \u002f St-Joseph","M\u00e9tro Lasalle (de Rushbrooke \u002f Caisse)","4e avenue \u002f Masson","Bourgeoys \u002f Favard","Parc Jeanne Mance (monument George-\u00c9tienne Cartier)","Hutchison \u002f Van Horne","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Quai de la navette fluviale","de l'Esplanade \u002f Laurier","Parc Henri-Julien (Legendre \u002f Henri-Julien)","Angus (Molson \u002f William-Tremblay)","Parc riverain de Lachine (des Iroquois)","Calixa-Lavall\u00e9e \u002f Rachel","Tupper \u002f Atwater","Parc Kent (de Kent \u002f Hudson)","Du Quesne \u002f Pierre-de-Coubertin","Ste-Catherine \u002f Sanguinet","Marquette \u002f Fleury","Valli\u00e8res \u002f St-Dominique","Fleury \u002f Lajeunesse","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","de l'Esplanade \u002f Fairmount","M\u00e9tro Cadillac (Sherbrooke \u002f de Cadillac)","Villeneuve \u002f St-Urbain","du Mont-Royal \u002f Augustin-Frigon","Desjardins \u002f Ontario","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","Hamilton \u002f Jolicoeur","1\u00e8re avenue \u002f Rosemont","Victoria \u002f de Maisonneuve","de Bleury \u002f de Maisonneuve","Hutchison \u002f Van Horne","Waverly \u002f Van Horne","Bernard \u002f Jeanne-Mance","Bernard \u002f Clark","16e avenue \u002f Beaubien","Logan \u002f d'Iberville","Island \u002f Centre","Parc Plage","M\u00e9tro Laurier (Berri \u002f St-Joseph)","M\u00e9tro Snowdon (de Westbury \u002f Queen-Mary)","St-Hubert \u002f Roy","Guizot \u002f St-Denis","Garnier \u002f St-Joseph","Clark \u002f Rachel","des Seigneurs \u002f Notre-Dame","Notre-Dame \u002f Chatham","Omer-Lavall\u00e9e \u002f Midway","Monkland \u002f Girouard","Gare d'autocars de Montr\u00e9al (Berri \u002f Ontario)","Chabot \u002f Everett","Mentana \u002f Marie-Anne","du Mont-Royal \u002f Clark","de Champlain \u002f Ontario","des \u00c9rables \u002f Sherbrooke","2e avenue \u002f Jean-Rivard","Boyer \u002f B\u00e9langer","Villeneuve \u002f St-Denis","Girouard \u002f de Terrebonne","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Coloniale \u002f du Mont-Royal","St-Andr\u00e9 \u002f Duluth","St-Jacques \u002f Gauvin","Parc St-Claude (7e avenue \u002f 8e rue)","du Havre \u002f Hochelaga","Parthenais \u002f St-Joseph","Gounod \u002f St-Denis","Highlands \u002f LaSalle","de Kent \u002f Victoria","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","Georges-Baril \u002f Fleury","de la Salle \u002f Ste-Catherine","Marcil \u002f Sherbrooke","Boyer \u002f Jarry","Alma \u002f Beaubien","Jean Langlois \u002f Fullum","Alma \u002f Beaubien","de la Gaucheti\u00e8re \u002f Robert-Bourassa","St-Jacques \u002f St-Pierre","Palm \u002f St-Remi","Louis-H\u00e9mon \u002f Villeray","Garnier \u002f St-Gr\u00e9goire","St-Jean-Baptiste \u002f Ren\u00e9-L\u00e9vesque","de Maisonneuve \u002f Stanley","Boyer \u002f St-Zotique","Argyle \u002f de Verdun","4e avenue \u002f Masson","de Chateaubriand \u002f Beaubien","Jeanne-Mance \u002f St-Viateur","Roy \u002f St-Denis","Ste-Famille \u002f Sherbrooke","Wilderton  \u002f Van Horne","3e avenue \u002f Dandurand","Sanguinet \u002f de Maisonneuve","Fullum \u002f Sherbrooke ","Ontario \u002f Sicard","Beurling \u002f Godin","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Logan \u002f d'Iberville","Montcalm \u002f de Maisonneuve","2e avenue \u002f Jean-Rivard","Robin \u002f de la Visitation","Hogan \u002f Ontario","Stuart \u002f de la C\u00f4te-Ste-Catherine","Boyer \u002f Beaubien","Pierre-de-Coubertin \u002f Aird","10e avenue \u002f Masson","St-Charles \u002f St-Sylvestre","Ville-Marie \u002f Ste-Catherine","Parc Jeanne Mance (monument sir George-Etienne Cartier)","du Rosaire \u002f St-Hubert","St-Catherine \u002f St-Laurent","de la Commune \u002f Berri","Robert-Bourassa \u002f de Maisonneuve","Fullum \u002f Gilford","La Ronde","de Lanaudi\u00e8re \u002f Gilford","Boyer \u002f B\u00e9langer","Francis \u002f Fleury","de Repentigny \u002f Sherbrooke","Villeneuve \u002f St-Laurent","du Pr\u00e9sident-Kennedy \u002f Jeanne-Mance","Terrasse Guindon \u002f Fullum","Marcil \u002f Sherbrooke","des Seigneurs \u002f Notre-Dame","Cartier \u002f B\u00e9langer","Notre-Dame \u002f Peel","Durocher \u002f Bernard","Coloniale \u002f du Mont-Royal","Logan \u002f Fullum","de Bordeaux \u002f Sherbrooke","Aylmer \u002f Prince-Arthur","Gounod \u002f St-Denis","Rosemont \u002f Viau","Parc Howard (de l'\u00c9p\u00e9e \u002f de Li\u00e8ge)","Sherbrooke \u002f Frontenac","de la Rotonde \u002f Jacques-Le Ber","1\u00e8re avenue \u002f Rosemont","De la Commune \u002f King","Prince-Arthur \u002f du Parc","Chabot \u002f Everett","Boyer \u002f B\u00e9langer","Atwater \u002f Greene","Parc Luigi-Pirandello (de Bretagne \u002f Jean-Rivard)","de Maisonneuve \u002f Aylmer","Drolet \u002f Gounod","de la Gaucheti\u00e8re \u002f de Bleury","Pierre-de-Coubertin \u002f Louis-Veuillot","Berlioz \u002f de l'\u00cele des Soeurs","Smith \u002f Peel","LaSalle \u002f S\u00e9n\u00e9cal","Sherbrooke \u002f Frontenac","Parc Plage","Prince-Arthur \u002f Ste-Famille","de Bordeaux \u002f Marie-Anne","Ste-\u00c9milie \u002f Sir-Georges-Etienne-Cartier","Duvernay \u002f Charlevoix","Angus (Molson \u002f William-Tremblay)","Marie-Anne \u002f de la Roche","Boyer \u002f Rosemont","M\u00e9tro C\u00f4te-des-Neiges (Jean-Brillant \u002f de la C\u00f4te-des-Neiges)","Vezina \u002f Victoria","Ann \u002f William","de Bordeaux \u002f Rachel","Stanley \u002f du Docteur-Penfield","Angus (Molson \u002f William-Tremblay)","St-Andr\u00e9 \u002f Duluth","Drake \u002f de S\u00e8ve","Ste-Catherine \u002f St-Denis","18e avenue \u002f Rosemont","Muray \u002f Notre-Dame","Mansfield \u002f Ste-Catherine","Casgrain \u002f de Bellechasse","Chabot \u002f du Mont-Royal","1\u00e8re avenue \u002f Masson","Parc Noel-Nord (Toupin \u002f Keller)","Hogan \u002f Ontario","Calixa-Lavall\u00e9e \u002f Rachel","Notre-Dame \u002f St-Gabriel","Atwater \u002f Greene","Ste-Catherine \u002f St-Laurent","18e avenue \u002f Rosemont","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","M\u00e9tro Beaubien (de Chateaubriand \u002f Beaubien)","Hogan \u002f Ontario","Wilderton  \u002f Van Horne","Richardson \u002f de Montmorency","M\u00e9tro Sauv\u00e9 (Sauv\u00e9 \u002f Berri)","Drolet \u002f Beaubien","Aylwin \u002f Ontario","de Gasp\u00e9 \u002f Jarry","Henri-Julien \u002f de Castelneau","St-Hubert \u002f Rachel","Guizot \u002f St-Denis","Square Phillips 2","M\u00e9tro Angrignon (Lamont \u002f  des Trinitaires)","Louis H\u00e9mon \u002f Rosemont","du Pr\u00e9sident-Kennedy \u002f McGill College","de Maisonneuve \u002f Greene","41e avenue \u002f Victoria","16e avenue \u002f Beaubien","Metcalfe \u002f Ste-Catherine","St-Jacques \u002f Guy","Crescent \u002f Ste-Catherine","Cypress \u002f Peel","Laporte \u002f Guay","Ontario \u002f Viau","Milton \u002f Durocher","Bernard \u002f Clark","des Ormeaux \u002f de Grosbois","Clark \u002f St-Viateur","Hamel \u002f Sauv\u00e9","Notre-Dame \u002f de la Montagne","10e avenue \u002f Masson","Sanguinet \u002f Ontario","Louis-H\u00e9bert \u002f St-Zotique","Drolet \u002f St-Zotique","Metcalfe \u002f St-Catherine","St-Dominique \u002f Jean-Talon","Casgrain \u002f Maguire","des \u00c9rables \u002f Sherbrooke","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Villeneuve \u002f du Parc","Place Rodolphe-Rousseau (Gohier \u002f \u00c9douard-Laurin)","Gary-Carter \u002f St-Laurent","Cartier \u002f Rosemont","M\u00e9tro Rosemont (Rosemont \u002f de St-Vallier)","Square Phillips","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","Hamilton \u002f Jolicoeur","du Havre \u002f de Rouen","Drolet \u002f Faillon","d'Orl\u00e9ans \u002f Hochelaga","M\u00e9tro de la Savane (de Sorel \u002f Bougainville)","Hutchison \u002f des Pins","St-Jacques \u002f St-Pierre","Ryde \u002f Charlevoix","Robin \u002f de la Visitation","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Robin \u002f de la Visitation","St-Dominique \u002f St-Viateur","de la Commune \u002f Berri","Park Row O \u002f Sherbrooke","Parc Painter (Marcotte \u002f Quenneville)","Labadie \u002f du Parc","St-Dominique \u002f Bernard","M\u00e9tro Lionel-Groulx (Atwater \u002f Lionel-Groulx)","16e avenue \u002f Beaubien","Atwater \u002f Sherbrooke","de Bullion \u002f St-Joseph","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Wolfe \u002f Robin","Gary Carter \u002f St-Laurent","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Mansfield \u002f Sherbrooke","Chabot \u002f St-Joseph","de Montmorency \u002f Richardson","M\u00e9tro C\u00f4te-des-Neiges (Jean-Brillant \u002f de la C\u00f4te-des-Neiges)","Hutchison \u002f Sherbrooke","Villeneuve \u002f du Parc","Durocher \u002f Bernard","Robin \u002f de la Visitation","de Bullion \u002f du Mont-Royal","Rivard \u002f Rachel","Roy \u002f St-Denis","Berri \u002f St-Gr\u00e9goire","de la Salle \u002f Hochelaga","Cartier \u002f Rosemont","du Parc-La Fontaine \u002f Roy","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","St-Hubert \u002f Laurier","de Kent \u002f de la C\u00f4te-des-Neiges","Quesnel \u002f Vinet","Place Longueuil","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","Duke \u002f Brennan","Duluth \u002f St-Denis","Cathcart \u002f Union","Gagne \u002f LaSalle","de Gasp\u00e9 \u002f St-Viateur","Biblioth\u00e8que du Bois\u00e9 (Thimens \u002f Todd)","M\u00e9tro Vend\u00f4me (de Marlowe \u002f de Maisonneuve)","LaSalle \u002f Crawford","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Clark \u002f de Li\u00e8ge","Parthenais \u002f Laurier","Mansfield \u002f Sherbrooke","M\u00e9tro Lionel-Groulx (Lionel-Groulx \u002f Charlevoix)","Drolet \u002f Jarry","Park Row O \u002f Sherbrooke","Bloomfield \u002f Van Horne","de Bellechasse \u002f de St-Vallier","Fran\u00e7ois-Boivin \u002f B\u00e9langer","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Montcalm \u002f de Maisonneuve","Ellendale \u002f de la C\u00f4te-des-Neiges","de Gasp\u00e9 \u002f Jarry","la Fontaine \u002f Alexandre-de-S\u00e8ve","Parc du P\u00e9lican (1\u00e8re avenue \u002f Masson)","Clark \u002f Rachel","Parc Plage","5e avenue \u002f Bannantyne","Hutchison \u002f Beaumont","Pierre-de-Coubertin \u002f Langelier","Marquette \u002f Gilford","10e avenue \u002f Masson","Parc Wilfrid-Bastien (Lacordaire \u002f des Galets)","Metcalfe \u002f de Maisonneuve","Parc Jean-Brillant (Swail \u002f Decelles)","M\u00e9tro Lionel-Groulx (Atwater \u002f Lionel-Groulx)","Lucien L'Allier \u002f St-Jacques","Chabot \u002f du Mont-Royal","McGill \u002f des R\u00e9collets","Marquette \u002f Laurier","Boyer \u002f Jean-Talon","Cit\u00e9 des Arts du Cirque (Paul Boutet \u002f des Regrattiers)","5e avenue \u002f Bannantyne","St-Denis \u002f de Maisonneuve","Tolhurst \u002f Fleury","10e avenue \u002f Holt","Marcil \u002f Sherbrooke","Fortune \u002f Wellington","Gilford \u002f des \u00c9rables","St-Zotique \u002f 39e avenue","Casgrain \u002f Mozart","Mansfield \u002f Ren\u00e9-L\u00e9vesque","de Bordeaux \u002f Beaubien","Rousselot \u002f Villeray","St-Viateur \u002f St-Laurent","Fabre \u002f Beaubien","Louis-H\u00e9bert \u002f B\u00e9langer","Laval \u002f du Mont-Royal","Fairmount \u002f St-Dominique","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","Joliette \u002f Ste-Catherine","Chabanel \u002f du Parc","Jean Langlois \u002f Fullum","St-Jacques \u002f St-Pierre","Coll\u00e8ge \u00c9douard-Montpetit","Le Caron \u002f Marc-Sauvalle ","de Maisonneuve \u002f Mansfield","Georges-Vanier \u002f Notre-Dame","University \u002f Prince-Arthur","St-Zotique \u002f Clark","Georges-Vanier \u002f Notre-Dame","Parc Rosemont (Dandurand \u002f d'Iberville)","de l'\u00c9p\u00e9e \u002f Jean-Talon","Ross \u002f de l'\u00c9glise","D\u00e9z\u00e9ry \u002f Ontario","Logan \u002f Fullum","de la Commune \u002f Berri","Maguire \u002f St-Laurent","Clark \u002f Rachel","de Bordeaux \u002f Sherbrooke","Waverly \u002f Van Horne","Graham \u002f Wicksteed","Roy \u002f St-Laurent","Parc Marlborough (Beauz\u00e8le \u002f Robichaud)","Marie-Anne \u002f de la Roche","Villeneuve \u002f St-Laurent","St-Dominique \u002f Gounod","de l'\u00c9glise \u002f de Verdun","St-Charles \u002f Grant","de Gasp\u00e9 \u002f Fairmount","Laurier \u002f de Bordeaux","Montcalm \u002f Ontario","St-Joseph \u002f 21e avenue Lachine","du Mont-Royal \u002f du Parc","Valli\u00e8res \u002f St-Dominique","de Lanaudi\u00e8re \u002f B\u00e9langer","Marcil \u002f Sherbrooke","Ste-Catherine \u002f St-Hubert","St-Mathieu \u002f Sherbrooke","Duvernay \u002f Charlevoix","Wolfe \u002f Robin","de Maisonneuve \u002f de Bleury","Garnier \u002f du Mont-Royal","Hutchison \u002f Prince-Arthur","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","5e avenue \u002f Bannantyne","Prince-Arthur \u002f St-Urbain","St-Andr\u00e9 \u002f Ste-Catherine","Young \u002f Wellington","St-Andr\u00e9 \u002f St-Gr\u00e9goire","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Marquette \u002f Jean-Talon","Li\u00e9bert \u002f Hochelaga","\u00c9douard-Montpetit \u002f de Stirling","M\u00e9tro George-Vanier (St-Antoine \u002f Canning)","Berri \u002f de Maisonneuve","Drummond \u002f Ste-Catherine","du Mont-Royal \u002f Augustin-Frigon","Poupart \u002f Ontario","M\u00e9tro Lucien-L'Allier (Lucien l'Allier \u002f Argyle)","Ontario \u002f du Havre","St-Andr\u00e9 \u002f Ontario","LaSalle \u002f 37e avenue","M\u00e9tro Place-des-Arts (de Maisonneuve \u002f de Bleury)","Elsdale\u002f Louis-Hebert","Gilford \u002f Br\u00e9beuf","de Bellechasse \u002f de St-Vallier","Bordeaux \u002fGilford","Par\u00e9 \u002f Mountain Sights","St-Denis \u002f de Maisonneuve","de Li\u00e8ge \u002f Lajeunesse","Bourgeoys \u002f Favard","Bloomfield \u002f Bernard","McGill College \u002f Ste-Catherine","Aylmer \u002f Prince-Arthur","Parc Summerlea (53e avenue \u002f St-Joseph)","Parthenais \u002f Ste-Catherine","Terrasse Mercure \u002f Fullum","Eadie\u002fDubois","Logan \u002f de Champlain","Parc Outremont (Bloomfield \u002f Elmwood)","M\u00e9tro \u00c9douard-Montpetit (du Mont-Royal \u002f Vincent-d'Indy)","du Mont-Royal \u002f Clark","Guilbault \u002f Clark","de Lanaudi\u00e8re \u002f Gilford","CHU Sainte-Justine (de la C\u00f4te Sainte-Catherine \u002f Hudson)","Laval \u002f du Mont-Royal","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Fortune \u002f Wellington","St-Charles \u002f Montarville","de Maisonneuve \u002f Stanley","6e avenue \u002f Rosemont","Hochelaga \u002f Chapleau","M\u00e9tro Monk (Allard \u002f Beaulieu)","Hudson \u002f de Kent","Mackay \u002f Ste-Catherine","Waverly \u002f Van Horne","Fullum \u002f Sherbrooke ","Mozart \u002f St-Laurent","Travaux publics LaSalle (Cordner \u002f L\u00e9ger)","Casgrain \u002f Mozart","H\u00f4pital g\u00e9n\u00e9ral juif","St-Urbain \u002f de la Gaucheti\u00e8re","de Gasp\u00e9 \u002f de Li\u00e8ge","Quai de la navette fluviale","March\u00e9 Maisonneuve","St-Maurice \u002f Robert-Bourassa","Marquette \u002f des Carri\u00e8res","de Gasp\u00e9 \u002f Dante","Rousselot \u002f Jarry","Berri \u002f Cherrier","Fortune \u002f Wellington","Ryde \u002f Charlevoix","West Broadway \u002f Sherbrooke","Queen \u002f Ottawa","Guy \u002f Notre-Dame","de Bordeaux \u002f Sherbrooke","Sanguinet \u002f Ste-Catherine","LaSalle \u002f 90e avenue","St-Ambroise \u002f Louis-Cyr","Omer-Lavall\u00e9e \u002f du Midway","de Gasp\u00e9 \u002f de Li\u00e8ge","18e avenue \u002f Rosemont","Jacques-Le Ber \u002f de la Pointe Nord","Grand Trunk \u002f Hibernia","Jardin Botanique (Pie-IX \u002f Sherbrooke)","de Maisonneuve \u002f Aylmer","Sanguinet \u002f Ste-Catherine","du Mont-Royal \u002f Clark","28e avenue \u002f Rosemont","Rouen \u002f Fullum","M\u00e9tro Radisson (Sherbrooke \u002f des Groseilliers)","Clark \u002f Guilbault","St-Nicolas \u002f Place d'Youville","Belmont \u002f Beaver Hall","Garnier \u002f St-Joseph","Bishop \u002f de Maisonneuve","Louis-H\u00e9bert \u002f B\u00e9langer","Boyer \u002f Rosemont","Ste-Catherine \u002f St-Marc","St-Mathieu \u002fSte-Catherine","Benny \u002f Sherbrooke","Faillon \u002f St-Hubert","M\u00e9tro Jean-Talon (de Chateaubriand \u002f Jean-Talon)","Valois \u002f Ontario","Queen \u002f Ottawa","Eadie \u002f Dubois","Peel \u002f de la Gauchetiere","Desjardins \u002f Ontario","Wilderton  \u002f Van Horne","8e avenue \u002f Beaubien","du Souvenir \u002f Chomedey","1\u00e8re avenue \u002f Rosemont","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Cartier \u002f Rosemont","Natatorium (LaSalle \u002f Rolland)","d'Orl\u00e9ans \u002f Masson","Maguire \u002f Henri-Julien","Nicolet \u002f Ste-Catherine","Lincoln \u002f du Fort","Clark \u002f St-Zotique","M\u00e9tro Champ-de-Mars (Viger \u002f Sanguinet)","Chambord \u002f Jean-Talon","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","Bourbonni\u00e8re \u002f du Mont-Royal","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","St-Andr\u00e9 \u002f St-Antoine","Boyer \u002f du Mont-Royal","Bossuet \u002f Pierre-de-Coubertin","de Normanville \u002f Villeray","Metcalfe \u002f de Maisonneuve","CHSLD Benjamin-Victor-Rousselot (Dickson \u002f Sherbrooke)","de Belleville \u002f Fleury","Mozart \u002f St-Laurent","Rachel \u002f de Br\u00e9beuf","St-Dominique \u002f St-Zotique","30e avenue \u002f St-Zotique","Drolet \u002f Beaubien","de St-Vallier \u002f St-Zotique","Thimens \u002f Alexis-Nihon","de la Salle \u002f Ste-Catherine","Roy \u002f St-Denis","Messier \u002f du Mont-Royal","Monsabr\u00e9 \u002f Boileau","Napol\u00e9on \u002f  St-Dominique","Metcalfe \u002f de Maisonneuve","St-Andr\u00e9 \u002f St-Gr\u00e9goire","Young \u002f Wellington","Cartier \u002f B\u00e9langer","18e avenue \u002f Rosemont","Desjardins \u002f Hochelaga","Benny \u002f Monkland","Marquette \u002f Villeray","Jacques-Le Ber \u002f de la Pointe Nord","10e Avenue \u002f Rosemont","Casino de Montr\u00e9al","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Robin \u002f de la Visitation","Larivi\u00e8re \u002f de Lorimier","Willibrord \u002f Verdun","St-Viateur \u002f St-Laurent","Cathcart \u002f Place Phillips","Jardin Botanique (Pie-IX \u002f Sherbrooke)","M\u00e9tro Georges-Vanier (St-Antoine \u002f Canning)","Villeneuve \u002f St-Urbain","Gauthier \u002f Parthenais","Terrasse Mercure \u002f Fullum","Gilford \u002f de Lanaudi\u00e8re","Chapleau \u002f du Mont-Royal","Marquette \u002f St-Gr\u00e9goire","Parc d'Antioche (Place d'Antioche \u002f St-Zotique)","1\u00e8re avenue \u002f Masson","Square Victoria","St-Zotique \u002f Clark","Basin \u002f Richmond","M\u00e9tro Snowdon (de Westbury \u002f Queen-Mary)","Jacques-Le Ber \u002f de la Pointe Nord","M\u00e9tro Place-des-Arts (du Pr\u00e9sident-Kennedy \u002f Jeanne-Mance)","Villeneuve \u002f du Parc","Marquette \u002f Laurier","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","M\u00e9tro Atwater (Atwater \u002f Ste-Catherine)","M\u00e9tro Villa-Maria (D\u00e9carie \u002f de Monkland)","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Grand Boulevard \u002f de Terrebonne","29e avenue \u002f St-Zotique","Grand Boulevard \u002f Sherbrooke","de l'Esplanade \u002f Laurier","M\u00e9tro Sauv\u00e9 (Sauv\u00e9 \u002f Berri)","Parc des Hirondelles (Fleury \u002f Andr\u00e9-Jobin)","Argyle \u002f Bannantyne","Argyle \u002f Bannantyne","Ste-Catherine \u002f St-Hubert","Davaar \u002f de la C\u00f4te-Ste-Catherine","Beaudry \u002f Sherbrooke","Guizot \u002f St-Laurent","Milton \u002f Durocher","Valli\u00e8res \u002f St-Laurent","Darling \u002f Sherbrooke","Notre-Dame \u002f Peel","26e avenue \u002f Beaubien","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Chomedey \u002f de Maisonneuve","Gauthier \u002f de Lorimier","M\u00e9tro Snowdon (Dornal \u002f Westbury)","Elsdale \u002f Louis-H\u00e9bert","de Kent \u002f de la C\u00f4te-des-Neiges","Pontiac \u002f Gilford","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","Montcalm \u002f Ontario","de St-Vallier \u002f St-Zotique","Garnier \u002f du Mont-Royal","Ottawa \u002f William","Gauthier \u002f De Lorimier","Resther \u002f St-Joseph","Duluth \u002f St-Laurent","Hutchison \u002f Van Horne","Augustin-Cantin \u002f Shearer","Place de la paix (Pl. du march\u00e9 \u002f St-Dominique)","de Gasp\u00e9 \u002f Jean-Talon","Aylwin \u002f Ontario","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","M\u00e9tro Villa-Maria (D\u00e9carie \u002f de Monkland)","de Soissons \u002f de Darlington","Fleury \u002f Lajeunesse","Cypress \u002f Peel","Lambert-Closse \u002f Tupper","Parc St-Henri (Laporte \u002f Guay)","de l'Esplanade \u002f du Mont-Royal","Drolet \u002f St-Zotique","M\u00e9tro Laurier (Rivard \u002f Laurier)","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","de Maisonneuve \u002f Greene","du Parc-La Fontaine \u002f Duluth","de la Commune \u002f McGill","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","de St-Vallier \u002f St-Zotique","Notre-Dame \u002f Peel","de Li\u00e8ge \u002f Lajeunesse","St-Dominique \u002f Villeray","M\u00e9tro Place-des-Arts (de Maisonneuve \u002f de Bleury)","Logan \u002f d'Iberville","de la Roche \u002f du Mont-Royal","Rouen \u002f Fullum","de Chateaubriand \u002f Jarry","41e avenue \u002f Rosemont","de l'Esplanade \u002f Duluth","St-Mathieu \u002fSte-Catherine","28e avenue \u002f Rosemont","St-Andr\u00e9 \u002f Laurier","Ste-Catherine \u002f Parthenais","Parc Beaubien (Stuart \u002f St-Viateur)","d'Orl\u00e9ans \u002f Masson","Drolet \u002f St-Zotique","Peel \u002f avenue des Canadiens de Montr\u00e9al","St-Andr\u00e9 \u002f St-Antoine","Square Victoria","Louis-H\u00e9bert \u002f B\u00e9langer","Park Row O \u002f Sherbrooke","des \u00c9rables \u002f Gauthier","15e avenue \u002f Beaubien","Lajeunesse \u002f de Castelnau","Chomedey \u002f Lincoln","Milton \u002f Durocher","de Maisonneuve \u002f Greene","Harvard \u002f Monkland","de la Roche \u002f St-Joseph","8e avenue \u002f Beaubien","Parc de la Conf\u00e9d\u00e9ration (Fielding \u002f West Hill)","Beatty \u002f Verdun","D\u00e9z\u00e9ry \u002f Ontario","Chambord \u002f Laurier","Basile-Routhier \u002f Gouin","Parc Outremont (Bloomfield \u002f Elmwood)","Pontiac \u002f Gilford","Roy \u002f St-Laurent","des \u00c9rables \u002f Gauthier","M\u00e9tro Monk (Allard \u002f Beaulieu)","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Querbes \u002f Laurier","Square Victoria","Stuart \u002f de la C\u00f4te-Ste-Catherine","Valois \u002f Ontario","Letourneux \u002f Ste-Catherine","Fairmount \u002f St-Dominique","Marquette \u002f Rachel","1\u00e8re  avenue \u002fSt-Zotique","Rousselot \u002f Villeray","Rushbrooke \u002f  Caisse","Parc Grovehill (St-Antoine \u002f Ivan-Franko)","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f de la Salle)","Chambord \u002f Jean-Talon","de l'H\u00f4tel-de-Ville \u002f Roy","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Logan \u002f Fullum","Bishop \u002f Ren\u00e9-L\u00e9vesque","Lesp\u00e9rance \u002f Rouen","Gascon \u002f Rachel","Clark \u002f Fleury","Square Chaboillez (St-Jacques \u002f de la Cath\u00e9drale)","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","Labelle \u002f Ste-Catherine","Drummond \u002f Ste-Catherine","Clark \u002f Prince-Arthur","Palm \u002f St-Remi","M\u00e9tro Lasalle (de Rushbrooke \u002f Caisse)","West Broadway \u002f Sherbrooke","Gordon \u002f Wellington","Valois \u002f Ontario","Milton \u002f Clark","Fullum \u002f de Rouen","M\u00e9tro Jolicoeur (Drake \u002f de S\u00e8ve)","Chambord \u002f Beaubien","5e avenue \u002f Verdun","de la Roche \u002f  de Bellechasse","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Louis-H\u00e9bert \u002f de Bellechasse","Laval \u002f Rachel","Hutchison \u002f Fairmount","de Montmorency \u002f Richardson","Square Ahmerst \u002f Wolfe","Louis-H\u00e9bert \u002f de Bellechasse","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","Marquette \u002f du Mont-Royal","St-Jacques \u002f St-Pierre","De Lanaudi\u00e8re \u002f du Mont-Royal","St-Mathieu \u002f Sherbrooke","Fran\u00e7ois-Perrault \u002f L.-O.-David","Shearer \u002f Centre","Louis-H\u00e9bert \u002f Beaubien","Henri-Julien \u002f de Castelnau","St-Dominique \u002f Gounod","M\u00e9tro Honor\u00e9-Beaugrand (Sherbrooke \u002f Honor\u00e9-Beaugrand)","Benny \u002f Sherbrooke","St-Alexandre \u002f Ste-Catherine","7e avenue \u002f St-Joseph","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","Clark \u002f Prince-Arthur","Montcalm \u002f de Maisonneuve","M\u00e9tro Villa-Maria (de Monkland \u002f D\u00e9carie)","Roy \u002f St-Laurent","Poupart \u002f Ontario","Valli\u00e8res \u002f St-Laurent","Gilford \u002f de Br\u00e9beuf","Lambert-Closse \u002f Tupper","Bel Air \u002f St-Antoine","Stuart \u002f de la C\u00f4te-Ste-Catherine","Lincoln \u002f du Fort","15e avenue \u002f Masson","Marie-Anne \u002f St-Hubert","Parc des Rapides (LaSalle \u002f 6e avenue)","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","de Bordeaux \u002f Rachel","Messier \u002f du Mont-Royal","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Henri-Julien \u002f du Carmel","\u00c9mile-Journault \u002f de Chateaubriand","de Maisonneuve \u002f Aylmer (ouest)","Berri \u002f Rachel","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Square Victoria (Viger \u002f du Square-Victoria)","Calixa-Lavall\u00e9e \u002f Sherbrooke","Place du Commerce","de Gasp\u00e9 \u002f Dante","Towers \u002f Ste-Catherine","Coloniale \u002f du Mont-Royal","Palm \u002f St-Remi","Guizot \u002f Clark","Bernard \u002f Jeanne-Mance","28e avenue \u002f Rosemont","M\u00e9tro Vend\u00f4me","St-Joseph \u002f 34e avenue Lachine","B\u00e9langer \u002f St-Denis","de Gasp\u00e9 \u002f Dante","Duluth \u002f de l'Esplanade","Laval \u002f Rachel","1\u00e8re avenue \u002f Beaubien","St-Urbain \u002f Beaubien","Louis-H\u00e9bert \u002f B\u00e9langer","Parc Joyce (Rockland \u002f Lajoie)","10e Avenue \u002f Rosemont","Laurier \u002f de Br\u00e9beuf","4e avenue \u002f Masson","6e avenue \u002f B\u00e9langer","CIUSSS (du Trianon \u002f Pierre-de-Coubertin)","St-Marc \u002f Sherbrooke","Chabot \u002f Laurier","du President-Kennedy \u002f Robert Bourassa","de la Commune \u002f Place Jacques-Cartier","Roy \u002f St-Christophe","5e avenue \u002f Bannantyne","L\u00e9a-Roback \u002f Sir-Georges-Etienne-Cartier","1\u00e8re avenue \u002f Masson","Parc Rosemont (Dandurand \u002f d'Iberville)","Jacques-Cassault \u002f Christophe-Colomb","St-Zotique \u002f Clark","Basile-Routhier \u002f Chabanel","Ontario \u002f Viau","Plessis \u002f Ren\u00e9-L\u00e9vesque","Clark \u002f Prince-Arthur","March\u00e9 Atwater","St-Jacques \u002f St-Laurent","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Dollard \u002f Bernard","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","BAnQ (Berri \u002f de Maisonneuve)","Marie-Anne \u002f St-Hubert","St-Andr\u00e9 \u002f Cherrier","Chomedey \u002f de Maisonneuve","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Chabot \u002f Everett","Wolfe \u002f Ren\u00e9-L\u00e9vesque","13e avenue \u002f St-Zotique","de Bullion \u002f du Mont-Royal","M\u00e9tro Jarry (Lajeunesse \u002f Jarry)","Prince-Arthur \u002f St-Urbain","St-Marc \u002f Sherbrooke","Desjardins \u002f Hochelaga","5e avenue \u002f Masson","35e avenue \u002f Beaubien","Maguire \u002f St-Laurent","Square Victoria (Viger \u002f du Square-Victoria)","Coloniale \u002f du Mont-Royal","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","de Repentigny \u002f Sherbrooke","M\u00e9tro de l'\u00c9glise (Ross \u002f de l'\u00c9glise)","Casgrain \u002f St-Viateur","d'Orl\u00e9ans \u002f du Mont-Royal","de la Roche \u002f du Mont-Royal","M\u00e9tro Assomption (Chauveau \u002f de l'Assomption)","Bercy \u002f Rachel","de Montmorency \u002f Richardson","Jeanne-d'Arc \u002f Ontario","U. Concordia - Campus Loyola (Sherbrooke \u002f West Broadway)","Messier \u002f du Mont-Royal","M\u00e9tro Henri-Bourassa (Henri-Bourassa \u002f Millen)","St-Joseph \u002f 34e avenue Lachine","M\u00e9tro Villa Maria (D\u00e9carie \u002f Monkland)","Berlioz \u002f Boul de l'Ile des Soeurs","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","St-Dominique \u002f St-Zotique","Aylwin \u002f Ontario","Basile-Routhier \u002f Gouin","Guy \u002f Notre-Dame","Parc LaSalle (10e avenue \u002f Victoria)","Br\u00e9beuf \u002f Laurier","d'Oxford \u002f de Monkland","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","St-Dominique \u002f Rachel","Mansfield \u002f Ren\u00e9-L\u00e9vesque","Bloomfield \u002f Van Horne","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","de St-Vallier \u002f Rosemont","Briand \u002f Carron","de Ville-Marie \u002f Ste-Catherine","Labadie \u002f du Parc","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","d'Outremont \u002f Ogilvy","Square Phillips","Desjardins \u002f de Rouen","de St-Firmin \u002f Fleury","de la Sucrerie \u002f Centre","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Ontario \u002f Sicard","Jacques-Le Ber \u002f de la Pointe Nord","15e avenue \u002f Masson","Logan \u002f de Champlain","de Maisonneuve \u002f Mansfield","Fairmount \u002f St-Dominique","des \u00c9rables \u002f Rachel","Boyer \u002f B\u00e9langer","St-Viateur\u002fParc","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Parc MacDonald (Clanranald \u002f Isabella)","Ste-Catherine \u002f Clark","\u00c9douard-Montpetit \u002f Stirling","Champlain \u002f Ontario","Bordeaux \u002f St-Zotique","Berri \u002f St-Gr\u00e9goire","Hutchison\u002f Prince-Arthur","Gare d'autocars de Montr\u00e9al (Berri \u002f Ontario)","de Kent \u002f Victoria","Canning \u002f Notre-Dame","M\u00e9tro de l'\u00c9glise (Ross \u002f de l'\u00c9glise)","M\u00e9tro Radisson (Sherbrooke \u002f des Groseilliers)","Drummond \u002f Ste-Catherine","Chabot \u002f du Mont-Royal","Parc Jean-Brillant (Swail \u002f Decelles)","de Br\u00e9beuf \u002f St-Gr\u00e9goire","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Rousselot \u002f Villeray","de la Commune \u002f King","Jean Langlois \u002f Fullum","Quai de la navette fluviale","Roy \u002f St-Laurent","Chambord \u002f B\u00e9langer","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","du Parc-La Fontaine \u002f Duluth","Guy \u002f Notre-Dame","Laval \u002f du Mont-Royal","McGill \u002f Place d'Youville","Nicolet \u002f Ste-Catherine","de Bordeaux \u002f Sherbrooke","de la Roche \u002f du Mont-Royal","Hutchison \u002f Sherbrooke","du Mont-Royal \u002f du Parc","C\u00f4te St-Antoine \u002f Clarke","Turgeon \u002f Notre-Dame","Argyle \u002f Bannantyne","Parc Pratt (Dunlop \u002f Van Horne)","Canning \u002f Notre-Dame","Drolet \u002f Jarry","des \u00c9rables \u002f B\u00e9langer","de Bordeaux \u002f Masson","Parc Outremont (Bloomfield \u002f Elmwood)","George-Baril \u002f Sauv\u00e9","Louis-H\u00e9bert \u002f Beaubien","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","St-Christophe \u002f Cherrier","Pontiac \u002f Gilford","Metcalfe \u002f de Maisonneuve","Gary-Carter \u002f St-Laurent","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","Lajeunesse \u002f de Li\u00e8ge","Ontario \u002f Viau","St-Zotique \u002f Clark","C\u00e9gep Marie-Victorin","Metcalfe \u002f Square Dorchester","Ann \u002f William","Parc Stoney-Point (St-Joseph \u002f 47e avenue)","d'Orl\u00e9ans \u002f Masson","St-Jacques \u002f Gauvin","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","Viger \u002f Chenneville","Bernard \u002f Clark","M\u00e9tro Villa Maria (D\u00e9carie \u002f Monkland)","Ontario \u002f Viau","Lajeunesse \u002f Villeray (place Tap\u00e9o)","Mairie d'arrondissement Montr\u00e9al-Nord (H\u00e9bert \u002f de Charleroi)","8e avenue \u002f Beaubien","Basile-Routhier \u002f Chabanel","Querbes \u002f Laurier","Chabanel \u002f de l'Esplanade","du Parc-La Fontaine \u002f Duluth","McTavish \u002f Sherbrooke","Villeneuve \u002f St-Laurent","19e avenue \u002f St-Zotique","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","St-Hubert \u002f St-Joseph","Fleury \u002f Lajeunesse","M\u00e9tro Guy-Concordia (Guy \u002f St-Catherine)","St-Charles \u002f St-Sylvestre","Mentana \u002f Marie-Anne","Square Phillips","Fortune\u002fwellington","de Belleville \u002f Fleury","Hillside \u002f Ste-Catherine","Duluth \u002f St-Denis","de Lisieux \u002f Jean-Talon","de la C\u00f4te St-Antoine \u002f Royal","de la P\u00e9pini\u00e8re \u002f Pierre-de-Coubertin","Prince-Arthur \u002f du Parc","Gauthier \u002f De Lorimier","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","d'Orl\u00e9ans \u002f Masson","Cartier \u002f B\u00e9langer","Stanley \u002f Sherbrooke","Chapleau \u002f du Mont-Royal","Marseille \u002f Viau","Gagne \u002f LaSalle","March\u00e9 Maisonneuve","Bloomfield \u002f Van Horne","Square St-Louis","Place du Commerce","Clark \u002f Prince-Arthur","Ste-Famille \u002f Sherbrooke","University \u002f des Pins","Hutchison \u002f des Pins","St-Jacques \u002f St-Pierre","Natatorium (LaSalle \u002f Rolland)","Mansfield \u002f Ste-Catherine","M\u00e9tro Montmorency (de l'Avenir \u002f Jacques-T\u00e9treault)","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","des \u00c9rables \u002f Laurier","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Union \u002f Ren\u00e9-L\u00e9vesque","Boyer \u002f Jean-Talon","Wolfe \u002f Ren\u00e9-L\u00e9vesque","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Sicard \u002f Adam","de Mentana \u002f Gilford","Milton \u002f University","de Gasp\u00e9 \u002f de Li\u00e8ge","Robin \u002f 15e rue","Milton \u002f du Parc","Lincoln \u002f du Fort","Chambord \u002f Beaubien","26e avenue \u002f Beaubien","Valois \u002f Ste-Catherine","Hamel \u002f Sauv\u00e9","Monk \u002f de Biencourt","M\u00e9tro Champ-de-Mars (Sanguinet \u002f Viger)","Crescent \u002f de Maisonneuve","de la Commune \u002f St-Sulpice","Belmore \u002f Sherbrooke","Desjardins \u002f Hochelaga","de l\u2019h\u00f4tel de ville \u002f Saint-Joseph","Fullum \u002f Gilford","de la Commune \u002f St-Sulpice","St-Hubert \u002f Ren\u00e9-L\u00e9vesque","Poupart \u002f Ontario","Poupart \u002f Ontario","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","Square St-Louis","Sanguinet \u002f Ontario","Chapleau \u002f du Mont-Royal","Crescent \u002f Ren\u00e9-L\u00e9vesque","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Milton \u002f Universit\u00e9","de la Montagne \u002f Sherbrooke","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","de Gasp\u00e9 \u002f St-Viateur","Victoria \u002f de Maisonneuve","Hutchison \u002f Fairmount","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Graham \u002f Kindersley","Atwater \u002f Greene","Drummond \u002f de Maisonneuve","de Bullion \u002f St-Joseph","Gatineau \u002f Swail","Roy \u002f St-Andr\u00e9","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Wurtele \u002f de Rouen","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","de la Commune \u002f Place Jacques-Cartier","de Gasp\u00e9 \u002f Marmier","Calixa-Lavall\u00e9e \u002f Sherbrooke","Wilderton  \u002f Van Horne","St-Antoine \u002f St-Fran\u00e7ois-Xavier","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Logan \u002f d'Iberville","5e avenue \u002f Rosemont","St-Urbain \u002f de la Gaucheti\u00e8re","La Ronde","Notre-Dame \u002f Chatham","de Gasp\u00e9 \u002f Jarry","Berri \u002f Cherrier","Louis H\u00e9mon \u002f Rosemont","Jeanne-Mance \u002f Beaubien","de Chambly \u002f Rachel","Gilford \u002f de Lanaudi\u00e8re","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Metcalfe \u002f du Square Dorchester","Rivard \u002f Rachel","Marquette \u002f Fleury","Basin \u002f Richmond","de la Roche \u002f St-Joseph","Victoria \u002f de Maisonneuve","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","St-Alexandre \u002f Ste-Catherine","des \u00c9rables \u002f Rachel","Parc Outremont (Bloomfield \u002f Elmwood)","Prince-Arthur \u002f St-Urbain","Beatty \u002f de Verdun","M\u00e9tro Beaudry (Montcalm \u002f de Maisonneuve)","de la Montagne \u002f Sherbrooke","St-Andr\u00e9 \u002f Cherrier","Place \u00c9milie-Gamelin (de Maisonneuve \u002f Berri)","de Maisonneuve \u002f de Bleury","de Vend\u00f4me \u002f de Maisonneuve","St-Joseph \u002f 8e avenue Lachine","Duke \u002f Brennan","Pontiac \u002f Gilford","Marquette \u002f Laurier","Campus MIL (Outremont \u002f Th\u00e9r\u00e8se-Lavoie Roux)","Argyle \u002f de Verdun","Louis-H\u00e9mon \u002f Villeray","Darling \u002f Sherbrooke","M\u00e9tro Jean-Drapeau","Faillon \u002f St-Denis","Alfred Lalibert\u00e9 \u002f de Poutrincourt","Chabot \u002f de Bellechasse","Berri \u002f Sherbrooke","Ste-Catherine \u002f Ste-\u00c9lisabeth","Berri \u002f Jean-Talon","Parc de Beaus\u00e9jour (Gouin \u002f Henri-d'Arles)","Laval \u002f Rachel","Waverly \u002f St-Zotique","Robert Bourassa \u002f Sainte-Catherine","St-Marc \u002f Sherbrooke","La Ronde","Alfred Lalibert\u00e9 \u002f de Poutrincourt","St-Denis \u002f de Maisonneuve","Papineau \u002f Ren\u00e9-L\u00e9vesque","28e avenue \u002f Rosemont","St-Alexandre \u002f Ste-Catherine","de la Roche \u002f Fleury","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","M\u00e9tro Cr\u00e9mazie (Cr\u00e9mazie \u002f Lajeunesse)","des \u00c9cores \u002f Jean-Talon","Place Jean-Paul Riopelle (Viger \u002f de Bleury)","Milton \u002f Clark","M\u00e9tro Angrignon (Lamont \u002f  des Trinitaires)","de Lanaudi\u00e8re \u002f Gilford","CCI Outremont (Dollard \u002f Durcharme)","Evans \u002f Clark","Victoria Hall","Notre-Dame-de-Gr\u00e2ce \u002f Westmount","Gare Bois-de-Boulogne","St-Dominique \u002f Napol\u00e9on","Eleanor \u002f Ottawa","Lesp\u00e9rance \u002f Rouen","M\u00e9tro Lionel-Groulx (Lionel-Groulx \u002f Charlevoix)","M\u00e9tro Lionel-Groulx (Walker \u002f Saint-Jacques)","de Maisonneuve \u002f Robert-Bourassa","Guy \u002f Notre-Dame","Casgrain \u002f de Bellechasse","M\u00e9tro Jean-Drapeau","de la Cath\u00e9drale \u002f Ren\u00e9-L\u00e9vesque","de Gasp\u00e9 \u002f de Li\u00e8ge","St-Zotique \u002f Clark","Gilford \u002f Br\u00e9beuf","St-Andr\u00e9 \u002f Ontario","19e avenue \u002f St-Zotique","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Cit\u00e9 des Arts du Cirque (Paul Boutet \u002f des Regrattiers)","Victoria Hall","Rockland \u002f Lajoie","Cartier \u002f Marie-Anne","de Bordeaux \u002f Jean-Talon","Rachel \u002f de Br\u00e9beuf","Quai de la navette fluviale","9e avenue \u002f Dandurand","de Maisonneuve \u002f Peel","University \u002f Milton","Hutchison \u002f des Pins","de la Gaucheti\u00e8re \u002f Robert-Bourassa","de Gasp\u00e9 \u002f Fairmount","Place Longueuil","de la Montagne \u002f Sherbrooke","Bishop \u002f Ste-Catherine","Regina \u002f de Verdun","Plessis \u002f Ontario","Parc Ludger-Duvernay (St-Andr\u00e9 \u002f St-Hubert)","Place Jean-Paul Riopelle (Viger \u002f de Bleury)","Marquette \u002f St-Gr\u00e9goire","Sherbrooke \u002f Vignal","Belmont \u002f Beaver Hall","Parc Maurice B\u00e9langer (Hebert \u002f Forest)","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","Chambord \u002f Beaubien","Messier \u002f du Mont-Royal","M\u00e9tro Monk (Allard \u002f Beaulieu)","Harvard \u002f Monkland","Bloomfield \u002f Van Horne","Hutchison \u002f des Pins","Bannantyne \u002f de l'\u00c9glise","Laurier \u002f Parisi","Garnier \u002f du Mont-Royal","Ellendale \u002f de la C\u00f4te-des-Neiges","Bordeaux \u002fGilford","18e avenue \u002f Rosemont","Ste-Catherine \u002f McGill College","Rouen \u002f Fullum","Mansfield \u002f Cathcart","\u00c9mile-Journault \u002f de Chateaubriand","Hutchison \u002f Edouard Charles","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","St-Andr\u00e9 \u002f Cherrier","Alexandre-DeS\u00e8ve \u002f la Fontaine","Hutchison \u002f Sherbrooke","de Maisonneuve\u002f Mansfield (est)","Victoria Hall","Ste-Catherine \u002f D\u00e9z\u00e9ry","Atwater \u002f Sherbrooke","Chambord \u002f Jean-Talon","Mackay \u002f Ren\u00e9-L\u00e9vesque","Laval \u002f du Mont-Royal","Villeneuve \u002f de l'H\u00f4tel-de-Ville","St-Germain \u002f Hochelaga","St-Dominique \u002f St-Viateur","Gauthier \u002f Papineau","Rivard \u002f Rachel","St-Hubert \u002f Duluth","Bloomfield \u002f Jean-Talon","Park Row O \u002f Sherbrooke","Victoria Hall","Ryde \u002f Charlevoix","de Bordeaux \u002f St-Zotique","de la Salle \u002f Ste-Catherine","Aylwin \u002f Ontario","Berri \u002f Jean-Talon","C\u00f4te St-Antoine \u002f Royal","La Ronde","St-Hubert \u002f Rachel","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","Lionel-Groulx \u002f George-Vanier","Ste-Catherine \u002f Dezery","Towers \u002f Ste-Catherine","Jacques-Le Ber \u002f de la Pointe Nord","Sherbrooke \u002f Frontenac","Marquette \u002f Rachel","St-Charles \u002f Thomas-Keefer","Centre des loisirs (Tass\u00e9 \u002f Grenet)","4e avenue \u002f Masson","St-Charles \u002f Montarville","Nicolet \u002f Sherbrooke","Chauveau \u002f de l'Assomption","Francis \u002f Fleury","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","Prince-Arthur \u002f Ste-Famille","Quesnel \u002f Vinet","des Seigneurs \u002f Notre-Dame","Sainte-Catherine \u002f Parthenais","St-Andr\u00e9 \u002f Laurier","Graham \u002f Brookfield","Crescent \u002f de Maisonneuve","Berri \u002f Duluth","Laurier \u002f 15e avenue","St-Alexandre \u002f Ste-Catherine","Marie-Anne \u002f de la Roche","Greene \u002f St-Ambroise","De la Commune \u002f King","M\u00e9tro du Coll\u00e8ge (Cartier \u002f D\u00e9carie)","de Mentana \u002f Laurier","St-Charles \u002f Grant","Lajeunesse \u002f de Castelnau","Metcalfe \u002f du Square-Dorchester","Louis-H\u00e9mon \u002f Rosemont","Pierre-de-Coubertin \u002f Aird","Jacques-Casault \u002f Christophe-Colomb","30e avenue \u002f St-Zotique","de Maisonneuve \u002f Aylmer (est)","Hutchison \u002f Sherbrooke","Jacques-Casault \u002f Christophe-Colomb","d'Orl\u00e9ans \u002f Hochelaga","de Lanaudi\u00e8re \u002f Marie-Anne","26e avenue \u002f Beaubien","Hudson \u002f de Kent","St-Jacques \u002f St-Laurent","Sanguinet \u002f Sainte-Catherine","Parc J.-Arthur-Champagne (Chambly \u002f du Mont-Royal)","University \u002f des Pins","Hamel \u002f Sauv\u00e9","\u00c9douard-Montpetit \u002f de Stirling","Hutchison \u002f des Pins","Union \u002f Ren\u00e9-L\u00e9vesque","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","16e avenue \u002f Beaubien","Parc Maurice B\u00e9langer (Forest \u002f de l'H\u00f4tel-de-Ville)","Boyer \u002f B\u00e9langer","de Bordeaux \u002f Marie-Anne","19e avenue \u002f St-Joseph Lachine","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","Plessis \u002f Ontario","Victoria Hall","Panet \u002f Ste-Catherine","Ernest-Gendreau \u002f du Mont-Royal","de Kent \u002f Victoria","Jean-Brillant \u002f McKenna","Pierre-de-Coubertin \u002f Louis-Veuillot","Laurier \u002f 15e avenue","de Maisonneuve \u002f Robert-Bourassa","Square Phillips","Turgeon \u002f Notre-Dame","Marlowe \u002f de Maisonneuve","M\u00e9tro Plamondon (de Kent \u002f Victoria)","Joseph Manceau \u002f Ren\u00e9-L\u00e9vesque","Berri \u002f St-Joseph","M\u00e9tro Honor\u00e9-Beaugrand (Sherbrooke \u002f Honor\u00e9-Beaugrand)","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Drolet \u002f Laurier","Milton \u002f Durocher","Parthenais \u002f Ste-Catherine","de Lanaudi\u00e8re \u002f Marie-Anne","de Melrose \u002f Sherbrooke","Cartier \u002f Rosemont","St-Dominique \u002f St-Viateur","Villeneuve \u002f St-Urbain","University \u002f Prince-Arthur","de l'Esplanade \u002f du Mont-Royal","Ren\u00e9-L\u00e9vesque \u002f Papineau","Marie-Anne \u002f St-Hubert","Georges-Vanier \u002f Notre-Dame","Marie-Anne \u002f St-Hubert","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","LaSalle \u002f S\u00e9n\u00e9cal","City Councillors \u002f du President-Kennedy","Fullum \u002f St-Joseph","3e avenue \u002f Dandurand","St-Andr\u00e9 \u002f St-Gr\u00e9goire","Casino de Montr\u00e9al","CHSLD St-Michel (Jarry \u002f 8e avenue)","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Mackay \u002fde Maisonneuve (Sud)","Messier \u002f du Mont-Royal","M\u00e9tro George-Vanier (St-Antoine \u002f Canning)","de Vend\u00f4me \u002f de Maisonneuve","M\u00e9tro Atwater (Atwater \u002f Ste-Catherine)","de Li\u00e8ge \u002f Lajeunesse","16e avenue \u002f St-Joseph","Villeray \u002f St-Michel","de Bordeaux \u002f Masson","Notre-Dame \u002f St-Gabriel","Kirkfield \u002f de Chambois","de la Commune \u002f King","12e avenue \u002f St-Zotique","Villeneuve \u002f St-Laurent","St-Jacques \u002f Gauvin","St-Dominique \u002f du Mont-Royal","Drummond \u002f Ste-Catherine","Boyer \u002f Jean-Talon","Bordeaux \u002f St-Zotique","Hutchison \u002f Beaubien","Hamilton \u002f Jolicoeur","LaSalle \u002f Crawford","Faillon \u002f St-Hubert","Augustin-Cantin \u002f Shearer","Gilford \u002f de Br\u00e9beuf","Ste-Croix \u002f Tass\u00e9","Chambord \u002f Beaubien","Mansfield \u002f Sherbrooke","de Bordeaux \u002f Gilford","de Bellechasse \u002f de St-Vallier","Laval \u002f Duluth","de l'H\u00f4tel-de-Ville \u002f St-Joseph","St-Andr\u00e9 \u002f Ontario","de Maisonneuve \u002f Aylmer","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Notre-Dame \u002f Li\u00e9bert","Queen \u002f Ottawa","de la Commune \u002f King","de l'H\u00f4tel-de-Ville \u002f Roy","16e avenue \u002f Beaubien","St-Charles \u002f St-Jean","Notre-Dame \u002f de la Montagne","8e avenue \u002f Beaubien","Trans Island \u002f Queen-Mary","Stuart \u002f de la C\u00f4te-Ste-Catherine","Fairmount \u002f St-Dominique","Parc Dixie (54e avenue \u002f Dixie)","Beaudry \u002f Ontario","Louis-H\u00e9bert \u002f B\u00e9langer","de l'\u00c9glise \u002f de Verdun","Chomedey \u002f de Maisonneuve","Gauthier \u002f Papineau","Duluth \u002f St-Denis","16e avenue \u002f Jean-Talon","Poupart \u002f Ontario","du Mont-Royal \u002f du Parc","Chapleau \u002f du Mont-Royal","St-Zotique \u002f Clark","du Mont-Royal \u002f Vincent-d'Indy","des R\u00e9collets \u002f Martial","Milton \u002f du Parc","Bernard \u002f Jeanne-Mance","Poupart \u002f Ste-Catherine","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","de la Gaucheti\u00e8re \u002f Robert-Bourassa","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","Parc Plage","Waverly \u002f St-Zotique","de Bullion \u002f St-Joseph","Tellier \u002f des Futailles","La Ronde","Fran\u00e7ois-Perrault \u002f L.-O.-David","Lajeunesse \u002f Jarry","Marquette \u002f St-Gr\u00e9goire","3e avenue \u002f Dandurand","Lajeunesse \u002f Jarry","Berri \u002f Gilford","de Hampton \u002f de Monkland","Atateken \u002f du Square Amherst","Davaar \u002f de la C\u00f4te-Ste-Catherine","St-Andr\u00e9 \u002f Cherrier","de Bordeaux \u002f Rachel","Casgrain \u002f Mozart","Guizot \u002f St-Laurent","M\u00e9tro Assomption (Chauveau \u002f de l'Assomption)","Francis \u002f Fleury","des Seigneurs \u002f Workman","City Councillors \u002f du President-Kennedy","Ste-Catherine \u002f St-Denis","Francis \u002f Fleury","St-Andr\u00e9 \u002f Duluth","St-Marc \u002f Sherbrooke","de la C\u00f4te St-Antoine \u002f Argyle","C\u00f4te St-Antoine \u002f Clarke","Biblioth\u00e8que de Verdun (Brown \u002f Bannantyne)","Grand Boulevard \u002f Sherbrooke","Resther \u002f St-Joseph","M\u00e9tro Sherbrooke (de Rigaud \u002f St-Denis)","Fabre \u002f St-Zotique","Drummond \u002f de Maisonneuve","Mackay \u002f Ren\u00e9-L\u00e9vesque","St-Andr\u00e9 \u002f Duluth","Champagneur \u002f Jean-Talon","Park Row O \u002f Sherbrooke","Laval \u002f Rachel","Resther \u002f St-Joseph","Jogues \u002f Allard","Cypress \u002f Peel","Basile-Routhier \u002f Chabanel","Marmier","\u00c9douard-Montpetit \u002f Stirling","du President-Kennedy \u002f Union","Monkland \u002f Girouard","M\u00e9tro Sauv\u00e9 (Berri \u002f Sauv\u00e9)","Lucien L'Allier \u002f St-Jacques","La Ronde","27e avenue \u002f Rosemont","M\u00e9tro Angrignon","Molson \u002f Beaubien","Stanley \u002f Sherbrooke","Calixa-Lavall\u00e9e \u002f Sherbrooke","Davaar \u002f de la C\u00f4te-Ste-Catherine","Casgrain \u002f de Bellechasse","Beaudry \u002f Sherbrooke","Biblioth\u00e8que d'Ahuntsic (Lajeunesse \u002f Fleury)","Parc H\u00e9bert (Provencher \u002f Buies)","de la Roche \u002f Everett","5e avenue \u002f Verdun","M\u00e9tro Henri-Bourassa (Henri-Bourassa \u002f Millen)","du Mont-Royal \u002f Clark","Milton \u002f Clark","de Chateaubriand \u002f Jarry","Marquette \u002f Rachel","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","de la Commune \u002f Place Jacques-Cartier","9e avenue \u002f Dandurand","Louis-H\u00e9bert \u002f B\u00e9langer","M\u00e9tro Universit\u00e9 de Montr\u00e9al (\u00c9douard-Montpetit \u002f Louis-Collin)","Fleury \u002f Lajeunesse","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","de L\u00e9ry \u002f Sherbrooke","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","St-Dominique \u002f St-Zotique","de l'H\u00f4tel-de-Ville \u002f Roy","Sanguinet \u002f Ontario","Ste-Famille \u002f Sherbrooke","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","de la Roche \u002f St-Joseph","Grey \u002f Sherbrooke","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Waverly \u002f Van Horne","President-Kennedy \u002f Robert Bourassa","Br\u00e9beuf \u002f St-Gr\u00e9goire","BAnQ (Berri \u002f de Maisonneuve)","Lionel-Groulx \u002f George-Vanier","Fran\u00e7ois-Perrault \u002f L.-O.-David","Briand \u002f Carron","Berri \u002f Cherrier","Garnier \u002f St-Joseph","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Beaudet \u002f Marcel-Laurin","de St-Vallier \u002f St-Zotique","St-Cuthbert \u002f St-Urbain","de Bullion \u002f du Mont-Royal","Cartier \u002f Dandurand","M\u00e9tro Montmorency","de Bullion \u002f St-Joseph","M\u00e9tro C\u00f4te-des-Neiges (Lacombe \u002f de la C\u00f4te-des-Neiges)","Bernard \u002f Jeanne-Mance","du Tricentenaire \u002f de Montigny","M\u00e9tro Sauv\u00e9 (Sauv\u00e9 \u002f Berri)","Dumesnil \u002f B\u00e9langer","Napol\u00e9on \u002f  St-Dominique","des \u00c9cores \u002f Jean-Talon","Metcalfe \u002f Ste-Catherine","Queen \u002f Wellington","Gare Canora","Aylwin \u002f Ontario","Milton \u002f du Parc","des Jockeys \u002f D\u00e9carie","Sherbrooke \u002f Frontenac","Parc Champdor\u00e9 (Champdor\u00e9 \u002f de Lille)","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","Parc Benny (Benny \u002f de Monkland)","Maguire \u002f St-Laurent","Parc Kent (de Kent \u002f Hudson)","du Mont-Royal \u002f Augustin-Frigon","de Mentana \u002f Laurier","Notre-Dame \u002f St-Gabriel","Tupper \u002f du Fort","Mentana \u002f Marie-Anne","Roy \u002f St-Denis","Casino de Montr\u00e9al","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","McGill \u002f Place d'Youville","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f De La Salle)","Bourgeoys \u002f Favard","Belmont \u002f du Beaver Hall","McKenna \u002f \u00c9douard-Montpetit","Cote St-Paul \u002f St-Ambroise","Louis-H\u00e9bert \u002f St-Zotique","Terrasse Guindon \u002f Fullum","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Sanguinet \u002f Ste-Catherine","Boyer \u002f Rosemont","Gounod \u002f St-Denis","Berri \u002f Rachel","de la Cath\u00e9drale \u002f Ren\u00e9-L\u00e9vesque","Bloomfield \u002f Bernard","William \u002f Robert-Bourassa","St-Nicolas \u002f Place d'Youville","Marcil \u002f Sherbrooke","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","Parc Henri-Durant (L\u00e9ger \u002f Henri-Durant)","M\u00e9tro Monk (Allard \u002f Beaulieu)","Place du Canada ( Peel \u002f de la Gaucheti\u00e8re)","Resther \u002f St-Joseph","Charlevoix \u002f Lionel-Groulx","Parc nature de l'\u00cele-de-la-Visitation (Gouin \u002f de Bruch\u00e9si)","Casino de Montr\u00e9al","Notre-Dame \u002f de la Montagne","6e avenue \u002f Rosemont","Drolet \u002f Marie-Anne","de Gasp\u00e9 \u002f Fairmount","Lajeunesse \u002f Jean-Talon","St-Urbain \u002f Beaubien","Laval \u002f Duluth","Regina \u002f Verdun","March\u00e9 Atwater","Plessis \u002f Ren\u00e9-L\u00e9vesque","Parc Baldwin (Fullum \u002f Sherbrooke)","Quai de la navette fluviale","Boyer \u002f du Mont-Royal"],"lat":[45.493909,45.471926,45.550698,45.520604,45.472503,45.546408,45.525628,45.470303,45.48204,45.544846,45.533349,45.529693,45.544513,45.522365,45.530027,45.531674,45.49426,45.533348,45.509864,45.497011,45.499326,45.550737,45.546907,45.521094,45.533496,45.518263,45.553262,45.514115,45.491754,45.500779,45.537877,45.532137,45.466851,45.478228,45.539632,45.573798,45.58077,45.525643,45.509188,45.539984,45.539,45.514905,45.549755,45.4771,45.568225,45.538938,45.519492,45.526543,45.537259,45.415349,45.434434,45.5219,45.470696,45.50623,45.522586,45.609626,45.46714,45.58077,45.523194,45.546408,45.49975,45.539259,45.529686,45.530674,45.536378,45.558457,45.50206,45.537205,45.547362,45.506223,45.5171,45.542055,45.527432,45.504542,45.537259,45.536785,45.480508,45.599266,45.52841,45.456823,45.52729,45.550692,45.539224,45.531475,45.508295,45.483006,45.519237,45.547218,45.496302,45.506448,45.542055,45.497974,45.537159,45.56244,45.459753,45.573733,45.516897,45.529155,45.489727,45.477353,45.49606,45.532674,45.514108,45.519373,45.552364,45.456884,45.531938,45.546408,45.49659,45.49351,45.503475,45.495894,45.530351,45.52689,45.54167,45.539864,45.4829,45.51889,45.5341,45.527426,45.50345,45.518253,45.517643,45.540154,45.559978,45.456049,45.501864,45.480914,45.533735,45.456358,45.524683,45.563921,45.561396,45.505403,45.491384,45.519237,45.526027,45.4688,45.521556,45.553408,45.493569,45.500674,45.529504,45.577136,45.532977,45.506448,45.471926,45.461701,45.435664,45.487531,45.510042,45.498673,45.536178,45.475142,45.526252,45.50623,45.500877,45.52123,45.544867,45.550684,45.512587,45.514115,45.53923,45.526991,45.498112,45.521069,45.570081,45.540951,45.521751,45.532674,45.504278,45.497605,45.570081,45.539824,45.493718,45.471179,45.53217,45.534377,45.527201,45.53848,45.555814,45.529802,45.553053,45.533409,45.505312,45.54257,45.509229,45.526856,45.49642,45.50015,45.513917,45.459488,45.460729,45.532008,45.543376,45.549598,45.550668,45.491322,45.546907,45.569789,45.431648,45.519216,45.480914,45.499965,45.546978,45.52689,45.574147,45.506371,45.514821,45.53318,45.518593,45.544136,45.520604,45.529686,45.539833,45.523492,45.474419,45.479483,45.521293,45.486219,45.530505,45.611231,45.53293,45.480208,45.463257,45.559884,45.47995,45.519153,45.542579,45.534136,45.493029,45.545776,45.541484,45.481184,45.507144,45.539316,45.520188,45.509361,45.491687,45.524634,45.532281,45.508144,45.499757,45.538984,45.502602,45.501715,45.539824,45.485048,45.564944,45.514531,45.523401,45.53543,45.514379,45.549686,45.53229,45.519153,45.550769,45.50931,45.45073,45.592596,45.491722,45.485048,45.524266,45.549755,45.535614,45.542898,45.523027,45.474565,45.50501,45.5219,45.5332,45.543452,45.521768,45.502989,45.545776,45.517643,45.5359,45.514773,45.53611,45.517354,45.483928,45.524963,45.494514,45.497411,45.552923,45.53576,45.53696,45.459986,45.509759,45.4816,45.499677,45.529666,45.533682,45.446934,45.42506,45.551119,45.561783,45.513542,45.52974,45.496851,45.520075,45.521342,45.463397,45.449901,45.520688,45.539385,45.553898,45.506373,45.496583,45.498767,45.500339,45.54458,45.541955,45.517,45.490452,45.537159,45.501367,45.460023,45.544867,45.489545,45.54257,45.548186,45.415769,45.507078,45.459986,45.518215,45.527616,45.532592,45.570506,45.609626,45.510016,45.544859,45.530871,45.521751,45.539824,45.550769,45.527795,45.516079,45.544513,45.466569,45.514784,45.49606,45.561594,45.51086,45.54918,45.552649,45.547397,45.514784,45.493034,45.519373,45.542983,45.544819,45.588303,45.4688,45.51086,45.512734,45.553053,45.501715,45.533661,45.486279,45.53133,45.540627,45.569789,45.524375,45.457894,45.50449,45.4918,45.534563,45.5215,45.559705,45.506251,45.519092,45.498919,45.549714,45.568225,45.505721,45.547997,45.547179,45.511132,45.541369,45.523584,45.519373,45.525021,45.521495,45.495761,45.504347,45.519581,45.47393,45.561594,45.516427,45.480264,45.508356,45.536344,45.434434,45.556262,45.51889,45.538308,45.493034,45.521161,45.508144,45.539984,45.506251,45.492504,45.491754,45.537226,45.464877,45.53877,45.433557,45.531178,45.51059,45.520574,45.501967,45.526246,45.492897,45.429471,45.519157,45.53229,45.503475,45.525628,45.500779,45.532674,45.531938,45.527727,45.529693,45.535341,45.539294,45.532674,45.537649,45.527363,45.510042,45.506176,45.540881,45.539487,45.45002,45.56833,45.538021,45.534136,45.573471,45.505377,45.49871,45.456365,45.582847,45.556065,45.47968,45.520295,45.459488,45.4703,45.545459,45.53707,45.539487,45.534453,45.51735,45.535189,45.501449,45.537226,45.503666,45.518128,45.51494,45.523507,45.531178,45.556262,45.517459,45.49947,45.530046,45.547405,45.528724,45.490472,45.481491,45.475743,45.532331,45.482826,45.528449,45.549121,45.51494,45.549121,45.4952,45.506373,45.566869,45.514379,45.456049,45.505973,45.533348,45.503063,45.453246,45.536785,45.503552,45.461685,45.512801,45.504242,45.49871,45.550684,45.499326,45.523413,45.484859,45.508356,45.500043,45.504242,45.4703,45.523319,45.537259,45.434434,45.4952,45.536505,45.477923,45.552016,45.551584,45.547408,45.52689,45.457591,45.507402,45.530221,45.543452,45.531164,45.5195,45.544513,45.53696,45.61443,45.51059,45.535985,45.56169,45.536299,45.507078,45.541435,45.524634,45.590892,45.532281,45.5452,45.49871,45.529219,45.55896,45.497165,45.49606,45.493596,45.531818,45.53703,45.495789,45.493569,45.446061,45.552123,45.4891,45.532415,45.50723,45.515616,45.527616,45.500105,45.549126,45.553457,45.52689,45.543771,45.543285,45.553215,45.526566,45.564944,45.525172,45.544376,45.5427,45.534415,45.544377,45.5262,45.514946,45.490687,45.561594,45.55197,45.55816,45.529802,45.463254,45.463001,45.63065,45.502813,45.512871,45.49683,45.51086,45.522821,45.52211,45.457509,45.510086,45.561783,45.43746,45.529135,45.49642,45.529033,45.484018,45.551584,45.547179,45.532331,45.526206,45.499745,45.526856,45.47968,45.480508,45.516818,45.514335,45.456365,45.474224,45.513406,45.544599,45.483928,45.552123,45.437914,45.577136,45.505403,45.542972,45.504095,45.439779,45.497595,45.46982,45.544599,45.527041,45.484896,45.522878,45.525164,45.488877,45.535341,45.649546,45.527748,45.471162,45.54111,45.533924,45.48184,45.4891,45.506587,45.522278,45.551584,45.510086,45.517643,45.5452,45.51339,45.541484,45.517354,45.498673,45.547161,45.456355,45.552649,45.461685,45.51829,45.504095,45.538958,45.438544,45.528449,45.556905,45.483629,45.53569,45.512813,45.52841,45.516664,45.533924,45.525669,45.527363,45.480208,45.537041,45.539857,45.519376,45.461685,45.471669,45.542983,45.542283,45.514531,45.467666,45.539271,45.502602,45.559482,45.52246,45.494564,45.557895,45.53217,45.506445,45.553262,45.515175,45.4981,45.520188,45.53092,45.597221,45.535189,45.507662,45.521342,45.526904,45.537159,45.532955,45.550669,45.497504,45.547218,45.504993,45.537159,45.53543,45.543541,45.530505,45.491687,45.520604,45.526058,45.559178,45.522847,45.456498,45.52087,45.540881,45.542119,45.49497,45.477429,45.509864,45.547408,45.544101,45.547227,45.529802,45.498312,45.540823,45.540944,45.480846,45.532331,45.533409,45.549911,45.536261,45.531938,45.543376,45.507144,45.539857,45.524025,45.549911,45.524037,45.550737,45.524566,45.576944,45.487693,45.529326,45.5171,45.543962,45.553571,45.548931,45.50015,45.517,45.513338,45.549911,45.51059,45.553262,45.470303,45.527933,45.527117,45.51791,45.542119,45.53057,45.501974,45.517459,45.49659,45.537877,45.521565,45.538589,45.52765,45.47968,45.550316,45.531164,45.52457,45.58077,45.439145,45.515858,45.579325,45.549605,45.513413,45.49659,45.52317,45.567766,45.546115,45.488855,45.5452,45.541549,45.495415,45.498312,45.521495,45.523035,45.50723,45.518593,45.55197,45.549704,45.554961,45.534134,45.5452,45.54977,45.52708,45.517354,45.549833,45.535828,45.527795,45.520125,45.492222,45.549121,45.530317,45.467369,45.515775,45.518967,45.53318,45.526246,45.499677,45.502326,45.539236,45.494514,45.564353,45.542868,45.546907,45.542119,45.477249,45.529002,45.482779,45.547218,45.547362,45.461685,45.534358,45.49123,45.539932,45.576963,45.517062,45.559828,45.50038,45.49947,45.525048,45.55001,45.517333,45.491384,45.51561,45.533547,45.532977,45.456355,45.550316,45.444408,45.544958,45.549755,45.472668,45.537026,45.521069,45.49606,45.544867,45.49947,45.532926,45.532281,45.499326,45.481491,45.531783,45.505403,45.51496,45.52479,45.51059,45.537041,45.524037,45.480073,45.53565,45.553408,45.456351,45.521495,45.456355,45.481491,45.573733,45.523584,45.468927,45.519216,45.55001,45.526246,45.523525,45.532977,45.514379,45.461078,45.53848,45.517333,45.5499,45.520188,45.512014,45.480914,45.530351,45.523492,45.545776,45.51908,45.526027,45.483006,45.5306,45.510072,45.554947,45.518593,45.49659,45.595352,45.539932,45.555004,45.522278,45.531565,45.562668,45.51066,45.477107,45.543424,45.526006,45.504993,45.541484,45.522426,45.545542,45.534739,45.485889,45.480208,45.543651,45.542993,45.549686,45.475963,45.545542,45.515604,45.51908,45.517062,45.531917,45.524628,45.529686,45.501441,45.541435,45.533924,45.529155,45.486319,45.515065,45.525048,45.466569,45.542645,45.512813,45.54816,45.54467,45.53229,45.499757,45.531327,45.552016,45.557921,45.504777,45.509655,45.50781,45.537628,45.505377,45.506373,45.462362,45.551582,45.542794,45.640003,45.535134,45.4918,45.46714,45.559199,45.523276,45.512813,45.534882,45.563921,45.53057,45.522558,45.462742,45.484489,45.542794,45.502326,45.465058,45.539932,45.510625,45.53543,45.477107,45.503063,45.524942,45.524518,45.531958,45.519088,45.494499,45.493029,45.542579,45.515052,45.533682,45.521485,45.518309,45.527933,45.549833,45.467369,45.478228,45.51059,45.535341,45.534864,45.535248,45.529686,45.523856,45.55896,45.531774,45.53073,45.477923,45.444547,45.506899,45.529219,45.530351,45.599086,45.469046,45.539663,45.547108,45.53696,45.500779,45.471743,45.53611,45.551937,45.520688,45.552289,45.551774,45.51496,45.542055,45.562299,45.506421,45.48498,45.53341,45.553571,45.533348,45.546978,45.576,45.466965,45.502813,45.574984,45.478671,45.535709,45.520178,45.497718,45.456498,45.546939,45.529693,45.538589,45.502311,45.534185,45.494527,45.49642,45.52479,45.534453,45.527117,45.494805,45.497605,45.49368,45.537813,45.487522,45.558457,45.640003,45.525048,45.496302,45.522907,45.4703,45.522278,45.533496,45.553948,45.521461,45.570727,45.515604,45.512801,45.537226,45.514087,45.498639,45.446534,45.557022,45.529033,45.53229,45.538627,45.520688,45.447554,45.553571,45.54813,45.521015,45.52697,45.550769,45.500278,45.52479,45.5079,45.459707,45.533348,45.459593,45.527432,45.514114,45.537041,45.550769,45.543541,45.537041,45.542119,45.526557,45.50761,45.4703,45.530351,45.53867,45.532408,45.504561,45.547304,45.549963,45.516091,45.649546,45.519049,45.553219,45.485801,45.484018,45.49123,45.52974,45.508205,45.513406,45.602728,45.54458,45.492222,45.550476,45.524025,45.544867,45.544101,45.516777,45.547227,45.512832,45.501715,45.489727,45.521161,45.493718,45.498673,45.539833,45.53576,45.529559,45.538958,45.50761,45.539984,45.544513,45.501449,45.514767,45.546939,45.553262,45.470303,45.546094,45.497055,45.498765,45.488855,45.496302,45.526206,45.530871,45.519157,45.525628,45.495372,45.512832,45.504803,45.53318,45.49975,45.457894,45.524673,45.489609,45.490452,45.53229,45.527791,45.444433,45.532408,45.500105,45.523856,45.544681,45.524286,45.527363,45.484489,45.49975,45.535189,45.514145,45.527748,45.504803,45.536408,45.48204,45.550316,45.450851,45.4829,45.521492,45.520688,45.538799,45.515634,45.549911,45.577136,45.551886,45.527839,45.532977,45.494872,45.547408,45.508567,45.478796,45.485889,45.475743,45.520796,45.55001,45.47968,45.480073,45.533661,45.474565,45.491415,45.52741,45.614519,45.501974,45.506371,45.552923,45.550769,45.540881,45.553491,45.527513,45.576963,45.472668,45.479483,45.533195,45.509864,45.504347,45.515617,45.486219,45.520295,45.486971,45.5332,45.532674,45.53536,45.478228,45.547283,45.51069,45.533706,45.514108,45.560762,45.498673,45.477249,45.522318,45.506843,45.543699,45.544471,45.576944,45.437914,45.547218,45.5079,45.51339,45.52516,45.539292,45.456086,45.491415,45.53092,45.532514,45.500606,45.568225,45.469027,45.5341,45.514905,45.558063,45.527727,45.52589,45.523276,45.484859,45.50919,45.551584,45.547179,45.521094,45.547625,45.481491,45.529686,45.500392,45.474224,45.559884,45.532955,45.55001,45.545776,45.529135,45.544377,45.519581,45.537041,45.492825,45.517333,45.50038,45.485385,45.497055,45.477353,45.536423,45.51164,45.529219,45.538984,45.546408,45.52511,45.497595,45.525021,45.512933,45.477313,45.539824,45.53227,45.582757,45.521495,45.53611,45.547997,45.557192,45.538021,45.512797,45.51299,45.444408,45.558063,45.466569,45.543376,45.482098,45.530351,45.570081,45.501302,45.553262,45.4743,45.547227,45.532227,45.494579,45.54467,45.503666,45.557921,45.536738,45.539829,45.501402,45.579349,45.534727,45.525048,45.487723,45.526128,45.510086,45.51383,45.502786,45.520179,45.523035,45.519153,45.506421,45.527839,45.486971,45.543763,45.500035,45.52798,45.519146,45.535189,45.51066,45.5341,45.49975,45.516091,45.500234,45.524394,45.539461,45.500675,45.522413,45.534136,45.467666,45.523854,45.540686,45.532281,45.509685,45.503475,45.477353,45.551584,45.531117,45.516091,45.579325,45.523217,45.485801,45.527795,45.567352,45.539293,45.478228,45.50233,45.569789,45.543651,45.502786,45.523194,45.530351,45.53576,45.47968,45.534311,45.523575,45.505312,45.559199,45.547218,45.541435,45.550525,45.500674,45.443351,45.54467,45.529993,45.496302,45.4816,45.537159,45.48916,45.5536,45.532013,45.481184,45.53923,45.510086,45.56169,45.546539,45.524566,45.538887,45.549714,45.486971,45.49659,45.53543,45.511673,45.538738,45.550476,45.56842,45.480846,45.53565,45.537964,45.525914,45.537088,45.52457,45.53848,45.516779,45.493034,45.535295,45.54257,45.529802,45.530871,45.516079,45.524296,45.489865,45.514396,45.527616,45.514565,45.522233,45.54111,45.545425,45.541406,45.467369,45.536261,45.5367,45.532955,45.556708,45.515604,45.506445,45.499191,45.525669,45.461078,45.520278,45.523584,45.601913,45.501449,45.506448,45.510016,45.552842,45.466914,45.549704,45.510042,45.536738,45.501967,45.531944,45.54206,45.538589,45.563872,45.520188,45.529686,45.486971,45.489476,45.502054,45.530871,45.500451,45.531164,45.51941,45.551937,45.53248,45.547172,45.5468,45.527839,45.570081,45.5387,45.513178,45.539259,45.534134,45.482826,45.54475,45.519139,45.542119,45.46714,45.5359,45.513741,45.50233,45.51166,45.516427,45.560742,45.550698,45.471162,45.564811,45.531177,45.489113,45.495962,45.560742,45.524963,45.50176,45.480208,45.493029,45.547304,45.457597,45.504501,45.501399,45.510086,45.506421,45.429471,45.5195,45.544828,45.507144,45.5294,45.502386,45.498151,45.512832,45.524037,45.511007,45.541198,45.587531,45.552298,45.533924,45.527616,45.53707,45.5294,45.553579,45.448262,45.554961,45.553215,45.513986,45.555205,45.543541,45.51829,45.544136,45.50501,45.501165,45.486919,45.450851,45.540881,45.511007,45.51496,45.511119,45.53536,45.517062,45.499757,45.545776,45.550737,45.50176,45.470366,45.472668,45.52951,45.522278,45.466914,45.493029,45.552923,45.534185,45.557192,45.53057,45.499757,45.475743,45.549963,45.429711,45.547648,45.456351,45.527041,45.547362,45.515299,45.549686,45.498639,45.522866,45.549704,45.527933,45.544376,45.508978,45.564937,45.493034,45.519216,45.493569,45.540881,45.522907,45.532331,45.563921,45.519581,45.523319,45.52687,45.497595,45.544867,45.49123,45.471162,45.53696,45.456885,45.54475,45.512587,45.533661,45.552443,45.539663,45.549714,45.547397,45.505011,45.533924,45.547227,45.523413,45.526712,45.559842,45.592803,45.457597,45.607616,45.497411,45.549911,45.53848,45.491687,45.461685,45.552923,45.49642,45.611231,45.550684,45.500779,45.561594,45.537964,45.530365,45.553027,45.554961,45.489609,45.496302,45.537226,45.56003,45.516876,45.435996,45.52697,45.544846,45.550223,45.497697,45.495709,45.517921,45.523026,45.54816,45.5332,45.477429,45.527738,45.507662,45.512614,45.531834,45.550784,45.530816,45.640739,45.570081,45.509685,45.546408,45.575707,45.527616,45.508567,45.514565,45.533706,45.456823,45.63138,45.494198,45.520178,45.521116,45.549868,45.537553,45.525643,45.512734,45.500043,45.467369,45.54167,45.49486,45.556262,45.506448,45.437914,45.529219,45.535854,45.497515,45.557513,45.482826,45.520524,45.52687,45.540396,45.54257,45.53569,45.483006,45.539824,45.444043,45.539833,45.533348,45.526798,45.471926,45.549598,45.52246,45.546406,45.579325,45.551119,45.529033,45.53227,45.503475,45.524673,45.503063,45.527849,45.49947,45.509685,45.529559,45.529802,45.481491,45.54467,45.549963,45.501402,45.491513,45.526712,45.515617,45.512559,45.547581,45.52114,45.537114,45.507662,45.554152,45.527009,45.556905,45.504242,45.497605,45.549868,45.506775,45.547218,45.496302,45.518977,45.515617,45.456884,45.50235,45.486919,45.56842,45.52479,45.559482,45.426124,45.570081,45.547997,45.549126,45.491687,45.541729,45.510351,45.505263,45.528758,45.505973,45.503982,45.472561,45.52114,45.482826,45.52697,45.518128,45.51829,45.482779,45.493718,45.518341,45.530046,45.507144,45.460729,45.587778,45.540881,45.485889,45.53707,45.546696,45.46714,45.520604,45.503215,45.5566,45.50708,45.517354,45.523217,45.539324,45.533314,45.531818,45.553571,45.468927,45.528,45.523856,45.537813,45.526904,45.52697,45.502145,45.527731,45.466569,45.52729,45.549911,45.541766,45.500035,45.51086,45.534311,45.531917,45.500231,45.520688,45.482779,45.507975,45.553194,45.539807,45.477249,45.533496,45.527758,45.52114,45.446934,45.435665,45.49803,45.550723,45.45073,45.527791,45.490874,45.550316,45.435794,45.505403,45.553139,45.515299,45.491513,45.539737,45.509655,45.457597,45.518593,45.512813,45.529155,45.483928,45.54006,45.51941,45.50781,45.54918,45.534882,45.480914,45.53867,45.558457,45.531203,45.512495,45.580315,45.482779,45.534453,45.538021,45.522586,45.544376,45.529971,45.527513,45.515901,45.431471,45.51735,45.520271,45.52114,45.53248,45.537964,45.5499,45.481367,45.489186,45.505377,45.5536,45.544377,45.500685,45.482852,45.504014,45.516783,45.477353,45.537226,45.50235,45.52445,45.494044,45.529408,45.489951,45.534563,45.540627,45.524505,45.557192,45.541247,45.432894,45.52697,45.521461,45.539224,45.553194,45.52931,45.495323,45.502054,45.466965,45.463254,45.543712,45.530505,45.484018,45.461078,45.470544,45.456358,45.543424,45.521461,45.46714,45.4703,45.522233,45.459707,45.548004,45.51069,45.515065,45.500832,45.5171,45.519373,45.495581,45.559828,45.491513,45.538799,45.494805,45.49368,45.516818,45.527426,45.437914,45.54078,45.477313,45.530871,45.54111,45.53318,45.50297,45.53576,45.515038,45.461685,45.498578,45.539632,45.540154,45.520578,45.521496,45.507629,45.560742,45.482778,45.549695,45.470696,45.527041,45.541484,45.448262,45.563947,45.523575,45.4891,45.531367,45.509418,45.527738,45.529971,45.542794,45.434898,45.531938,45.518593,45.53703,45.525933,45.537877,45.489186,45.553898,45.513303,45.5306,45.520056,45.526027,45.556708,45.48204,45.47993,45.554931,45.486279,45.490628,45.537964,45.544867,45.58116,45.528746,45.52689,45.464893,45.487531,45.477107,45.524673,45.551937,45.485889,45.452858,45.505305,45.515092,45.492897,45.482098,45.486452,45.5367,45.527616,45.545425,45.514905,45.49871,45.50235,45.470696,45.505305,45.521069,45.515617,45.525931,45.509418,45.501441,45.536123,45.556905,45.457509,45.50971,45.535295,45.543583,45.523544,45.559842,45.5566,45.507402,45.519088,45.543651,45.553262,45.556905,45.462362,45.547283,45.606235,45.51908,45.525914,45.449901,45.524287,45.518128,45.497697,45.51339,45.53229,45.537877,45.529802,45.508439,45.541447,45.539259,45.484018,45.516818,45.515172,45.511638,45.498468,45.51484,45.523217,45.55325,45.524673,45.533195,45.551173,45.526252,45.477923,45.486219,45.515172,45.472668,45.515617,45.530871,45.523615,45.542055,45.530347,45.53092,45.51252,45.54172,45.487723,45.529559,45.525177,45.546939,45.539632,45.529033,45.493569,45.43074,45.546124,45.512541,45.54046,45.534557,45.50038,45.497697,45.533661,45.538799,45.539833,45.537114,45.567352,45.536785,45.547329,45.544594,45.478432,45.524025,45.49642,45.54006,45.531955,45.550692,45.497024,45.500234,45.467666,45.472486,45.501302,45.549121,45.51889,45.525643,45.552923,45.470696,45.546978,45.47993,45.51496,45.525021,45.5079,45.513741,45.521116,45.546036,45.541729,45.430543,45.527791,45.488855,45.505721,45.575054,45.512894,45.567352,45.519396,45.5536,45.459488,45.521495,45.577136,45.520188,45.543763,45.551119,45.52353,45.456355,45.5478,45.477429,45.507097,45.525021,45.527363,45.524286,45.52697,45.558281,45.53057,45.481491,45.502828,45.527231,45.485385,45.521051,45.543347,45.532978,45.517354,45.487693,45.486919,45.545776,45.4771,45.516792,45.550723,45.525983,45.51941,45.525109,45.531367,45.562509,45.54006,45.525115,45.478432,45.516873,45.521156,45.523171,45.500674,45.562668,45.535753,45.538544,45.541766,45.426124,45.494805,45.477249,45.558376,45.550056,45.471743,45.547283,45.532372,45.5265,45.532436,45.500606,45.502786,45.470303,45.554931,45.535834,45.641557,45.50038,45.53848,45.453016,45.546978,45.535506,45.522894,45.51908,45.510016,45.510142,45.547295,45.513405,45.533924,45.5566,45.447916,45.523854,45.53057,45.519139,45.562509,45.52089,45.535295,45.515384,45.537041,45.55896,45.550306,45.536408,45.557921,45.51484,45.544819,45.509932,45.51086,45.503858,45.539224,45.521565,45.531985,45.54006,45.563872,45.58077,45.521342,45.508205,45.537114,45.471743,45.487693,45.546939,45.4952,45.522225,45.521161,45.528449,45.530027,45.508567,45.541766,45.56833,45.56169,45.5367,45.47188,45.547997,45.49803,45.51059,45.550723,45.54006,45.480495,45.579406,45.50501,45.540951,45.504561,45.572789,45.459986,45.492838,45.434434,45.5367,45.502828,45.511638,45.533409,45.472668,45.48204,45.541729,45.527758,45.534136,45.496265,45.497047,45.495894,45.53227,45.502525,45.541729,45.523171,45.456351,45.51383,45.557895,45.493986,45.501399,45.529993,45.534185,45.5452,45.517583,45.535295,45.527795,45.507144,45.480495,45.509813,45.557895,45.523854,45.53536,45.535295,45.510142,45.484947,45.550692,45.533348,45.544311,45.54078,45.537877,45.523507,45.543347,45.50345,45.447044,45.544599,45.503005,45.486971,45.440347,45.558481,45.500685,45.491415,45.497092,45.499326,45.480073,45.559482,45.508141,45.52697,45.605902,45.525172,45.560742,45.493029,45.550306,45.514767,45.55001,45.534882,45.500832,45.535247,45.525933,45.531367,45.478228,45.51829,45.512994,45.536123,45.541247,45.531327,45.504095,45.51166,45.456355,45.534563,45.53927,45.550668,45.500451,45.51069,45.5028,45.475743,45.52087,45.478228,45.52087,45.526543,45.51086,45.460729,45.528631,45.519895,45.527933,45.482941,45.558281,45.491226,45.524037,45.509188,45.519581,45.536153,45.544513,45.50297,45.536273,45.484896,45.495955,45.50781,45.51829,45.522225,45.520923,45.5215,45.522278,45.51908,45.529802,45.554152,45.541198,45.522907,45.520271,45.472668,45.52931,45.501302,45.485889,45.52841,45.51164,45.495581,45.520604,45.50345,45.422451,45.527748,45.504635,45.473868,45.437914,45.553879,45.539984,45.539259,45.50297,45.483928,45.541955,45.460729,45.522586,45.533349,45.592596,45.471926,45.519157,45.499757,45.540627,45.523304,45.5452,45.517354,45.502602,45.456365,45.528446,45.580315,45.533851,45.550316,45.5864,45.501715,45.497011,45.482941,45.493569,45.534185,45.500779,45.53543,45.541586,45.559842,45.456365,45.514385,45.544149,45.551912,45.471743,45.477923,45.537527,45.573541,45.534727,45.50015,45.54475,45.548931,45.525931,45.54206,45.551585,45.522413,45.524505,45.52353,45.542868,45.538308,45.5265,45.5028,45.537226,45.461685,45.502292,45.485048,45.507662,45.531401,45.484859,45.544377,45.527932,45.462742,45.540944,45.528449,45.51086,45.524628,45.517354,45.530027,45.527363,45.516937,45.515616,45.522342,45.527758,45.521342,45.538738,45.463001,45.539824,45.525669,45.537259,45.521081,45.433557,45.517,45.519396,45.543651,45.471743,45.51533,45.495789,45.48204,45.5195,45.50708,45.53092,45.509759,45.575707,45.456365,45.512587,45.516427,45.492825,45.53133,45.579325,45.547408,45.595875,45.50723,45.4891,45.515299,45.498639,45.543763,45.531834,45.49486,45.533661,45.51889,45.41631,45.50708,45.54918,45.53073,45.533195,45.536178,45.4981,45.513917,45.545604,45.47993,45.520688,45.501929,45.508567,45.438367,45.526206,45.535614,45.457597,45.523194,45.518215,45.510086,45.51941,45.514108,45.532013,45.503982,45.522413,45.52729,45.477923,45.533703,45.50038,45.550613,45.534453,45.450851,45.505973,45.495709,45.527363,45.533924,45.53318,45.445272,45.534727,45.496302,45.507078,45.542485,45.513741,45.553194,45.498578,45.538799,45.534358,45.550899,45.519088,45.477923,45.475743,45.456358,45.497974,45.489951,45.529971,45.512734,45.417147,45.475193,45.545776,45.542579,45.557895,45.472503,45.47968,45.555205,45.50501,45.512797,45.51941,45.564353,45.53217,45.589175,45.514403,45.501974,45.502326,45.532977,45.497411,45.551582,45.534136,45.492897,45.493718,45.464877,45.542972,45.540138,45.547161,45.497974,45.457597,45.497718,45.551192,45.510042,45.553215,45.551886,45.5478,45.52974,45.509229,45.541198,45.444433,45.557192,45.527041,45.544594,45.493034,45.531611,45.510351,45.544846,45.51066,45.553571,45.536785,45.514087,45.527432,45.573733,45.54717,45.501715,45.573798,45.58116,45.53318,45.52689,45.532265,45.568225,45.533348,45.536299,45.508979,45.550056,45.51908,45.537114,45.571798,45.51674,45.501715,45.53133,45.492825,45.546939,45.557895,45.55325,45.466965,45.550684,45.472503,45.552649,45.506373,45.49486,45.52089,45.529134,45.459707,45.525931,45.503805,45.555004,45.4891,45.520188,45.532926,45.535614,45.532008,45.53867,45.53703,45.595352,45.5452,45.50206,45.531401,45.487531,45.485385,45.472503,45.508205,45.51829,45.53543,45.527201,45.489476,45.479483,45.523575,45.470544,45.567766,45.466569,45.521116,45.550692,45.576,45.453319,45.453246,45.51533,45.514379,45.523035,45.539632,45.508144,45.518967,45.545459,45.4952,45.564944,45.494872,45.4918,45.531674,45.485631,45.54918,45.501302,45.527009,45.459488,45.521094,45.536261,45.53092,45.489545,45.531818,45.527727,45.516876,45.525021,45.484018,45.509685,45.536582,45.544284,45.546907,45.479483,45.506876,45.553579,45.499326,45.489609,45.480134,45.518253,45.534882,45.527616,45.491513,45.486971,45.524566,45.498673,45.52974,45.536261,45.4952,45.545542,45.537813,45.506194,45.53057,45.528746,45.53217,45.545955,45.57179,45.515288,45.493718,45.564353,45.529559,45.526235,45.516181,45.557192,45.534795,45.497697,45.514087,45.50206,45.551585,45.460729,45.532408,45.557771,45.540396,45.492227,45.508144,45.486971,45.4743,45.531164,45.553215,45.471882,45.450837,45.540944,45.532994,45.556562,45.518309,45.527009,45.515616,45.532408,45.450851,45.553879,45.51754,45.50206,45.515384,45.547405,45.550784,45.524505,45.529802,45.55197,45.548859,45.470696,45.439824,45.555241,45.544846,45.517333,45.54178,45.528366,45.495323,45.538799,45.537964,45.547648,45.496563,45.50038,45.514995,45.498639,45.513303,45.470303,45.470696,45.456358,45.461078,45.547304,45.512541,45.532132,45.456085,45.539932,45.456049,45.537159,45.523319,45.546661,45.520018,45.519505,45.484896,45.521069,45.546661,45.51066,45.53229,45.5028,45.530221,45.495789,45.561396,45.483006,45.54816,45.537891,45.538738,45.596796,45.464877,45.505359,45.547227,45.5219,45.513303,45.519157,45.478968,45.515616,45.531939,45.518967,45.530816,45.489609,45.482779,45.515634,45.493034,45.553914,45.524683,45.429471,45.509328,45.53227,45.537114,45.553879,45.527839,45.549833,45.505063,45.522878,45.50623,45.50206,45.52479,45.46714,45.534298,45.492504,45.521156,45.470303,45.539679,45.524286,45.564353,45.474138,45.433251,45.53696,45.534377,45.515172,45.520018,45.550476,45.52951,45.551585,45.515617,45.552649,45.532514,45.546978,45.554564,45.587531,45.494514,45.536856,45.504407,45.507629,45.521565,45.456365,45.471669,45.5452,45.544377,45.552123,45.531401,45.549933,45.559482,45.519376,45.513338,45.480208,45.506421,45.50623,45.516783,45.52353,45.515299,45.524683,45.520458,45.491754,45.537026,45.559199,45.550698,45.516873,45.557487,45.521485,45.543712,45.512601,45.494514,45.55325,45.547558,45.570081,45.52457,45.50206,45.521156,45.524673,45.58077,45.462846,45.527513,45.554133,45.528758,45.569789,45.538403,45.484896,45.549457,45.457509,45.537114,45.556751,45.43284,45.479483,45.459593,45.50931,45.532265,45.544136,45.556708,45.489951,45.438056,45.53248,45.475142,45.52974,45.518593,45.50015,45.522586,45.54178,45.531917,45.462362,45.557921,45.519895,45.559199,45.526856,45.503738,45.55224,45.570657,45.484081,45.52729,45.556262,45.472503,45.553914,45.523194,45.502292,45.524505,45.533548,45.54006,45.522233,45.541435,45.482693,45.509569,45.50723,45.525048,45.546406,45.529802,45.509864,45.516897,45.494564,45.486319,45.462726,45.589175,45.498639,45.534134,45.497011,45.5341,45.523854,45.548931,45.497515,45.5265,45.513741,45.51561,45.542983,45.51164,45.524566,45.489951,45.522365,45.49975,45.544594,45.529971,45.528746,45.50781,45.517,45.486452,45.478889,45.453319,45.515007,45.486319,45.541955,45.549126,45.538984,45.518215,45.556591,45.548186,45.5079,45.52029,45.527009,45.501715,45.536123,45.51164,45.545581,45.559482,45.531401,45.6175,45.500234,45.495894,45.436568,45.557192,45.500674,45.537088,45.505312,45.52697,45.479483,45.559482,45.542119,45.593985,45.553215,45.549933,45.517459,45.540686,45.524518,45.502813,45.521293,45.561594,45.51066,45.528088,45.5536,45.494981,45.536378,45.525983,45.504095,45.477923,45.58116,45.4829,45.520604,45.579349,45.468927,45.576259,45.51059,45.531818,45.559199,45.557192,45.546797,45.501041,45.53867,45.563921,45.422451,45.553219,45.522586,45.51735,45.46714,45.513303,45.510072,45.508356,45.51069,45.502786,45.444433,45.501402,45.558109,45.52353,45.538461,45.504591,45.50235,45.541586,45.516818,45.554214,45.554862,45.528724,45.506448,45.542485,45.559496,45.50971,45.493034,45.539932,45.564944,45.54523,45.560762,45.45843,45.510351,45.498151,45.504245,45.459233,45.553491,45.524277,45.539224,45.504242,45.514335,45.531939,45.531939,45.553262,45.51735,45.514726,45.53867,45.49606,45.50623,45.506448,45.499677,45.52353,45.527748,45.477429,45.519515,45.516873,45.51405,45.480495,45.49947,45.524037,45.496457,45.521564,45.523525,45.537628,45.523575,45.50761,45.529686,45.52479,45.510042,45.50449,45.509229,45.53057,45.5499,45.507078,45.521769,45.486919,45.54078,45.519146,45.54458,45.52765,45.549911,45.532008,45.507838,45.500234,45.522278,45.567352,45.487601,45.531164,45.477429,45.554214,45.505382,45.533496,45.518309,45.512614,45.45073,45.519132,45.499745,45.520541,45.51569,45.50708,45.474224,45.43155,45.495581,45.526712,45.53543,45.523172,45.453016,45.554931,45.545425,45.513409,45.539857,45.53308,45.5427,45.51791,45.512283,45.539032,45.526765,45.520178,45.5306,45.503215,45.494514,45.521565,45.53308,45.513917,45.52114,45.564353,45.505314,45.559,45.477249,45.546408,45.552364,45.503727,45.512541,45.44691,45.532013,45.520019,45.511119,45.4816,45.48047,45.539726,45.516664,45.492463,45.538799,45.483928,45.483221,45.503475,45.490042,45.529936,45.512014,45.499237,45.542485,45.531401,45.53073,45.51889,45.561594,45.523413,45.559872,45.4816,45.515617,45.532674,45.549755,45.52689,45.513413,45.550769,45.500951,45.506511,45.51069,45.500606,45.525669,45.529512,45.499677,45.496583,45.467369,45.523584,45.531475,45.503727,45.53703,45.48983,45.502326,45.588082,45.50931,45.539932,45.537114,45.450851,45.4743,45.522586,45.51069,45.463257,45.549783,45.53092,45.499757,45.536178,45.557895,45.50176,45.53217,45.50076,45.549833,45.518263,45.530351,45.520578,45.523276,45.50781,45.502294,45.481291,45.53923,45.491322,45.544828,45.494499,45.522365,45.523525,45.543699,45.526543,45.529693,45.522278,45.522426,45.52746,45.460729,45.481291,45.475743,45.546539,45.550056,45.544311,45.539032,45.4688,45.521565,45.523507,45.52974,45.486279,45.539385,45.492504,45.472503,45.5367,45.529802,45.477605,45.514734,45.546978,45.533682,45.549695,45.569789,45.563947,45.52353,45.511638,45.485889,45.487723,45.526235,45.529559,45.520222,45.498112,45.521685,45.553408,45.505377,45.527738,45.478796,45.49803,45.508439,45.53045,45.539824,45.540396,45.500208,45.544599,45.559018,45.552123,45.568225,45.505386,45.507717,45.552123,45.550768,45.529326,45.564944,45.505973,45.506445,45.512861,45.551584,45.508356,45.560742,45.50723,45.51069,45.50233,45.537026,45.558481,45.588356,45.54006,45.533409,45.433124,45.514565,45.523492,45.481291,45.519957,45.549714,45.494564,45.500035,45.572789,45.553215,45.504263,45.503738,45.478888,45.47393,45.494564,45.491722,45.527231,45.596797,45.505403,45.526252,45.508144,45.5262,45.529219,45.46982,45.541247,45.526557,45.520188,45.507402,45.518253,45.52114,45.524634,45.485048,45.524634,45.49486,45.434434,45.506251,45.539833,45.547362,45.531412,45.506373,45.56169,45.523544,45.49659,45.537114,45.4891,45.474448,45.489476,45.545604,45.55287,45.561059,45.538984,45.507144,45.512933,45.49803,45.556905,45.521293,45.500674,45.52056,45.498639,45.541586,45.546406,45.526059,45.456355,45.437914,45.542972,45.484018,45.530816,45.519492,45.539932,45.50297,45.53611,45.53341,45.519216,45.524277,45.51889,45.50501,45.50449,45.592803,45.497974,45.497515,45.517401,45.558481,45.536907,45.493029,45.553215,45.484489,45.515384,45.524505,45.442397,45.52163,45.551585,45.463001,45.4918,45.529693,45.520604,45.56244,45.531834,45.517,45.53867,45.531401,45.510086,45.590892,45.50971,45.524296,45.529002,45.491513,45.500675,45.575707,45.502629,45.530505,45.524037,45.58601,45.521769,45.561396,45.543583,45.53703,45.547362,45.543583,45.526566,45.471162,45.520634,45.514379,45.520578,45.53227,45.534727,45.539632,45.569789,45.563947,45.487991,45.506251,45.513834,45.563947,45.523171,45.494514,45.486241,45.486452,45.448262,45.466569,45.527727,45.517999,45.543452,45.49947,45.494499,45.523171,45.526904,45.460729,45.520018,45.527727,45.45002,45.499326,45.549868,45.531007,45.50723,45.505011,45.4771,45.550692,45.493569,45.521769,45.563679,45.446534,45.549686,45.501041,45.52479,45.514379,45.529993,45.523035,45.5534,45.571627,45.544867,45.456049,45.556751,45.51941,45.512541,45.545955,45.529802,45.575707,45.50761,45.550769,45.551582,45.50367,45.5536,45.537026,45.541,45.514565,45.532281,45.517333,45.514784,45.510072,45.50038,45.531164,45.476344,45.477313,45.523544,45.527363,45.504407,45.5341,45.515299,45.486219,45.5613,45.462362,45.519088,45.532977,45.544513,45.506843,45.536299,45.514887,45.5215,45.539236,45.558109,45.524037,45.496851,45.524296,45.651406,45.550692,45.587778,45.516777,45.552443,45.500685,45.497595,45.51378,45.544284,45.509361,45.494044,45.5367,45.569176,45.520295,45.466936,45.524628,45.505721,45.543763,45.53045,45.507144,45.49123,45.525983,45.51908,45.506373,45.472668,45.49975,45.555241,45.47993,45.502326,45.501195,45.467666,45.55001,45.537114,45.559199,45.512734,45.534136,45.541816,45.522719,45.499191,45.520688,45.497165,45.501967,45.471743,45.5219,45.56844,45.450851,45.498606,45.527698,45.483928,45.578332,45.506373,45.493029,45.550613,45.522403,45.525669,45.539737,45.52951,45.519237,45.467369,45.480208,45.519376,45.534071,45.513413,45.527432],"legendgroup":"","lon":[-73.559507,-73.58247,-73.60953,-73.575984,-73.539285,-73.638428,-73.581775,-73.589848,-73.574863,-73.610536,-73.601136,-73.567401,-73.571564,-73.584146,-73.563176,-73.565413,-73.563818,-73.605834,-73.576105,-73.618575,-73.571768,-73.615222,-73.551557,-73.562251,-73.566931,-73.595775,-73.638615,-73.552645,-73.583972,-73.558826,-73.618323,-73.558713,-73.631628,-73.569651,-73.634448,-73.550345,-73.544766,-73.581748,-73.55458,-73.638135,-73.614251,-73.619682,-73.605566,-73.6212,-73.579436,-73.624293,-73.685104,-73.598233,-73.57917,-73.626514,-73.586694,-73.55534,-73.565422,-73.55976,-73.612658,-73.522739,-73.54259,-73.544766,-73.555843,-73.638428,-73.55566,-73.577338,-73.601728,-73.56625,-73.512642,-73.65884,-73.56295,-73.579254,-73.579221,-73.711481,-73.57921,-73.597793,-73.579917,-73.559164,-73.57917,-73.614888,-73.58661,-73.512261,-73.517166,-73.546804,-73.56463,-73.656367,-73.573837,-73.592421,-73.580643,-73.562187,-73.577215,-73.631103,-73.629822,-73.576349,-73.597793,-73.556141,-73.597476,-73.595333,-73.596421,-73.539446,-73.563949,-73.559411,-73.564063,-73.615372,-73.57348,-73.571776,-73.57523,-73.551958,-73.602868,-73.572607,-73.617842,-73.638428,-73.57851,-73.561728,-73.572103,-73.560097,-73.624392,-73.57264,-73.607605,-73.620313,-73.591,-73.56353,-73.58852,-73.579918,-73.568404,-73.588338,-73.588928,-73.623531,-73.547791,-73.571877,-73.543961,-73.56846,-73.589307,-73.64013,-73.578897,-73.548912,-73.601544,-73.621166,-73.586013,-73.577215,-73.577859,-73.6198,-73.562264,-73.569531,-73.568576,-73.561082,-73.604865,-73.546289,-73.581222,-73.576349,-73.58247,-73.584072,-73.605652,-73.56507,-73.624835,-73.552563,-73.57641,-73.623505,-73.589758,-73.55976,-73.559006,-73.553861,-73.614841,-73.615293,-73.573739,-73.552645,-73.541082,-73.558212,-73.577615,-73.563757,-73.573047,-73.626375,-73.585268,-73.571776,-73.617977,-73.55535,-73.573047,-73.508752,-73.579186,-73.627002,-73.55859,-73.612278,-73.564729,-73.60556,-73.577853,-73.57029,-73.532427,-73.570657,-73.560891,-73.592144,-73.55447,-73.626672,-73.57616,-73.56928,-73.560765,-73.572092,-73.634073,-73.580444,-73.632267,-73.541874,-73.549502,-73.58781,-73.551557,-73.548081,-73.671125,-73.57718,-73.56846,-73.556154,-73.575515,-73.57264,-73.574621,-73.564201,-73.578296,-73.61544,-73.581566,-73.545609,-73.575984,-73.601728,-73.575559,-73.56073,-73.604059,-73.61979,-73.589365,-73.573747,-73.614708,-73.531242,-73.584151,-73.577599,-73.575621,-73.633568,-73.580536,-73.616469,-73.636557,-73.595478,-73.564818,-73.562175,-73.612441,-73.579349,-73.555119,-73.620661,-73.590559,-73.573864,-73.576767,-73.578755,-73.610991,-73.574772,-73.629298,-73.582445,-73.527503,-73.574131,-73.508752,-73.571963,-73.577639,-73.568471,-73.576456,-73.5822,-73.610871,-73.591155,-73.57544,-73.616469,-73.620911,-73.554431,-73.572575,-73.566175,-73.57689,-73.571963,-73.611274,-73.605566,-73.566014,-73.561846,-73.60184,-73.624036,-73.57069,-73.55534,-73.5156,-73.60101,-73.557103,-73.689113,-73.562175,-73.588928,-73.622443,-73.578312,-73.576287,-73.582129,-73.577312,-73.575559,-73.583368,-73.578271,-73.565752,-73.57867,-73.61199,-73.543956,-73.576107,-73.6,-73.578849,-73.567336,-73.515261,-73.572199,-73.649599,-73.540764,-73.546611,-73.562871,-73.59527,-73.62332,-73.629776,-73.589419,-73.596436,-73.600251,-73.608428,-73.541,-73.571387,-73.524577,-73.576502,-73.61887,-73.682621,-73.588069,-73.629942,-73.589,-73.560172,-73.597476,-73.565837,-73.644442,-73.614841,-73.564139,-73.592144,-73.59201,-73.637924,-73.561395,-73.543956,-73.603251,-73.589115,-73.585195,-73.651781,-73.522739,-73.570654,-73.666814,-73.598346,-73.585268,-73.508752,-73.620911,-73.571939,-73.607798,-73.571564,-73.623295,-73.565338,-73.57348,-73.585643,-73.54983,-73.593858,-73.581185,-73.607664,-73.565338,-73.583836,-73.551958,-73.606381,-73.62151,-73.577837,-73.6198,-73.54983,-73.561141,-73.532428,-73.574131,-73.552197,-73.573599,-73.59155,-73.630089,-73.548081,-73.59181,-73.567528,-73.5592,-73.584006,-73.554786,-73.5851,-73.551554,-73.571391,-73.55785,-73.557254,-73.563848,-73.579436,-73.629459,-73.58523,-73.551739,-73.567907,-73.583974,-73.560941,-73.551958,-73.610737,-73.596758,-73.576826,-73.568949,-73.560116,-73.604735,-73.585643,-73.558112,-73.580994,-73.580498,-73.512778,-73.586694,-73.537441,-73.56353,-73.654884,-73.583836,-73.585616,-73.574772,-73.638135,-73.571391,-73.581088,-73.583972,-73.495067,-73.626597,-73.588679,-73.684043,-73.632136,-73.57547,-73.556144,-73.555153,-73.613747,-73.580294,-73.593099,-73.55786,-73.57544,-73.572103,-73.581775,-73.558826,-73.571776,-73.617842,-73.586284,-73.567401,-73.603659,-73.60042,-73.571776,-73.579785,-73.607723,-73.624835,-73.711186,-73.584157,-73.623556,-73.600306,-73.566663,-73.583524,-73.595478,-73.574975,-73.56762,-73.56671,-73.57614,-73.542476,-73.653502,-73.56385,-73.614824,-73.572092,-73.6162,-73.556669,-73.61529,-73.623556,-73.559652,-73.56906,-73.617459,-73.633051,-73.495067,-73.569576,-73.561851,-73.57821,-73.576413,-73.632136,-73.537441,-73.597927,-73.57591,-73.58677,-73.544674,-73.583859,-73.638532,-73.564754,-73.564601,-73.620187,-73.590843,-73.551085,-73.600874,-73.57821,-73.600874,-73.56328,-73.524577,-73.641017,-73.610871,-73.571877,-73.629593,-73.605834,-73.573986,-73.576815,-73.614888,-73.568755,-73.584039,-73.531116,-73.55347,-73.56671,-73.615293,-73.571768,-73.623526,-73.571585,-73.580498,-73.560412,-73.553469,-73.6162,-73.520127,-73.57917,-73.586694,-73.56328,-73.568039,-73.559038,-73.581855,-73.561916,-73.607669,-73.57264,-73.567559,-73.578444,-73.576997,-73.60101,-73.583695,-73.560269,-73.571564,-73.61199,-73.612233,-73.57547,-73.622544,-73.610512,-73.607773,-73.561395,-73.554314,-73.578755,-73.640617,-73.610991,-73.576451,-73.56671,-73.574777,-73.548687,-73.55933,-73.57348,-73.561964,-73.565317,-73.58584,-73.582596,-73.568576,-73.6508,-73.630103,-73.57656,-73.564708,-73.615085,-73.575808,-73.589115,-73.560199,-73.600912,-73.587842,-73.57264,-73.628509,-73.616286,-73.58752,-73.585428,-73.577639,-73.599375,-73.595645,-73.59212,-73.612316,-73.581018,-73.549266,-73.556233,-73.562538,-73.585643,-73.593775,-73.5484,-73.57029,-73.575879,-73.571569,-73.494995,-73.576024,-73.558466,-73.57913,-73.54983,-73.601296,-73.5962,-73.639485,-73.611429,-73.546611,-73.693532,-73.559234,-73.57616,-73.54641,-73.563437,-73.561916,-73.551739,-73.620187,-73.54925,-73.579034,-73.626672,-73.56385,-73.58661,-73.554188,-73.556463,-73.57614,-73.603721,-73.562594,-73.588223,-73.577312,-73.630103,-73.58274,-73.546289,-73.621166,-73.617939,-73.568696,-73.681371,-73.555237,-73.616961,-73.588223,-73.593471,-73.560119,-73.577495,-73.599306,-73.583537,-73.603659,-73.491916,-73.597439,-73.627206,-73.54755,-73.562425,-73.595379,-73.57656,-73.573299,-73.577591,-73.561916,-73.611429,-73.588928,-73.576451,-73.56255,-73.612441,-73.582129,-73.552563,-73.543773,-73.597557,-73.581185,-73.584039,-73.59235,-73.568696,-73.582432,-73.620216,-73.551085,-73.58925,-73.579703,-73.565923,-73.577017,-73.517166,-73.57722,-73.562425,-73.593016,-73.607723,-73.577599,-73.602026,-73.620383,-73.551951,-73.584039,-73.582644,-73.606381,-73.607025,-73.568471,-73.593918,-73.568372,-73.527503,-73.535496,-73.691564,-73.638702,-73.576529,-73.55859,-73.556352,-73.638615,-73.581095,-73.652942,-73.590559,-73.57674,-73.56279,-73.617459,-73.57886,-73.589419,-73.624331,-73.597476,-73.584194,-73.549036,-73.552845,-73.631103,-73.571703,-73.597476,-73.5822,-73.600982,-73.614708,-73.576767,-73.575984,-73.612969,-73.61413,-73.583612,-73.582144,-73.55922,-73.584157,-73.622547,-73.558042,-73.600169,-73.576105,-73.607669,-73.667386,-73.56987,-73.591766,-73.567256,-73.58421,-73.547142,-73.583862,-73.620187,-73.570657,-73.558263,-73.607545,-73.617842,-73.632267,-73.555119,-73.620383,-73.600422,-73.558263,-73.590143,-73.615222,-73.572079,-73.540453,-73.56904,-73.574806,-73.57921,-73.610768,-73.560494,-73.616837,-73.56928,-73.589,-73.57295,-73.558263,-73.57547,-73.638615,-73.589848,-73.601223,-73.597736,-73.567143,-73.622547,-73.54913,-73.555157,-73.597927,-73.57851,-73.618323,-73.535332,-73.586302,-73.6123,-73.56385,-73.573358,-73.583695,-73.59588,-73.544766,-73.591737,-73.559891,-73.570504,-73.541939,-73.538253,-73.57851,-73.569013,-73.579688,-73.557093,-73.583593,-73.576451,-73.565012,-73.622341,-73.567256,-73.596758,-73.565395,-73.615085,-73.581566,-73.593775,-73.605512,-73.610737,-73.573492,-73.576451,-73.554853,-73.573278,-73.582129,-73.640633,-73.57869,-73.571939,-73.595214,-73.569348,-73.600874,-73.554867,-73.570769,-73.600609,-73.583616,-73.61544,-73.613747,-73.578849,-73.565138,-73.585946,-73.583368,-73.571244,-73.538224,-73.551557,-73.622547,-73.587238,-73.546199,-73.584525,-73.631103,-73.579221,-73.584039,-73.612231,-73.580782,-73.599343,-73.546836,-73.559781,-73.658325,-73.57507,-73.57591,-73.560036,-73.59576,-73.574436,-73.586013,-73.57569,-73.566867,-73.581222,-73.597557,-73.573358,-73.575568,-73.572886,-73.605566,-73.58539,-73.593166,-73.563757,-73.57348,-73.614841,-73.57591,-73.56378,-73.610991,-73.571768,-73.564754,-73.597793,-73.621166,-73.58503,-73.56545,-73.57547,-73.602026,-73.590143,-73.585479,-73.57172,-73.569531,-73.582292,-73.596758,-73.597557,-73.564754,-73.539446,-73.560941,-73.61987,-73.57718,-73.59576,-73.613747,-73.587509,-73.581222,-73.610871,-73.567307,-73.60556,-73.574436,-73.583084,-73.590559,-73.532547,-73.56846,-73.624392,-73.56073,-73.562175,-73.5727,-73.577859,-73.562187,-73.614616,-73.570752,-73.555295,-73.581566,-73.57851,-73.55859,-73.599343,-73.555355,-73.577591,-73.616946,-73.692362,-73.56497,-73.621438,-73.632462,-73.548864,-73.571703,-73.612441,-73.574184,-73.634748,-73.620769,-73.577514,-73.577599,-73.60565,-73.606372,-73.591155,-73.607234,-73.634748,-73.572135,-73.5727,-73.559781,-73.597983,-73.595811,-73.601728,-73.560144,-73.554314,-73.562425,-73.559411,-73.570233,-73.55923,-73.560036,-73.623295,-73.622445,-73.577017,-73.592026,-73.621581,-73.57544,-73.629298,-73.597888,-73.581855,-73.529147,-73.561189,-73.554009,-73.57208,-73.553394,-73.56762,-73.524577,-73.59665,-73.599156,-73.618105,-73.490113,-73.554864,-73.584006,-73.54259,-73.599658,-73.558218,-73.577017,-73.609412,-73.548912,-73.54913,-73.574409,-73.565845,-73.628834,-73.618105,-73.565138,-73.566775,-73.599343,-73.566903,-73.5822,-73.621438,-73.573986,-73.610606,-73.571977,-73.607001,-73.569509,-73.574173,-73.564818,-73.636557,-73.559221,-73.515261,-73.58498,-73.603286,-73.601223,-73.640633,-73.570769,-73.569651,-73.57547,-73.603659,-73.555455,-73.617564,-73.601728,-73.600127,-73.548687,-73.572446,-73.58153,-73.559038,-73.575091,-73.623526,-73.574777,-73.624392,-73.636948,-73.618575,-73.634475,-73.569367,-73.61199,-73.558826,-73.613924,-73.576287,-73.593928,-73.608428,-73.613007,-73.603478,-73.58503,-73.597793,-73.549443,-73.556466,-73.560333,-73.601124,-73.560494,-73.605834,-73.575515,-73.646765,-73.631664,-73.576024,-73.538849,-73.565705,-73.619624,-73.579424,-73.56818,-73.582144,-73.602616,-73.567401,-73.586302,-73.573364,-73.573589,-73.556753,-73.57616,-73.56545,-73.559652,-73.597736,-73.638772,-73.55535,-73.579146,-73.624872,-73.565003,-73.65884,-73.490113,-73.560036,-73.629822,-73.568364,-73.6162,-73.577591,-73.566931,-73.611575,-73.570614,-73.737086,-73.572135,-73.531116,-73.495067,-73.5523,-73.574227,-73.603541,-73.659902,-73.546348,-73.57544,-73.552839,-73.608428,-73.577263,-73.560494,-73.591935,-73.563745,-73.558021,-73.57711,-73.559877,-73.56545,-73.563112,-73.571438,-73.605834,-73.544377,-73.579917,-73.552645,-73.602026,-73.620911,-73.600982,-73.602026,-73.622547,-73.598276,-73.551836,-73.6162,-73.624392,-73.56936,-73.564776,-73.56279,-73.543827,-73.534434,-73.570129,-73.491915,-73.604981,-73.539782,-73.595797,-73.563437,-73.580782,-73.59527,-73.568814,-73.562594,-73.55793,-73.588069,-73.569348,-73.590446,-73.600422,-73.614841,-73.667386,-73.577784,-73.56987,-73.576888,-73.574131,-73.564063,-73.585616,-73.579186,-73.552563,-73.575559,-73.57867,-73.587602,-73.582432,-73.551836,-73.638135,-73.571564,-73.633051,-73.565491,-73.602616,-73.638615,-73.589848,-73.599279,-73.618652,-73.554477,-73.583593,-73.629822,-73.54925,-73.598345,-73.55786,-73.581775,-73.568589,-73.576888,-73.5612,-73.61544,-73.55566,-73.567528,-73.58255,-73.582445,-73.560172,-73.57544,-73.572056,-73.575788,-73.564776,-73.560199,-73.600127,-73.610161,-73.604973,-73.607723,-73.628834,-73.55566,-73.617459,-73.620302,-73.597439,-73.5612,-73.512776,-73.574863,-73.573358,-73.594027,-73.591,-73.570472,-73.608428,-73.552496,-73.606711,-73.558263,-73.546289,-73.74444,-73.594821,-73.581222,-73.571124,-73.607669,-73.577698,-73.576424,-73.577514,-73.564601,-73.603271,-73.59576,-73.56385,-73.585479,-73.552197,-73.624036,-73.569955,-73.603935,-73.612245,-73.555157,-73.564201,-73.565752,-73.57711,-73.584157,-73.54829,-73.598791,-73.546836,-73.58539,-73.61979,-73.601173,-73.576105,-73.568949,-73.614991,-73.573747,-73.614824,-73.589293,-73.5156,-73.571776,-73.603635,-73.569651,-73.625377,-73.57805,-73.552179,-73.57523,-73.650561,-73.552563,-73.587238,-73.606931,-73.681757,-73.553673,-73.567251,-73.540453,-73.58274,-73.631103,-73.563112,-73.56255,-73.599247,-73.541031,-73.581937,-73.569955,-73.57674,-73.584811,-73.565448,-73.579436,-73.609537,-73.58852,-73.619682,-73.529408,-73.586284,-73.650034,-73.558218,-73.571585,-73.660321,-73.561916,-73.551739,-73.562251,-73.612025,-73.564754,-73.601728,-73.578168,-73.603721,-73.633568,-73.584194,-73.59576,-73.562175,-73.559234,-73.581018,-73.560116,-73.602026,-73.557978,-73.574436,-73.57507,-73.628142,-73.618653,-73.615372,-73.570998,-73.562184,-73.574777,-73.582445,-73.638428,-73.560206,-73.555237,-73.610737,-73.63389,-73.615327,-73.508752,-73.56828,-73.542443,-73.596758,-73.576287,-73.58523,-73.569847,-73.583524,-73.561462,-73.682498,-73.575568,-73.529408,-73.623295,-73.632267,-73.574876,-73.624392,-73.573047,-73.633161,-73.638615,-73.6237,-73.56987,-73.560618,-73.558756,-73.621581,-73.569576,-73.529147,-73.565167,-73.620235,-73.571814,-73.581681,-73.614793,-73.560036,-73.569083,-73.546002,-73.611429,-73.560432,-73.559134,-73.579424,-73.565395,-73.616469,-73.556466,-73.594821,-73.589293,-73.56772,-73.618105,-73.623883,-73.569241,-73.617459,-73.56497,-73.58852,-73.55566,-73.570129,-73.571126,-73.611423,-73.576056,-73.565354,-73.58413,-73.595478,-73.593918,-73.519677,-73.653376,-73.610991,-73.562317,-73.572103,-73.615372,-73.561916,-73.632233,-73.570129,-73.570504,-73.558187,-73.595797,-73.571939,-73.653787,-73.62078,-73.569651,-73.566497,-73.548081,-73.605651,-73.559134,-73.555843,-73.624392,-73.57867,-73.56385,-73.549969,-73.623443,-73.560891,-73.599658,-73.631103,-73.554314,-73.582982,-73.561082,-73.716905,-73.621581,-73.604214,-73.629822,-73.6,-73.597476,-73.567771,-73.6616,-73.580541,-73.579349,-73.541082,-73.611429,-73.610512,-73.598665,-73.572079,-73.623907,-73.563848,-73.589293,-73.57851,-73.5822,-73.562042,-73.628022,-73.590446,-73.746873,-73.583862,-73.57172,-73.562643,-73.689112,-73.593218,-73.59588,-73.60556,-73.577403,-73.583836,-73.55116,-73.592144,-73.591766,-73.598346,-73.607798,-73.604847,-73.564731,-73.561766,-73.589115,-73.568407,-73.602082,-73.54755,-73.556527,-73.630071,-73.570769,-73.607545,-73.56081,-73.584194,-73.670634,-73.572135,-73.556352,-73.569421,-73.593016,-73.567307,-73.568216,-73.560941,-73.524419,-73.633051,-73.576349,-73.570654,-73.565295,-73.631704,-73.605512,-73.624835,-73.565167,-73.555153,-73.713605,-73.597873,-73.586302,-73.655568,-73.590559,-73.601728,-73.589293,-73.584566,-73.573465,-73.598346,-73.661428,-73.583695,-73.58685,-73.593928,-73.58507,-73.618521,-73.588893,-73.594821,-73.573047,-73.586222,-73.713061,-73.577459,-73.573492,-73.590843,-73.5949,-73.557866,-73.622547,-73.54259,-73.622442,-73.538382,-73.566497,-73.562136,-73.558112,-73.65041,-73.60953,-73.627206,-73.595804,-73.632136,-73.647013,-73.573254,-73.65041,-73.575559,-73.57128,-73.577599,-73.564818,-73.543827,-73.590529,-73.617652,-73.571786,-73.611429,-73.556466,-73.593099,-73.560269,-73.610458,-73.555119,-73.5178,-73.573338,-73.577609,-73.576888,-73.590143,-73.567602,-73.590943,-73.531907,-73.707383,-73.562425,-73.589115,-73.61529,-73.5178,-73.662083,-73.577856,-73.610737,-73.58752,-73.534294,-73.555823,-73.600982,-73.59235,-73.545609,-73.57069,-73.620548,-73.569429,-73.594027,-73.584157,-73.567602,-73.58503,-73.567974,-73.603635,-73.559781,-73.629298,-73.562175,-73.615222,-73.57128,-73.589816,-73.58539,-73.61068,-73.577591,-73.631704,-73.564818,-73.565752,-73.573589,-73.569847,-73.54913,-73.629298,-73.564601,-73.534434,-73.633883,-73.665566,-73.582292,-73.593471,-73.579221,-73.561273,-73.591155,-73.574227,-73.568377,-73.605512,-73.601223,-73.595645,-73.698564,-73.555591,-73.583836,-73.57718,-73.568576,-73.584157,-73.568364,-73.620187,-73.548912,-73.560116,-73.520127,-73.626723,-73.555237,-73.614841,-73.580782,-73.627206,-73.61199,-73.572607,-73.5949,-73.573739,-73.552197,-73.603002,-73.634475,-73.563848,-73.607664,-73.572028,-73.562425,-73.56987,-73.623526,-73.584838,-73.615447,-73.510466,-73.590529,-73.509822,-73.578271,-73.558263,-73.60556,-73.576767,-73.584039,-73.565752,-73.57616,-73.531242,-73.615293,-73.558826,-73.585643,-73.562643,-73.584393,-73.532382,-73.610737,-73.582445,-73.629822,-73.495067,-73.681336,-73.57946,-73.680111,-73.558021,-73.610536,-73.705467,-73.568646,-73.576952,-73.567168,-73.60184,-73.592026,-73.5156,-73.600169,-73.576249,-73.57886,-73.573726,-73.553478,-73.5341,-73.581626,-73.500826,-73.573047,-73.562317,-73.638428,-73.561562,-73.589115,-73.577698,-73.568407,-73.552179,-73.546804,-73.506857,-73.563748,-73.579424,-73.594784,-73.647657,-73.589939,-73.581748,-73.561141,-73.560412,-73.570769,-73.607604,-73.57108,-73.537441,-73.576349,-73.58274,-73.574777,-73.572055,-73.552571,-73.601795,-73.590843,-73.567697,-73.626723,-73.616818,-73.592144,-73.565923,-73.562187,-73.508752,-73.613752,-73.575559,-73.605834,-73.624213,-73.58247,-73.541874,-73.691564,-73.598421,-73.570504,-73.540764,-73.54641,-73.56828,-73.572103,-73.58255,-73.573986,-73.591861,-73.57591,-73.562317,-73.587602,-73.57029,-73.564754,-73.621581,-73.534434,-73.571814,-73.633649,-73.584838,-73.614991,-73.677011,-73.665234,-73.54926,-73.571003,-73.57886,-73.547494,-73.585766,-73.58925,-73.553469,-73.55535,-73.647657,-73.567742,-73.631103,-73.629822,-73.583632,-73.614991,-73.572607,-73.566583,-73.569429,-73.746873,-73.56545,-73.535496,-73.657416,-73.573047,-73.58523,-73.600912,-73.576767,-73.565494,-73.556508,-73.57048,-73.57868,-73.629593,-73.62284,-73.569989,-73.54926,-73.590843,-73.60261,-73.561851,-73.59235,-73.584525,-73.579186,-73.616647,-73.58677,-73.555119,-73.634073,-73.569354,-73.584157,-73.577514,-73.61529,-73.588738,-73.54259,-73.575984,-73.569979,-73.5373,-73.56917,-73.582129,-73.558187,-73.595878,-73.583737,-73.565317,-73.560494,-73.61987,-73.619236,-73.600127,-73.624872,-73.624331,-73.558021,-73.573427,-73.557538,-73.623295,-73.56463,-73.558263,-73.626128,-73.618105,-73.54983,-73.549969,-73.597983,-73.569394,-73.608428,-73.584525,-73.563072,-73.539814,-73.687266,-73.587238,-73.566931,-73.576185,-73.54926,-73.572199,-73.605652,-73.552665,-73.609571,-73.572575,-73.572056,-73.562389,-73.573358,-73.68019,-73.621166,-73.602711,-73.561273,-73.633649,-73.614256,-73.554009,-73.590529,-73.581566,-73.577017,-73.559411,-73.577312,-73.60897,-73.58685,-73.57208,-73.593858,-73.609412,-73.56846,-73.56936,-73.65884,-73.632147,-73.550878,-73.535314,-73.584525,-73.559652,-73.583524,-73.612658,-73.595645,-73.563053,-73.598791,-73.559948,-73.671205,-73.56906,-73.614849,-73.54926,-73.58507,-73.562643,-73.583084,-73.564319,-73.5719,-73.56762,-73.6616,-73.581018,-73.572156,-73.630344,-73.704427,-73.611423,-73.615372,-73.495067,-73.566583,-73.605174,-73.654495,-73.578154,-73.567091,-73.554787,-73.630089,-73.594142,-73.569847,-73.591044,-73.596853,-73.558021,-73.570614,-73.573837,-73.539814,-73.588657,-73.573887,-73.573465,-73.631664,-73.575879,-73.628284,-73.614708,-73.563437,-73.567307,-73.63168,-73.64013,-73.632462,-73.570614,-73.54259,-73.6162,-73.602082,-73.571438,-73.58139,-73.57805,-73.55923,-73.57246,-73.57921,-73.551958,-73.553711,-73.658325,-73.633649,-73.552496,-73.638772,-73.579146,-73.554188,-73.579918,-73.58274,-73.630401,-73.615327,-73.598346,-73.54755,-73.61544,-73.57505,-73.57867,-73.559201,-73.584039,-73.560242,-73.634448,-73.623531,-73.567733,-73.596827,-73.551876,-73.65041,-73.584621,-73.554647,-73.565422,-73.593471,-73.612441,-73.577856,-73.655573,-73.623443,-73.57656,-73.562624,-73.561358,-73.576249,-73.563053,-73.618105,-73.670819,-73.617842,-73.581566,-73.58584,-73.595253,-73.618323,-73.5719,-73.571387,-73.572961,-73.614616,-73.677952,-73.577859,-73.670634,-73.574863,-73.555921,-73.610711,-73.573599,-73.628878,-73.562643,-73.614841,-73.641243,-73.578654,-73.57264,-73.566998,-73.56507,-73.621438,-73.58255,-73.593928,-73.577514,-73.589249,-73.581002,-73.581142,-73.580294,-73.574876,-73.595234,-73.56081,-73.589115,-73.556527,-73.619682,-73.56671,-73.566583,-73.565422,-73.581002,-73.563757,-73.614991,-73.598839,-73.561358,-73.560144,-73.622445,-73.58925,-73.639485,-73.57354,-73.55116,-73.6284,-73.587332,-73.615447,-73.5373,-73.578444,-73.569509,-73.605651,-73.638615,-73.58925,-73.59665,-73.625377,-73.628784,-73.5727,-73.689112,-73.600251,-73.604973,-73.561851,-73.568646,-73.56255,-73.57544,-73.618323,-73.591766,-73.672965,-73.565,-73.577338,-73.563437,-73.554188,-73.581002,-73.574635,-73.568434,-73.584779,-73.558187,-73.54822,-73.58255,-73.601173,-73.541011,-73.589758,-73.559038,-73.573747,-73.581002,-73.58539,-73.614991,-73.598346,-73.592457,-73.597793,-73.566988,-73.57674,-73.57062,-73.565177,-73.569083,-73.587602,-73.595446,-73.602616,-73.634448,-73.54641,-73.568576,-73.591911,-73.646647,-73.570677,-73.540303,-73.55075,-73.57507,-73.568646,-73.552197,-73.552496,-73.575559,-73.569139,-73.653787,-73.614888,-73.625329,-73.537795,-73.624899,-73.600422,-73.57616,-73.60897,-73.59816,-73.656367,-73.559223,-73.571126,-73.593918,-73.58355,-73.633161,-73.600874,-73.56353,-73.581748,-73.565752,-73.565422,-73.575515,-73.555921,-73.58503,-73.610737,-73.563112,-73.538382,-73.594784,-73.646599,-73.565494,-73.669024,-73.572056,-73.583593,-73.629459,-73.538932,-73.561248,-73.653787,-73.583227,-73.6616,-73.572092,-73.596758,-73.546289,-73.590559,-73.56772,-73.540764,-73.55199,-73.597557,-73.584867,-73.600169,-73.568442,-73.610737,-73.607723,-73.604973,-73.60261,-73.583159,-73.54913,-73.564754,-73.527793,-73.586726,-73.628142,-73.571137,-73.632353,-73.581222,-73.582129,-73.56904,-73.569429,-73.562175,-73.6212,-73.564239,-73.609571,-73.577808,-73.58685,-73.560206,-73.562624,-73.61656,-73.60897,-73.586006,-73.624899,-73.554175,-73.585589,-73.573431,-73.561082,-73.692362,-73.557941,-73.576133,-73.626128,-73.657416,-73.638772,-73.587238,-73.658844,-73.534576,-73.613924,-73.625377,-73.606432,-73.545753,-73.606655,-73.565448,-73.559134,-73.589848,-73.610711,-73.587021,-73.500375,-73.57507,-73.60556,-73.571915,-73.575515,-73.603882,-73.601816,-73.5727,-73.570654,-73.624752,-73.578999,-73.562594,-73.562425,-73.5373,-73.583819,-73.519677,-73.54913,-73.557866,-73.61656,-73.559221,-73.55116,-73.606408,-73.602026,-73.548687,-73.573353,-73.512776,-73.529147,-73.584779,-73.62151,-73.563807,-73.54983,-73.571094,-73.573837,-73.535332,-73.580675,-73.60897,-73.655568,-73.544766,-73.589419,-73.568814,-73.569139,-73.613924,-73.56904,-73.602616,-73.56328,-73.606687,-73.585616,-73.551085,-73.563176,-73.577698,-73.626128,-73.566663,-73.610512,-73.56081,-73.53798,-73.58523,-73.552665,-73.57547,-73.609571,-73.60897,-73.577827,-73.602589,-73.57069,-73.626375,-73.56279,-73.540313,-73.543956,-73.55642,-73.586694,-73.56081,-73.527793,-73.574635,-73.570657,-73.58539,-73.574863,-73.565494,-73.576185,-73.595478,-73.621544,-73.643216,-73.560097,-73.56828,-73.580337,-73.565494,-73.573431,-73.582292,-73.560432,-73.576529,-73.563279,-73.571786,-73.604214,-73.573589,-73.576451,-73.726614,-73.55116,-73.571939,-73.555119,-73.577827,-73.563895,-73.576529,-73.519677,-73.603635,-73.55116,-73.624752,-73.56036,-73.656367,-73.605834,-73.545549,-73.630401,-73.618323,-73.576413,-73.632353,-73.568404,-73.602178,-73.588223,-73.573857,-73.589293,-73.702313,-73.583304,-73.572156,-73.569955,-73.575549,-73.571768,-73.585479,-73.535496,-73.57493,-73.60261,-73.53437,-73.599375,-73.65041,-73.564818,-73.573353,-73.565491,-73.59576,-73.609412,-73.572459,-73.617712,-73.595253,-73.562624,-73.569651,-73.59235,-73.682499,-73.622445,-73.591045,-73.597888,-73.568696,-73.562136,-73.597557,-73.554786,-73.620879,-73.549502,-73.661428,-73.57805,-73.55917,-73.564601,-73.55922,-73.569651,-73.55922,-73.598233,-73.54983,-73.634073,-73.677312,-73.596827,-73.601223,-73.579742,-73.583159,-73.58763,-73.590143,-73.55458,-73.560116,-73.622421,-73.571564,-73.57505,-73.578197,-73.560119,-73.62186,-73.57208,-73.59235,-73.606687,-73.559177,-73.5851,-73.577591,-73.5727,-73.591766,-73.547494,-73.590943,-73.568364,-73.614849,-73.58539,-73.588657,-73.633161,-73.577514,-73.517166,-73.562184,-73.553711,-73.575984,-73.568404,-73.601194,-73.597439,-73.704432,-73.604538,-73.58274,-73.551761,-73.638135,-73.577338,-73.57505,-73.577312,-73.629942,-73.634073,-73.612658,-73.601136,-73.566175,-73.58247,-73.55786,-73.629298,-73.630089,-73.558,-73.576451,-73.582129,-73.527503,-73.57614,-73.618702,-73.535314,-73.578912,-73.573358,-73.595218,-73.574131,-73.618575,-73.579742,-73.568576,-73.573589,-73.558826,-73.5822,-73.612481,-73.615447,-73.57614,-73.561732,-73.66752,-73.579211,-73.613924,-73.559038,-73.575124,-73.574923,-73.614793,-73.56928,-73.5949,-73.616837,-73.598839,-73.597873,-73.599165,-73.58413,-73.594142,-73.55199,-73.538224,-73.654884,-73.545753,-73.55917,-73.495067,-73.584039,-73.573252,-73.571963,-73.57886,-73.612674,-73.571585,-73.581018,-73.623805,-73.565845,-73.547142,-73.551085,-73.54983,-73.595811,-73.582129,-73.563176,-73.607723,-73.640483,-73.575808,-73.721679,-73.576185,-73.589419,-73.628021,-73.571569,-73.508752,-73.593016,-73.57917,-73.562275,-73.684043,-73.589,-73.583227,-73.605651,-73.613924,-73.559148,-73.582596,-73.574863,-73.560269,-73.56917,-73.57674,-73.576107,-73.561562,-73.57614,-73.573739,-73.558112,-73.557978,-73.59155,-73.570504,-73.607669,-73.523528,-73.615085,-73.57656,-73.561273,-73.574227,-73.56772,-73.553478,-73.57108,-73.552197,-73.56353,-73.612916,-73.56917,-73.593858,-73.58153,-73.601173,-73.57641,-73.652942,-73.560765,-73.63474,-73.555921,-73.608428,-73.571451,-73.577698,-73.715467,-73.54925,-73.566014,-73.590529,-73.555843,-73.603251,-73.611429,-73.58685,-73.57523,-73.580541,-73.62284,-73.58413,-73.56463,-73.559038,-73.515283,-73.57507,-73.582883,-73.559652,-73.594027,-73.629593,-73.576952,-73.607723,-73.562425,-73.61544,-73.624484,-73.614793,-73.629822,-73.561395,-73.636342,-73.538382,-73.539814,-73.560242,-73.5887,-73.612231,-73.62108,-73.569509,-73.559038,-73.564601,-73.64013,-73.556141,-73.567091,-73.563053,-73.561141,-73.644225,-73.580943,-73.562175,-73.636557,-73.576529,-73.539285,-73.56385,-73.555823,-73.57069,-73.561463,-73.58685,-73.571244,-73.55859,-73.539656,-73.575082,-73.555157,-73.565138,-73.581222,-73.578271,-73.599156,-73.595478,-73.580294,-73.579186,-73.626597,-73.617939,-73.614028,-73.543773,-73.556141,-73.590529,-73.56818,-73.540872,-73.624835,-73.58752,-73.74444,-73.584867,-73.59527,-73.55447,-73.590943,-73.575788,-73.569847,-73.593471,-73.537795,-73.583836,-73.61335,-73.556508,-73.610536,-73.56497,-73.560494,-73.614888,-73.5523,-73.579917,-73.539446,-73.618041,-73.574131,-73.550345,-73.641243,-73.61544,-73.57264,-73.611063,-73.579436,-73.605834,-73.607773,-73.698564,-73.534576,-73.5727,-73.571003,-73.54155,-73.577703,-73.57413,-73.59155,-73.557978,-73.602616,-73.576529,-73.54822,-73.631664,-73.615293,-73.539285,-73.581185,-73.524577,-73.57108,-73.559221,-73.559234,-73.571438,-73.598839,-73.568179,-73.555355,-73.57656,-73.590559,-73.56378,-73.566014,-73.580444,-73.56936,-73.58584,-73.55859,-73.576451,-73.56295,-73.612674,-73.56507,-73.628142,-73.539285,-73.568814,-73.59235,-73.5822,-73.564729,-73.584566,-73.61979,-73.623443,-73.63168,-73.579688,-73.623295,-73.594784,-73.656367,-73.646765,-73.576775,-73.576815,-73.559148,-73.610871,-73.565406,-73.634448,-73.574772,-73.583616,-73.556669,-73.56328,-73.577639,-73.571124,-73.584006,-73.565413,-73.628765,-73.593858,-73.633161,-73.585766,-73.572092,-73.562251,-73.607546,-73.57674,-73.564139,-73.565317,-73.586284,-73.57946,-73.610737,-73.563437,-73.562317,-73.616225,-73.54534,-73.551557,-73.61979,-73.623592,-73.662083,-73.571768,-73.582445,-73.585908,-73.588338,-73.609412,-73.589115,-73.633649,-73.589293,-73.572079,-73.552563,-73.59527,-73.607545,-73.56328,-73.634748,-73.624872,-73.569968,-73.54913,-73.578654,-73.55859,-73.626992,-73.564315,-73.581249,-73.579186,-73.571244,-73.587602,-73.54895,-73.608543,-73.569847,-73.609546,-73.568646,-73.5523,-73.56295,-73.599165,-73.634073,-73.564776,-73.584138,-73.616818,-73.584782,-73.574772,-73.589293,-73.6237,-73.583695,-73.58752,-73.640022,-73.572446,-73.547142,-73.584414,-73.670115,-73.603286,-73.585766,-73.575808,-73.564776,-73.594027,-73.551761,-73.597832,-73.56295,-73.606408,-73.544675,-73.5341,-73.594142,-73.57029,-73.593775,-73.616611,-73.565422,-73.694878,-73.550822,-73.610536,-73.574436,-73.555714,-73.551171,-73.573887,-73.552496,-73.562643,-73.665566,-73.564917,-73.57507,-73.559169,-73.574227,-73.572961,-73.589848,-73.565422,-73.64013,-73.567307,-73.543827,-73.570677,-73.55876,-73.581937,-73.599343,-73.571877,-73.597476,-73.520127,-73.588684,-73.579463,-73.598529,-73.560119,-73.563757,-73.588684,-73.56497,-73.57544,-73.55917,-73.576997,-73.582596,-73.601544,-73.562187,-73.592026,-73.618215,-73.628021,-73.535179,-73.626597,-73.567741,-73.56987,-73.55534,-73.572961,-73.55786,-73.619599,-73.575808,-73.553439,-73.583616,-73.581626,-73.582445,-73.584525,-73.606711,-73.583836,-73.571419,-73.578897,-73.593099,-73.554347,-73.56828,-73.571003,-73.551761,-73.594821,-73.640633,-73.570681,-73.577495,-73.55976,-73.56295,-73.56545,-73.54259,-73.612172,-73.581088,-73.585589,-73.589848,-73.634588,-73.604973,-73.571244,-73.603597,-73.693962,-73.61199,-73.612278,-73.581002,-73.579463,-73.590446,-73.61068,-73.599165,-73.614991,-73.581185,-73.584811,-73.575515,-73.596157,-73.531907,-73.583368,-73.579373,-73.572543,-73.551876,-73.57043,-73.57614,-73.582644,-73.576451,-73.581018,-73.630103,-73.612674,-73.647645,-73.535496,-73.551951,-73.57295,-73.577599,-73.556466,-73.55976,-73.611423,-73.55199,-73.561273,-73.578897,-73.567575,-73.583972,-73.593166,-73.599658,-73.60953,-73.554175,-73.588749,-73.58498,-73.628284,-73.57374,-73.583368,-73.54822,-73.575472,-73.573047,-73.59588,-73.56295,-73.585589,-73.58255,-73.544766,-73.565922,-73.598791,-73.560226,-73.57868,-73.548081,-73.561964,-73.560119,-73.541649,-73.639485,-73.571003,-73.667162,-73.693492,-73.61979,-73.544377,-73.554431,-73.611063,-73.545609,-73.670634,-73.567091,-73.672739,-73.58507,-73.623505,-73.59527,-73.581566,-73.56928,-73.612658,-73.555714,-73.597983,-73.59665,-73.529147,-73.596827,-73.599658,-73.626672,-73.568106,-73.544127,-73.651842,-73.560918,-73.56463,-73.537441,-73.539285,-73.571419,-73.555843,-73.573252,-73.594142,-73.566867,-73.60897,-73.602082,-73.554314,-73.634813,-73.564137,-73.615085,-73.560036,-73.598421,-73.591766,-73.576105,-73.563949,-73.638702,-73.570233,-73.565959,-73.539656,-73.574227,-73.573492,-73.618575,-73.58852,-73.519677,-73.616837,-73.552571,-73.545753,-73.538382,-73.57569,-73.606381,-73.562184,-73.572079,-73.567091,-73.584146,-73.55566,-73.537795,-73.563053,-73.578654,-73.57208,-73.589,-73.595234,-73.581989,-73.576775,-73.619788,-73.570233,-73.629942,-73.600912,-73.582445,-73.603251,-73.652676,-73.59201,-73.563112,-73.568119,-73.585766,-73.574131,-73.622445,-73.562184,-73.6349,-73.535496,-73.612674,-73.606011,-73.571126,-73.560097,-73.70837,-73.569847,-73.561082,-73.593219,-73.560891,-73.60261,-73.61979,-73.535496,-73.622547,-73.637672,-73.58752,-73.647645,-73.597927,-73.653376,-73.571977,-73.576024,-73.589365,-73.585643,-73.56497,-73.585863,-73.6616,-73.577733,-73.512642,-73.577808,-73.568696,-73.559038,-73.641243,-73.591,-73.575984,-73.581681,-73.61987,-73.538312,-73.57547,-73.565317,-73.599658,-73.569847,-73.602664,-73.577178,-73.56936,-73.548912,-73.601194,-73.539782,-73.612658,-73.56906,-73.54259,-73.572961,-73.570752,-73.580498,-73.57805,-73.559134,-73.575788,-73.571814,-73.719597,-73.55199,-73.577679,-73.559188,-73.566583,-73.612481,-73.554188,-73.55156,-73.532717,-73.583859,-73.576349,-73.636342,-73.698918,-73.57354,-73.583836,-73.599343,-73.577639,-73.53725,-73.650561,-73.59606,-73.556508,-73.577609,-73.553545,-73.637405,-73.54829,-73.58929,-73.573837,-73.553469,-73.556464,-73.553439,-73.553439,-73.638615,-73.56906,-73.565486,-73.56936,-73.57348,-73.55976,-73.576349,-73.578849,-73.55199,-73.597439,-73.600169,-73.598517,-73.554175,-73.647256,-73.577827,-73.57591,-73.590143,-73.620155,-73.570367,-73.587509,-73.553394,-73.623443,-73.551836,-73.601728,-73.56545,-73.624835,-73.5592,-73.55447,-73.54913,-73.583084,-73.561395,-73.534859,-73.569429,-73.630401,-73.569241,-73.588069,-73.6123,-73.558263,-73.580444,-73.563136,-73.571126,-73.577591,-73.653787,-73.565185,-73.583695,-73.600169,-73.55156,-73.567634,-73.566931,-73.603286,-73.573726,-73.572575,-73.55789,-73.579034,-73.567751,-73.560966,-73.56917,-73.603721,-73.67112,-73.553711,-73.584838,-73.5822,-73.618097,-73.571915,-73.610711,-73.556527,-73.533554,-73.620383,-73.694597,-73.59212,-73.567143,-73.561826,-73.614318,-73.730423,-73.579424,-73.614616,-73.569979,-73.583368,-73.535332,-73.694597,-73.560765,-73.54926,-73.571244,-73.567554,-73.658414,-73.587238,-73.638428,-73.602868,-73.561704,-73.570677,-73.60363,-73.580541,-73.618907,-73.567974,-73.6,-73.611216,-73.677165,-73.57722,-73.560784,-73.552496,-73.577312,-73.580764,-73.572103,-73.567111,-73.604096,-73.532547,-73.569442,-73.636342,-73.612674,-73.58153,-73.56353,-73.585643,-73.623526,-73.615541,-73.6,-73.614991,-73.571776,-73.605566,-73.57264,-73.538253,-73.57711,-73.574578,-73.576669,-73.57805,-73.565448,-73.593016,-73.517691,-73.578849,-73.576502,-73.570769,-73.560941,-73.592421,-73.561704,-73.58584,-73.589678,-73.565138,-73.632223,-73.554431,-73.599343,-73.571003,-73.594027,-73.6237,-73.612658,-73.57805,-73.575621,-73.70631,-73.57674,-73.629298,-73.57641,-73.576529,-73.57128,-73.55859,-73.570483,-73.640633,-73.595775,-73.624392,-73.567733,-73.558218,-73.57208,-73.573219,-73.60033,-73.541082,-73.58781,-73.610458,-73.574173,-73.584146,-73.587509,-73.553673,-73.598233,-73.567401,-73.577591,-73.574184,-73.624175,-73.634073,-73.60033,-73.564601,-73.598665,-73.534576,-73.545549,-73.614318,-73.6198,-73.535332,-73.576413,-73.59527,-73.573599,-73.541,-73.581088,-73.539285,-73.56081,-73.57029,-73.573775,-73.691449,-73.575515,-73.515261,-73.554647,-73.548081,-73.655573,-73.55199,-73.574635,-73.577514,-73.569083,-73.54895,-73.587602,-73.629856,-73.577615,-73.574942,-73.569531,-73.56762,-73.576249,-73.576424,-73.552665,-73.672965,-73.587638,-73.508752,-73.616818,-73.571138,-73.588223,-73.548481,-73.630103,-73.579436,-73.570432,-73.57167,-73.630103,-73.549427,-73.574806,-73.577639,-73.629593,-73.556352,-73.561595,-73.561916,-73.580498,-73.65041,-73.615085,-73.57805,-73.566497,-73.593166,-73.583304,-73.631712,-73.60897,-73.570657,-73.681405,-73.568407,-73.56073,-73.60033,-73.554817,-73.563848,-73.638702,-73.618105,-73.540313,-73.569496,-73.571523,-73.568106,-73.581989,-73.604735,-73.638702,-73.57689,-73.586726,-73.535179,-73.621166,-73.589758,-73.574772,-73.549266,-73.574777,-73.616961,-73.591045,-73.598276,-73.590559,-73.578444,-73.588338,-73.54926,-73.578755,-73.571963,-73.578755,-73.57108,-73.586694,-73.571391,-73.575559,-73.579221,-73.591749,-73.524577,-73.610512,-73.587332,-73.57851,-73.571003,-73.57656,-73.604031,-73.584566,-73.63474,-73.565631,-73.603967,-73.582445,-73.555119,-73.63389,-73.552665,-73.58925,-73.589365,-73.561082,-73.58582,-73.574227,-73.612481,-73.598421,-73.612969,-73.597557,-73.58274,-73.617939,-73.563437,-73.581626,-73.685104,-73.599343,-73.57505,-73.576287,-73.601124,-73.57718,-73.58929,-73.56353,-73.57069,-73.5592,-73.510466,-73.556141,-73.552571,-73.574605,-73.583304,-73.511914,-73.564818,-73.58752,-73.628834,-73.606408,-73.594142,-73.71765,-73.562348,-73.599165,-73.571569,-73.584006,-73.567401,-73.575984,-73.595333,-73.553478,-73.589,-73.56936,-73.612674,-73.611429,-73.640617,-73.57354,-73.604847,-73.546199,-73.633649,-73.565354,-73.561562,-73.52775,-73.614708,-73.590143,-73.521838,-73.534859,-73.601544,-73.6284,-73.58584,-73.579221,-73.6284,-73.585428,-73.627206,-73.563906,-73.610871,-73.567733,-73.56828,-73.614793,-73.634448,-73.548081,-73.655573,-73.569632,-73.571391,-73.560446,-73.655573,-73.573431,-73.583368,-73.595728,-73.595234,-73.577856,-73.623295,-73.586284,-73.568184,-73.60101,-73.57591,-73.574173,-73.573431,-73.624331,-73.634073,-73.579463,-73.586284,-73.600306,-73.571768,-73.647657,-73.598799,-73.615085,-73.572028,-73.6212,-73.656367,-73.568576,-73.534859,-73.571605,-73.603541,-73.591155,-73.577178,-73.56545,-73.610871,-73.604214,-73.565395,-73.662255,-73.592048,-73.614841,-73.571877,-73.667162,-73.58685,-73.570677,-73.626992,-73.57029,-73.561562,-73.551836,-73.57711,-73.599156,-73.61848,-73.6616,-73.593166,-73.558263,-73.568407,-73.610991,-73.574436,-73.565338,-73.570752,-73.57507,-73.583695,-73.606259,-73.615327,-73.587332,-73.607723,-73.572543,-73.58852,-73.561273,-73.573747,-73.60143,-73.59665,-73.569509,-73.581222,-73.571564,-73.681757,-73.607773,-73.578276,-73.5851,-73.585946,-73.719598,-73.590143,-73.62332,-73.604847,-73.500413,-73.656367,-73.569354,-73.577784,-73.603002,-73.572156,-73.555237,-73.632463,-73.54534,-73.573864,-73.654495,-73.56081,-73.636293,-73.614824,-73.631531,-73.595811,-73.629459,-73.56772,-73.587638,-73.555119,-73.580782,-73.577808,-73.5727,-73.524577,-73.58539,-73.55566,-73.550822,-73.555921,-73.565138,-73.620625,-73.593918,-73.59576,-73.569139,-73.599658,-73.561141,-73.595478,-73.626112,-73.577204,-73.569421,-73.608428,-73.55933,-73.555153,-73.613924,-73.55534,-73.675202,-73.594027,-73.569228,-73.586253,-73.577312,-73.655954,-73.524577,-73.564818,-73.582883,-73.581177,-73.593016,-73.614256,-73.61068,-73.577215,-73.570769,-73.577599,-73.551951,-73.562651,-73.538253,-73.579917],"marker":{"color":"#636efa","opacity":0.1},"mode":"markers+text","name":"","showlegend":false,"subplot":"mapbox","text":["Ottawa \u002f Peel","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Chabot \u002f Everett","Duluth \u002f St-Denis","Jacques-Le Ber \u002f de la Pointe Nord","Lajeunesse \u002f Cr\u00e9mazie","Resther \u002f du Mont-Royal","Palm \u002f St-Remi","Duvernay \u002f Charlevoix","Chambord \u002f Jean-Talon","de Bellechasse \u002f de St-Vallier","Gauthier \u002f Papineau","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Laval \u002f du Mont-Royal","de Bordeaux \u002f Sherbrooke","Gauthier \u002f de Lorimier","Notre-Dame \u002f Jean-d'Estr\u00e9es","Drolet \u002f Beaubien","Hutchison \u002f Prince-Arthur","Parc Jean-Brillant (Swail \u002f Decelles)","Cypress \u002f Peel","Marquette \u002f Villeray","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Montcalm \u002f Ontario","des \u00c9rables \u002f Rachel","Hutchison \u002f Edouard Charles","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","St-Andr\u00e9 \u002f St-Antoine","Chomedey \u002f de Maisonneuve","McGill \u002f des R\u00e9collets","Henri-Julien \u002f de Castelnau","Fullum \u002f de Rouen","Benny \u002f de Monkland","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Guizot \u002f St-Laurent","CHSLD Benjamin-Victor-Rousselot (Dickson \u002f Sherbrooke)","de Repentigny \u002f Sherbrooke","Resther \u002f du Mont-Royal","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Clark \u002f de Li\u00e8ge","M\u00e9tro Jean-Talon (Berri \u002f Jean-Talon)","Dunlop \u002f Van Horne","de Bordeaux \u002f Jean-Talon","Monkland \u002f Girouard","30e avenue \u002f St-Zotique","De Gasp\u00e9 \u002f Villeray","Ste-Croix \u002f Tass\u00e9","St-Dominique \u002f St-Viateur","Laurier \u002f Bordeaux","LaSalle \u002f 67e avenue","LaSalle \u002f S\u00e9n\u00e9cal","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","Rushbrooke \u002f  Caisse","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Bloomfield \u002f Van Horne","Duchesneau \u002f de Marseille","Place du Commerce","de Repentigny \u002f Sherbrooke","Logan \u002f de Champlain","M\u00e9tro Cr\u00e9mazie (Cr\u00e9mazie \u002f Lajeunesse)","McGill \u002f Place d'Youville","Parthenais \u002f Laurier","de Gasp\u00e9 \u002f Marmier","Dorion \u002f Gauthier","St-Charles \u002f St-Sylvestre","Georges-Baril \u002f Fleury","Square Victoria (Viger \u002f du Square-Victoria)","Laurier \u002f de Bordeaux","3e avenue \u002f Dandurand","Cavendish \u002f Poirier","Duluth \u002f St-Laurent","Fabre \u002f Beaubien","Boyer \u002f du Mont-Royal","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Laurier \u002f de Bordeaux","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","Laporte \u002f St-Antoine","des Ormeaux \u002f Notre-Dame","Place Longueuil","Elgar \u002f de l'\u00cele-des-S\u0153urs","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","M\u00e9tro Sauv\u00e9 (Berri \u002f Sauv\u00e9)","Fullum \u002f Gilford","St-Andr\u00e9 \u002f St-Gr\u00e9goire","University \u002f des Pins","Shearer \u002f Centre","Laval \u002f Duluth","Leman \u002f de Chateaubriand","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","Milton \u002f Universit\u00e9","Fabre \u002f Beaubien","Queen \u002f Ottawa","de la Roche \u002f  de Bellechasse","16e avenue \u002f Jean-Talon","Monk \u002f Jacques-Hertel","Bossuet \u002f Pierre-de-Coubertin","Gare d'autocars de Montr\u00e9al (Berri \u002f Ontario)","Larivi\u00e8re \u002f de Lorimier","Ottawa \u002f William","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Crescent \u002f Ren\u00e9-L\u00e9vesque","Cartier \u002f Marie-Anne","Guilbault \u002f Clark","Plessis \u002f Ren\u00e9-L\u00e9vesque","des \u00c9cores \u002f Jean-Talon","4e avenue \u002f de Verdun","Alexandra \u002f Waverly","Lajeunesse \u002f Cr\u00e9mazie","Mackay \u002fde Maisonneuve (Sud)","Murray \u002f William","de Maisonneuve \u002f Robert-Bourassa","Ann \u002f William","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Rachel \u002f Br\u00e9beuf","de la Roche \u002f B\u00e9langer","Faillon \u002f St-Denis","Hillside \u002f Ste-Catherine","St-Andr\u00e9 \u002f Ontario","de Br\u00e9beuf \u002f St-Gr\u00e9goire","Boyer \u002f du Mont-Royal","Square Phillips 2","de l'Esplanade \u002f du Mont-Royal","Jeanne Mance \u002f du Mont-Royal","Drolet \u002f Villeray","M\u00e9tro Viau (Pierre-de-Coubertin \u002f Sicard)","5e avenue \u002f de Verdun","Parc de Dieppe","Ropery \u002f Augustin-Cantin","Gerry-Boulet \u002f St-Gr\u00e9goire","West Broadway \u002f Sherbrooke","Marie-Anne \u002f St-Hubert","de Marseille \u002f Viau","Fran\u00e7ois-Perrault \u002f L.-O.-David","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Lincoln \u002f Lambert Closse","Laval \u002f Duluth","Mentana \u002f Marie-Anne","C\u00f4te St-Antoine \u002f Royal","Beaudry \u002f Ontario","Laurier \u002f 15e avenue","Lucien L'Allier \u002f St-Jacques","St-Jacques \u002f Gauvin","Henri-Julien \u002f de Bellechasse","Cadillac \u002f Sherbrooke","Garnier \u002f St-Joseph","Milton \u002f Universit\u00e9","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Parc St-Paul (Marc-Sauvalle \u002f Le Caron)","C\u00e9gep Andr\u00e9-Laurendeau","des Bassins \u002f Richmond","Wilderton  \u002f Van Horne","de la Commune \u002f McGill","Bordeaux \u002fGilford","d'Oxford \u002f de Monkland","Drolet \u002f Laurier","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","des R\u00e9collets \u002f McGill","Alexandre-DeS\u00e8ve \u002f Ste-Catherine","De La Roche \u002f Everett","Marquette \u002f Villeray","Prince-Arthur \u002f St-Urbain","St-Andr\u00e9 \u002f St-Antoine","Ste-Catherine \u002f D\u00e9z\u00e9ry","Dorion \u002f Ontario","Crescent \u002f de Maisonneuve","Square Ahmerst \u002f Wolfe","35e avenue \u002f Beaubien","Drolet \u002f Gounod","de Bullion \u002f du Mont-Royal","Cartier \u002f Marie-Anne","M\u00e9tro Universit\u00e9 de Montr\u00e9al (\u00c9douard-Montpetit \u002f Louis-Collin)","Queen \u002f Wellington","35e avenue \u002f Beaubien","St-Charles \u002f Charlotte","St-Mathieu \u002fSte-Catherine","de Hampton \u002f de Monkland","Rouen \u002f Fullum","de Gasp\u00e9 \u002f Dante","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Boyer \u002f St-Zotique","15e avenue \u002f Rosemont","Marquette \u002f Rachel","Bennett \u002f Ste-Catherine","de Bordeaux \u002f Marie-Anne","Viger \u002f Chenneville","Chabot \u002f de Bellechasse","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","d'Outremont \u002f Ogilvy","Bishop \u002f Ste-Catherine","Mansfield \u002f Ren\u00e9-L\u00e9vesque","St-Denis \u002f de Maisonneuve","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","Park Row O \u002f Sherbrooke","Gilford \u002f de Lanaudi\u00e8re","Guizot \u002f St-Denis","Jeanne-d'Arc \u002f Ontario","d'Orl\u00e9ans \u002f Hochelaga","Atwater \u002f Sherbrooke","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Chauveau \u002f de l'Assomption","8e avenue \u002f Saint-Joseph Lachine","Laval \u002f Duluth","Ropery \u002f Augustin-Cantin","Place d'Youville \u002f McGill","4e avenue \u002f Masson","Rachel \u002f Br\u00e9beuf","Centre \u00c9PIC (St-Zotique \u002f 40e avenue)","Jeanne-Mance \u002f Ren\u00e9-L\u00e9vesque","St-Cuthbert \u002f St-Urbain","Mozart \u002f St-Laurent","St-Dominique \u002f Rachel","Aylwin \u002f Ontario","Duluth \u002f St-Denis","de Gasp\u00e9 \u002f Marmier","Fullum \u002f St-Joseph","Plessis \u002f Ontario","de Vend\u00f4me \u002f de Maisonneuve","M\u00e9tro Villa-Maria (D\u00e9carie \u002f de Monkland)","Villeneuve \u002f St-Laurent","Lionel-Groulx \u002f George-Vanier","Waverly \u002f St-Zotique","de Grosbois \u002f Duchesneau","Chambord \u002f Laurier","March\u00e9 Atwater","Bannantyne \u002f de l'\u00c9glise","Papineau \u002f \u00c9mile-Journault","Rose-de-Lima \u002f Notre-Dame","Dollard \u002f Van Horne","de Gasp\u00e9 \u002f de Li\u00e8ge","Boyer \u002f Rosemont","Notre-Dame \u002f de la Montagne","Omer-Lavall\u00e9e \u002f Midway","Boyer \u002f Jean-Talon","Greene \u002f Workman","Notre-Dame \u002f St-Gabriel","Drolet \u002f Faillon","Villeneuve \u002f St-Urbain","Milton \u002f du Parc","Joseph-Manceau \u002f Ren\u00e9-L\u00e9vesque","Marie-Anne \u002f St-Hubert","St-Dominique \u002f St-Zotique","Milton \u002f Durocher","Ellendale \u002f C\u00f4te-des-Neiges","de Bordeaux \u002f Masson","Parc Plage","Metcalfe \u002f de Maisonneuve","St-Charles \u002f Grant","Georges-Vanier \u002f Notre-Dame","26e avenue \u002f Beaubien","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","St-Hubert \u002f Rachel","Marquette \u002f Laurier","Davaar \u002f de la C\u00f4te-Ste-Catherine","Molson \u002f Beaubien","Marquette \u002f du Mont-Royal","Dollard \u002f Van Horne","Rousselot \u002f Jarry","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","Beatty \u002f de Verdun","Fran\u00e7ois-Boivin \u002f B\u00e9langer","Joseph Manceau \u002f Ren\u00e9-L\u00e9vesque","Georges-Vanier \u002f Notre-Dame","Durocher \u002f Van Horne","de Bordeaux \u002f Jean-Talon","Terrasse Mercure \u002f Fullum","Andr\u00e9-Laurendeau \u002f Rachel","Jeanne-Mance \u002f St-Viateur","Harvard \u002f de Monkland","de Maisonneuve \u002f Aylmer","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","St-Charles \u002f Ch\u00e2teauguay","Fabre \u002f St-Zotique","Plessis \u002f Logan","Parc Chamberland (Dorais \u002f de l'\u00c9glise)","Omer-Lavall\u00e9e \u002f Midway","Jeanne Mance \u002f du Mont-Royal","Gary-Carter \u002f St-Laurent","St-Cuthbert \u002f St-Urbain","de Bordeaux \u002fGilford","Clark \u002f Rachel","Charlevoix \u002f Lionel-Groulx","de Mentana \u002f Rachel","St-Marc \u002f Sherbrooke","Bishop \u002f de Maisonneuve","16e avenue \u002f St-Joseph","Cartier \u002f St-Joseph","B\u00e9langer \u002f St-Denis","Berlioz \u002f de l'\u00cele des Soeurs","Hutchison \u002f Prince-Arthur","Victoria Hall","de la Montagne \u002f Sherbrooke","Gauthier \u002f Papineau","St-Charles \u002f Montarville","LaSalle \u002f Godin","Terminus Newman (Airlie \u002f Lafleur)","Desjardins \u002f Ontario","M\u00e9tro Viau (Pierre-de-Coubertin \u002f Viau)","Sanguinet \u002f de Maisonneuve","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","Lacombe \u002f de la C\u00f4te-des-Neiges","Graham \u002f Brookfield","Villeneuve \u002f St-Laurent","St-Patrick \u002f Briand","Jogues \u002f Allard","Bloomfield \u002f Bernard","Ste-Catherine \u002f Dezery","15e avenue \u002f Masson","Casino de Montr\u00e9al","Bishop \u002f Ste-Catherine","Jean-Brillant \u002f McKenna","Alexis-Nihon \u002f St-Louis","Louis H\u00e9mon \u002f Rosemont","Drolet \u002f Jarry","Parc Jeanne-Mance (du Mont-Royal \u002f du Parc)","Basin \u002f du S\u00e9minaire","de la Roche \u002f  de Bellechasse","Belmont \u002f Robert-Bourassa","Parc Loyola (Somerled \u002f Doherty)","de la Roche \u002f Everett","Ottawa \u002f William","Chabot \u002f de Bellechasse","Louis-H\u00e9bert \u002f Beaubien","LaSalle \u002f 80e avenue","St-Urbain \u002f de la Gaucheti\u00e8re","Berlioz \u002f de l'\u00cele des Soeurs","Parc Outremont (Bloomfield \u002f Elmwood)","M\u00e9tro Laurier (Rivard \u002f Laurier)","de Br\u00e9beuf \u002f Laurier","Fleury \u002f de St-Firmin","Duchesneau \u002f Marseille","Ste-Famille \u002f Sherbrooke","Meunier \u002f Fleury","Marmier \u002f St-Denis","de Bullion \u002f du Mont-Royal","St-Charles \u002f Charlotte","Rousselot \u002f Jarry","Calixa-Lavall\u00e9e \u002f Rachel","Stuart \u002f St-Viateur","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Grand Boulevard \u002f Sherbrooke","Sanguinet \u002f Ontario","Crescent \u002f Ren\u00e9-L\u00e9vesque","19e avenue \u002f St-Zotique","de la Commune \u002f Berri","Elsdale\u002f Louis-Hebert","10e Avenue \u002f Rosemont","Marquette \u002f Jean-Talon","Sanguinet \u002f Ontario","Lincoln \u002f du Fort","Plessis \u002f Ren\u00e9-L\u00e9vesque","Chambord \u002f B\u00e9langer","du Rosaire \u002f St-Hubert","de C\u00f4me \u002f Jean-Talon","C\u00f4te St-Antoine \u002f Royal","de la Commune \u002f Berri","Sanguinet \u002f Ste-Catherine","Bennett \u002f Ste-Catherine","Metcalfe \u002f de Maisonneuve","Ontario \u002f du Havre","Lionel-Groulx \u002f George-Vanier","St-Andr\u00e9 \u002f St-Gr\u00e9goire","de Gasp\u00e9 \u002f Jarry","Chauveau \u002f de l'Assomption","Casgrain \u002f Laurier","2e avenue \u002f Wellington","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Chomedey \u002f de Maisonneuve","du Havre \u002f de Rouen","de Bullion \u002f du Mont-Royal","La Tour de Montr\u00e9al (Pierre-de-Coubertin \u002f Bennett)","City Councillors \u002f du President-Kennedy","Montcalm \u002f de Maisonneuve","William \u002f St-Henri","Ernest-Gendreau \u002f du Mont-Royal","30e avenue \u002f St-Zotique","Parc Kent (de Kent \u002f Hudson)","1\u00e8re avenue \u002f Rosemont","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Clark \u002f Evans","des \u00c9rables \u002f Dandurand","Plessis \u002f Ontario","Plessis \u002f Ren\u00e9-L\u00e9vesque","Hutchison \u002f Van Horne","de l'Esplanade \u002f Fairmount","Mackay \u002f Sainte-Catherine","Aylmer \u002f Ste-Catherine","Wolfe \u002f Robin","M\u00e9tro Vend\u00f4me (de Marlowe \u002f de Maisonneuve)","19e avenue \u002f St-Zotique","St-Andr\u00e9 \u002f Ste-Catherine","Rose-de-Lima \u002f Workman","University \u002f des Pins","St-Charles \u002f St-Sylvestre","LaSalle \u002f S\u00e9n\u00e9cal","Ontario \u002f Sicard","St-Andr\u00e9 \u002f Ontario","Chabanel \u002f du Parc","Lincoln \u002f du Fort","Coloniale \u002f du Mont-Royal","Milton \u002f Durocher","Clark \u002f de Li\u00e8ge","City Councillors \u002f du President-Kennedy","Towers \u002f Ste-Catherine","Chomedey \u002f de Maisonneuve","Coll\u00e8ge \u00c9douard-Montpetit (de Gentilly \u002f de Normandie)","Benny \u002f Sherbrooke","Marquette \u002f des Carri\u00e8res","St-Joseph \u002f21e avenue Lachine","Ball \u002f Querbes","Prince-Arthur \u002f du Parc","Panet \u002f de Maisonneuve","St-Nicolas \u002f Place d'Youville","Hutchison \u002f Beaubien","Ste-Catherine \u002f St-Marc","Parc des Rapides (LaSalle \u002f 6e avenue)","Montcalm \u002f de Maisonneuve","Marquette \u002f du Mont-Royal","de Maisonneuve \u002f Robert-Bourassa","Resther \u002f du Mont-Royal","McGill \u002f des R\u00e9collets","Cartier \u002f Marie-Anne","Alexandra \u002f Waverly","Resther \u002f St-Joseph","Gauthier \u002f Papineau","de Ch\u00e2teaubriand \u002f Beaubien","de Normanville \u002f Beaubien","Cartier \u002f Marie-Anne","Laurier \u002f de Bordeaux","Waverly \u002f Van Horne","Wilderton  \u002f Van Horne","Ateliers municipaux de St-Laurent (Cavendish \u002f Poirier)","Dandurand \u002f de Lorimier","Henri-Julien \u002f Villeray","Jogues \u002f Allard","Rosemont \u002f Viau","Cartier \u002f Masson","Boyer \u002f Rosemont","St-Zotique \u002f 39e avenue","St-Alexandre \u002f Ste-Catherine","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f Mansfield)","5e avenue \u002f Bannantyne","M\u00e9tro Langelier (Langelier \u002f Sherbrooke)","St-Charles \u002f Sauv\u00e9","Grand Trunk \u002f Hibernia","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","Wilson \u002f Sherbrooke","Darling \u002f Sherbrooke","Henri-Julien \u002f Jean-Talon","Henri-Julien \u002f Villeray","Hochelaga \u002f Chapleau","Square St-Louis","St-Dominique \u002f Jean-Talon","de Kent \u002f de la C\u00f4te-des-Neiges","Coll\u00e8ge \u00c9douard-Montpetit","Ste-Catherine \u002f Union","St-Andr\u00e9 \u002f Robin","St-Cuthbert \u002f St-Urbain","St-Hubert \u002f Rachel","Ball \u002f Querbes","Ontario \u002f Sicard","Querbes \u002f Laurier","Drummond \u002f de Maisonneuve","de Mentana \u002f Laurier","Valois \u002f Ontario","de Mentana \u002f Gilford","Parc Van Horne (de la Peltrie \u002f de Westbury)","Island \u002f Centre","Ryde \u002f Charlevoix","Alexandra \u002f Jean-Talon","Hillside \u002f Ste-Catherine","Logan \u002f Fullum","des \u00c9rables \u002f B\u00e9langer","St-Cuthbert \u002f St-Urbain","des \u00c9rables \u002f B\u00e9langer","Notre-Dame \u002f Peel","Casino de Montr\u00e9al","Champdor\u00e9 \u002f de Lorimier","Davaar \u002f de la C\u00f4te-Ste-Catherine","5e avenue \u002f Verdun","Hudson \u002f de Kent","Drolet \u002f Beaubien","Pr\u00e9sident-Kennedy \u002f McGill College","Argyle \u002f Bannantyne","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","Union \u002f Cathcart","Le Caron \u002f Marc-Sauvalle ","Parc Jean-Drapeau (Chemin Macdonald)","de la Commune \u002f St-Sulpice","de la Gaucheti\u00e8re \u002f Mansfield","Marquette \u002f Villeray","Cypress \u002f Peel","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Georges-Vanier \u002f Notre-Dame","University \u002f des Pins","Gauvin \u002f Notre-Dame","de la Commune \u002f St-Sulpice","Wilson \u002f Sherbrooke","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Laurier \u002f Bordeaux","LaSalle \u002f S\u00e9n\u00e9cal","Notre-Dame \u002f Peel","Fullum \u002f Marie-Anne","Fortune \u002f Wellington","Biblioth\u00e8que de Rosemont (9e avenue \u002f Rosemont)","Parc J.-Arthur-Champagne (de Chambly \u002f du Mont-Royal)","M\u00e9tro Fabre (Marquette \u002f Jean-Talon)","Rachel \u002f Br\u00e9beuf","Wellington \u002f 2e avenue","University \u002f Prince-Arthur","de Lanaudi\u00e8re \u002f du Mont-Royal","Fabre \u002f St-Zotique","de la Roche \u002f St-Joseph","Wolfe \u002f Robin","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","B\u00e9langer \u002f St-Denis","Parc Langelier - Marie-Victorin","Prince-Arthur \u002f du Parc","Gary-Carter \u002f St-Laurent","8e avenue \u002f Jarry","de St-Vallier \u002f St-Zotique","St-Urbain \u002f de la Gaucheti\u00e8re","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Marie-Anne \u002f St-Hubert","des R\u00e9collets \u002f Martial","St-Dominique \u002f St-Zotique","1\u00e8re avenue \u002f Masson","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f Mansfield)","de Lanaudi\u00e8re \u002f Marie-Anne","Pierre-de-Coubertin \u002f Aird","William \u002f Robert-Bourassa","Crescent \u002f Ren\u00e9-L\u00e9vesque","Murray \u002f William","Gauthier \u002f De Lorimier","Marquette \u002f St-Gr\u00e9goire","St-Mathieu \u002f Sherbrooke","Lucien L'Allier \u002f St-Jacques","Parc Kirkland (des \u00c9rables \u002f Boyer)","Jacques-Cassault \u002f Christophe-Colomb","M\u00e9tro George-Vanier (St-Antoine \u002f Canning)","Gauthier\u002f Des \u00c9rables","\u00c9douard-Montpetit \u002f de Stirling","Roy \u002f St-Laurent","M\u00e9tro Laurier (Rivard \u002f Laurier)","Notre-Dame \u002f Gauvin","des \u00c9rables \u002f B\u00e9langer","Beaubien \u002f 8e avenue","Rachel \u002f de Br\u00e9beuf","Lajeunesse \u002f Jarry","Boyer \u002f Everett","8e avenue \u002f Beaubien","Berri \u002f Gilford","26e avenue \u002f Beaubien","Clark \u002f St-Viateur","Chabot \u002f Beaubien","Chabot \u002f de Bellechasse","Dante \u002f de Gasp\u00e9","Parc Rosemont (Dandurand \u002f d'Iberville)","Parthenais \u002f Ste-Catherine","St-Christophe \u002f Ren\u00e9-L\u00e9vesque","Ottawa \u002f St-Thomas","19e avenue \u002f St-Zotique","1\u00e8re  avenue \u002fSt-Zotique","Bennett \u002f Pierre-de-Coubertin","Marquette \u002f Rachel","de l'\u00c9glise \u002f Bannantyne","de l'\u00c9glise \u002f de Verdun","Parc de l'H\u00f4tel-de-Ville (Dub\u00e9 \u002f Laurendeau)","McTavish \u002f Sherbrooke","St-Denis \u002f Ren\u00e9-L\u00e9vesque","Mackay \u002f de Maisonneuve","de la Commune \u002f Berri","Jeanne-Mance \u002f St-Viateur","Waverly \u002f Fairmount","U. Concordia - Campus Loyola (Sherbrooke \u002f West Broadway)","du Mont-Royal \u002f Vincent-d'Indy","M\u00e9tro Viau (Pierre-de-Coubertin \u002f Viau)","Parc Ivan-Franko (33e avenue \u002f Victoria)","Larivi\u00e8re \u002f de Lorimier","Bishop \u002f Ste-Catherine","Poupart \u002f Ste-Catherine","Augustin-Cantin \u002f Shearer","Parc J.-Arthur-Champagne (Chambly \u002f du Mont-Royal)","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Alexandra \u002f Jean-Talon","Parthenais \u002f Ste-Catherine","de la Montagne \u002f Sherbrooke","d'Outremont \u002f Ogilvy","Grand Trunk \u002f Hibernia","Laporte \u002f St-Antoine","Wolfe \u002f Ren\u00e9-L\u00e9vesque","St-Hubert \u002f Ren\u00e9-L\u00e9vesque","5e avenue \u002f Bannantyne","de Vend\u00f4me \u002f de Maisonneuve","Sanguinet \u002f de Maisonneuve","Louis-H\u00e9mon \u002f Rosemont","M\u00e9tro Lionel-Groulx (Lionel-Groulx \u002f Charlevoix)","Jacques-Casault \u002f Christophe-Colomb","LaSalle \u002f Crawford","Cadillac \u002f Sherbrooke","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Faillon \u002f St-Hubert","Square Phillips","19e avenue \u002f Saint-Antoine","Queen \u002f Wellington","de Melrose \u002f Sherbrooke","Louis H\u00e9mon \u002f Rosemont","Maguire \u002f Henri-Julien","de Montmorency \u002f Richardson","Berri \u002f Rachel","Clark \u002f St-Viateur","Tupper \u002f Atwater","de Ch\u00e2teaubriand \u002f Beaubien","du Tricentenaire \u002f Prince-Albert","de Gasp\u00e9 \u002f St-Viateur","Hampton \u002f Monkland","D\u00e9z\u00e9ry \u002f Ontario","Fullum \u002f Sherbrooke","Melville \u002f de Maisonneuve","M\u00e9tro Georges-Vanier (St-Antoine \u002f Canning)","Aylmer \u002f Sherbrooke","Rivard \u002f Rachel","Parc J.-Arthur-Champagne (de Chambly \u002f du Mont-Royal)","du Mont-Royal \u002f Vincent-d'Indy","Jeanne Mance \u002f du Mont-Royal","1\u00e8re avenue \u002f Masson","Sanguinet \u002f de Maisonneuve","Boyer \u002f Jean-Talon","Clark \u002f Rachel","de la Commune \u002f McGill","Valois \u002f Ontario","Hamilton \u002f Jolicoeur","10e Avenue \u002f Rosemont","Le Caron \u002f Marc-Sauvalle ","Villeneuve \u002f du Parc","Square Phillips","Bordeaux \u002f Masson","Lapierre \u002f Serge","Logan \u002f Fullum","12e avenue \u002f St-Zotique","M\u00e9tro Lionel-Groulx (St-Jacques \u002f Atwater)","Terrasse Mercure \u002f Fullum","Ste-Famille \u002f des Pins","Place Longueuil","St-Dominique \u002f Napol\u00e9on","Fullum \u002f Sherbrooke ","de Gasp\u00e9 \u002f Fairmount","Waverly \u002f Van Horne","March\u00e9 Atwater","Boyer \u002f Beaubien","Faillon \u002f St-Denis","Plessis \u002f Ren\u00e9-L\u00e9vesque","Le Caron \u002f Marc-Sauvalle ","L\u00e9a-Roback \u002f Sir-Georges-Etienne-Cartier","Chambord \u002f B\u00e9langer","de Normanville \u002f B\u00e9langer","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","de la C\u00f4te St-Paul \u002f St-Ambroise","Frontenac \u002f du Mont-Royal","Parc Plage","Ontario \u002f Viau","O'Brien \u002f Poirier","de Kent \u002f Victoria","18e avenue \u002f Rosemont","Rouen \u002f Fullum","St-Jacques \u002f St-Laurent","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","Duluth \u002f de l'Esplanade","Par\u00e9 \u002f Mountain Sights","Villeneuve \u002f St-Urbain","Garnier \u002f du Mont-Royal","B\u00e9langer \u002f des Galeries d'Anjou","St-Dominique \u002f Jean-Talon","University \u002f Prince-Arthur","Villeneuve \u002f St-Laurent","Champagneur \u002f Jean-Talon","de la Roche \u002f  de Bellechasse","Chambord \u002f Laurier","d'Orl\u00e9ans \u002f Hochelaga","De la Commune \u002f King","Leman \u002f de Chateaubriand","Union \u002f du Pr\u00e9sident-Kennedy","de la Roche \u002f  de Bellechasse","Marquette \u002f Laurier","Fabre \u002f St-Zotique","Waverly \u002f St-Zotique","Joseph-Manseau \u002f Ren\u00e9-L\u00e9vesque","Duluth \u002f St-Denis","Hutchison \u002f Beaubien","Paul Boutet \u002f Jarry","Henri-Julien \u002f du Mont-Royal","M\u00e9tro Jolicoeur (Drake \u002f de S\u00e8ve)","Robin \u002f de la Visitation","Dandurand \u002f de Lorimier","Lajeunesse \u002f Villeray (place Tap\u00e9o)","Ann \u002f Ottawa","Victoria \u002f de Maisonneuve","Hutchison\u002f Prince-Arthur","Marquette \u002f Jean-Talon","Tolhurst \u002f Fleury","7e avenue \u002f St-Joseph","Berri \u002f St-Gr\u00e9goire","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f de la Cath\u00e9drale)","Dandurand \u002f des \u00c9rables","D\u00e9z\u00e9ry \u002f Ontario","Bourget \u002f St-Jacques","Alexandra \u002f Jean Talon","de Bordeaux \u002f Marie-Anne","de Chambly \u002f Rachel","de St-Vallier \u002f St-Zotique","Alexandra \u002f Waverly","Guizot \u002f St-Denis","Notre-Dame \u002f St-Gabriel","Faillon \u002f St-Denis","Waverly \u002f St-Viateur","Chambly \u002f Rachel","de Bullion \u002f St-Joseph","Marquette \u002f Villeray","du Parc-La Fontaine \u002f Duluth","CHSLD \u00c9loria-Lepage (de la P\u00e9pini\u00e8re \u002f de Marseille)","des Seigneurs \u002f Notre-Dame","de Lanaudi\u00e8re \u002f Marie-Anne","Duluth \u002f St-Laurent","de Normanville \u002f Jean-Talon","Bourbonni\u00e8re \u002f du Mont-Royal","Rousselot \u002f Villeray","Mansfield \u002f Ren\u00e9-L\u00e9vesque","du Mont-Royal \u002f du Parc","Clark \u002f Prince-Arthur","Chambly \u002f Rachel","Prince-Arthur \u002f du Parc","Complexe sportif Claude-Robillard","Palm \u002f St-Remi","St-Dominique \u002f Bernard","St-Viateur \u002f Casgrain","Berri \u002f Sherbrooke","Lajeunesse \u002f Villeray (place Tap\u00e9o)","Logan \u002f d'Iberville","St-Nicolas \u002f Place d'Youville","Querbes \u002f Laurier","Mackay \u002f de Maisonneuve","Henri-Julien \u002f de Castelnau","La Ronde","Dandurand \u002f Papineau","Jeanne-Mance \u002f Beaubien","Grand Trunk \u002f Hibernia","10e avenue \u002f Masson","de la Roche \u002f St-Joseph","Maguire \u002f St-Laurent","de Repentigny \u002f Sherbrooke","Parc Reine-\u00c9lizabeth (Penniston \u002f Crawford)","Place \u00c9milie-Gamelin (St-Hubert \u002f de Maisonneuve)","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Jeanne d'Arc \u002f Ontario","Quai de la navette fluviale","Mackay \u002f de Maisonneuve","du Parc-La Fontaine \u002f Roy","29e avenue \u002f St-Zotique","Rivier \u002f Davidson","Tupper \u002f Atwater","1\u00e8re avenue \u002f Masson","Molson \u002f William-Tremblay","Jean-Brillant \u002f de la C\u00f4te-des-Neiges","M\u00e9tro Bonaventure (de la Gaucheti\u00e8re \u002f de la Cath\u00e9drale)","de l'Esplanade \u002f Fairmount","Beaudry \u002f Sherbrooke","\u00c9douard-Montpetit \u002f Stirling","St-Dominique \u002f Rachel","1\u00e8re  avenue \u002fSt-Zotique","de Bordeaux \u002f Jean-Talon","Louis-H\u00e9mon \u002f Villeray","Chabot \u002f du Mont-Royal","1\u00e8re avenue \u002f Masson","Nicolet \u002f Sherbrooke","de Br\u00e9beuf \u002f Rachel","Clark \u002f Rachel","\u00c9mile-Journault \u002f de Chateaubriand","Cartier \u002f St-Joseph","Calixa-Lavall\u00e9e \u002f Rachel","Jeanne-Mance \u002f Laurier","Lusignan \u002f St-Jacques","des \u00c9rables \u002f B\u00e9langer","Fullum \u002f Ontario","Regina \u002f de Verdun","McCulloch \u002f de la C\u00f4te Ste-Catherine","Valli\u00e8res \u002f St-Laurent","Mozart \u002f St-Laurent","Hutchison \u002f Beaubien","de la Montagne \u002f Sherbrooke","Belmont \u002f du Beaver Hall","Cartier \u002f Dandurand","St-Marc \u002f Sherbrooke","28e avenue \u002f Rosemont","Joliette \u002f Ste-Catherine","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Lajeunesse \u002f Villeray","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","Poupart \u002f Ste-Catherine","Bel Air \u002f St-Antoine","Leman \u002f de Chateaubriand","3e avenue \u002f Dandurand","Le Caron \u002f Marc-Sauvalle ","de Gasp\u00e9 \u002f Dante","Tupper \u002f du Fort","Chambord \u002f Beaubien","M\u00e9tro Cadillac (Sherbrooke \u002f de Cadillac)","St-Andr\u00e9 \u002f de Maisonneuve","Chambord \u002f Fleury","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","Drummond \u002f de Maisonneuve","Champlain \u002f Ontario","Louis-H\u00e9bert \u002f St-Zotique","de l'H\u00f4tel-de-Ville \u002f Roy","Lincoln \u002f Lambert Closse","Roy \u002f St-Laurent","des \u00c9rables \u002f Rachel","Garnier \u002f St-Joseph","Hamilton \u002f Jolicoeur","10e avenue \u002f Masson","Natatorium (LaSalle \u002f Rolland)","Parc du P\u00e9lican (Laurier\u002f 2e avenue)","de Bordeaux \u002f Jean-Talon","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","du Square Ahmerst \u002f Wolfe","Crescent \u002f Ren\u00e9-L\u00e9vesque","De La Roche \u002f Everett","Drummond \u002f de Maisonneuve","Gauthier \u002f Parthenais","St-Dominique \u002f St-Zotique","Cypress \u002f Peel","Island \u002f Centre","M\u00e9tro Rosemont (de St-Vallier \u002f Rosemont)","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Parc Jeanne Mance (monument George-\u00c9tienne Cartier)","Calixa-Lavall\u00e9e \u002f Sherbrooke","Prince-Arthur \u002f du Parc","Boyer \u002f Beaubien","de Bullion \u002f St-Joseph","Laporte \u002f Guay","des \u00c9rables \u002f du Mont-Royal","Laurier \u002f 15e avenue","Drake \u002f de S\u00e8ve","de l'Esplanade \u002f Fairmount","Hamilton \u002f Jolicoeur","Island \u002f Centre","Bossuet \u002f Pierre-de-Coubertin","Plessis \u002f Ontario","de la C\u00f4te St-Antoine \u002f Royal","Laval \u002f Duluth","Louis-H\u00e9bert \u002f St-Zotique","Hutchison \u002f Beaubien","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Garnier \u002f St-Joseph","Davaar \u002f de la C\u00f4te-Ste-Catherine","Gordon \u002f Wellington","Boyer \u002f St-Zotique","de l'H\u00f4tel-de-Ville \u002f Roy","5e avenue \u002f Rosemont","Villeneuve \u002f St-Urbain","M\u00e9tro Jean-Drapeau","Ropery \u002f Augustin-Cantin","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Plessis \u002f Ontario","Omer-Lavall\u00e9e \u002f du Midway","Roy \u002f St-Denis","de Mentana \u002f Marie-Anne","Shearer \u002f Centre","Waverly \u002f St-Zotique","Ste-Famille \u002f Sherbrooke","Jardin Botanique (Pie-IX \u002f Sherbrooke)","St-Dominique \u002f Rachel","Mackay \u002fde Maisonneuve (Sud)","Parc d'Antioche","Chambord \u002f Beaubien","Jardin Botanique (Pie-IX \u002f Sherbrooke)","Rivard \u002f Rachel","Mozart \u002f Waverly","Parc St-Claude (7e avenue \u002f 8e rue)","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","de Monkland \u002f Girouard","Guizot \u002f St-Denis","Parthenais \u002f Ste-Catherine","Union \u002f President-Kennedy","Boyer \u002f Jean-Talon","St-Hubert \u002f Duluth","de Li\u00e8ge \u002f Lajeunesse","M\u00e9tro de Castelnau (de Castelnau \u002f Clark)","Quesnel \u002f Vinet","March\u00e9 Atwater","de Lanaudi\u00e8re \u002f B\u00e9langer","Chambord \u002f B\u00e9langer","Molson \u002f Beaubien","Vend\u00f4me \u002f Sherbrooke","de Li\u00e8ge \u002f Lajeunesse","Parc de Bullion (de Bullion \u002f Prince-Arthur)","Roy \u002f St-Denis","St-Andr\u00e9 \u002f de Maisonneuve","de St-Vallier \u002f Rosemont","Maguire \u002f St-Laurent","de Gasp\u00e9 \u002f Marmier","St-Jacques \u002f McGill","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Fullum \u002f Sherbrooke ","Larivi\u00e8re \u002f de Lorimier","Canning \u002f Notre-Dame","Ste-Catherine \u002f Labelle","de Champlain \u002f Ontario","Grand Boulevard \u002f Sherbrooke","St-G\u00e9rard \u002f Villeray","Ste-Famille \u002f des Pins","Louis-H\u00e9bert \u002f Beaubien","du Rosaire \u002f St-Hubert","Marquette \u002f du Mont-Royal","Ellendale \u002f C\u00f4te-des-Neiges","M\u00e9tro Rosemont (Rosemont \u002f de St-Vallier)","Biblioth\u00e8que de Rosemont (9e avenue \u002f Rosemont)","Ville-Marie \u002f Ste-Catherine","Viger \u002f Jeanne Mance","du Champ-de-Mars \u002f Gosford","Hutchison \u002f Sherbrooke","Wurtele \u002f de Rouen","St-Alexandre \u002f Ste-Catherine","Casino de Montr\u00e9al","Briand \u002f Carron","Louis-H\u00e9bert \u002f B\u00e9langer","Faillon \u002f St-Hubert","Place du Village-de-la-Pointe-aux-Trembles (St-Jean-Baptiste \u002f Notre-Dame)","Gascon \u002f de Rouen","Chomedey \u002f de Maisonneuve","Place du Commerce","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Alexandre-DeS\u00e8ve \u002f la Fontaine","Ste-Famille \u002f des Pins","Drolet \u002f St-Zotique","Marseille \u002f Viau","Logan \u002f d'Iberville","St-Hubert \u002f Duluth","Ross \u002f de l'\u00c9glise","Trans Island \u002f Queen-Mary","Faillon \u002f St-Hubert","Belmont \u002f Beaver Hall","Wellington \u002f Hickson","Chambord \u002f Beaubien","Clark \u002f Ontario","Marquette \u002f Laurier","de Monkland \u002f Girouard","Pr\u00e9sident-Kennedy \u002f McGill College","Hutchison \u002f Van Horne","du Parc-La Fontaine \u002f Duluth","de Gasp\u00e9 \u002f Beaubien","Berri \u002f Cherrier","Mackay \u002f Ren\u00e9-L\u00e9vesque","Notre-Dame \u002f de la Montagne","de Gasp\u00e9 \u002f de Li\u00e8ge","Ste-Catherine \u002f Labelle","St-Charles \u002f Montarville","de Bullion \u002f du Mont-Royal","Parc Outremont (Bloomfield \u002f Elmwood)","St-Dominique \u002f Bernard","\u00c9mile-Journault \u002f de Chateaubriand","Regina \u002f de Verdun","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Prince-Arthur \u002f du Parc","de Ch\u00e2teaubriand \u002f Beaubien","Rouen \u002f du Havre","St-Dominique \u002f Jean-Talon","de Gasp\u00e9 \u002f Marmier","Waverly \u002f St-Viateur","Pierre-de-Coubertin \u002f Aird","Marie-Anne \u002f Papineau","Gilford \u002f Br\u00e9beuf","Fortune\u002fwellington","Natatorium (LaSalle \u002f Rolland)","de Soissons \u002f de Darlington","de Lanaudi\u00e8re \u002f Marie-Anne","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Parc Sauv\u00e9 (Balzac \u002f de Bayonne)","de Clifton \u002f Sherbrooke","Guizot \u002f St-Laurent","7e avenue \u002f St-Joseph Rosemont","B\u00e9langer \u002f St-Denis","McGill \u002f des R\u00e9collets","Marcil \u002f Sherbrooke","de Bordeaux \u002f Gilford","1\u00e8re  avenue \u002fSt-Zotique","Bloomfield \u002f Bernard","Chabot \u002f Villeray","Louis-H\u00e9mon \u002f Jean-Talon","Parc Jeanne Mance (monument \u00e0 sir George-\u00c9tienne Cartier)","Fabre \u002f Beaubien","Centre Pierre-Charbonneau","St-Jacques \u002f St-Laurent","Richardson \u002f de Montmorency","de Bellechasse \u002f de St-Vallier","Bourbonni\u00e8re \u002f du Mont-Royal","Drolet \u002f Beaubien","4e avenue \u002f Masson","Fleury \u002f Andr\u00e9-Jobin","Benny \u002f Monkland","McTavish \u002f Sherbrooke","du Quesne \u002f Pierre-de-Coubertin","Ropery \u002f Grand Trunk","M\u00e9tro de Castelnau (de Castelnau \u002f St-Laurent)","Laval \u002f Rachel","Peel \u002f avenue des Canadiens de Montr\u00e9al","M\u00e9tro Jolicoeur (Drake \u002f de S\u00e8ve)","Cartier \u002f B\u00e9langer","Gauthier \u002f Papineau","Dandurand \u002f Papineau","de Maisonneuve \u002f Mansfield (nord)","Chabot \u002f du Mont-Royal","Ann \u002f Wellington","Bishop \u002f Ste-Catherine","Calixa-Lavall\u00e9e \u002f Sherbrooke","Hochelaga \u002f Chapleau","Saint-Viateur \u002f Casgrain","de Kent \u002f Victoria","Queen \u002f Wellington","St-Mathieu \u002f Ste-Catherine","St-Dominique \u002f Villeray","Richmond \u002f des Bassins","Georges-Baril \u002f Fleury","St-Jean-Baptiste \u002f Notre-Dame","de Champlain \u002f Ontario","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","du Parc-La Fontaine \u002f Roy","Wilson \u002f Sherbrooke","Rivard \u002f Rachel","des \u00c9rables \u002f Rachel","Des \u00c9rables \u002f Villeray","Roy \u002f St-Christophe","Parc Chopin (Dumouchel \u002f Tessier)","Parc de Bullion (de Bullion \u002f Prince-Arthur)","Metro Jean-Drapeau (Chemin Macdonald)","Coll\u00e8ge \u00c9douard-Montpetit (de Gentilly \u002f de Normandie)","St-Andr\u00e9 \u002f St-Antoine","Drummond \u002f Ste-Catherine","M\u00e9tro Angrignon","P\u00e9loquin \u002f Fleury","Poupart \u002f Ste-Catherine","Marquette \u002f du Mont-Royal","de Rouen \u002f Lesp\u00e9rance","Bloomfield \u002f Bernard","Godin \u002f Bannantyne","Bourbonni\u00e8re \u002f du Mont-Royal","Louis-H\u00e9bert \u002f Beaubien","du Square Ahmerst \u002f Wolfe","Dorion \u002f Ontario","9e avenue \u002f Dandurand","St-Henri \u002f Notre-Dame","Calixa-Lavall\u00e9e \u002f Sherbrooke","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Willibrord \u002f Verdun","Drolet \u002f Beaubien","Berlioz \u002f Boul de l'Ile des Soeurs","Boyer \u002f du Mont-Royal","St-Andr\u00e9 \u002f St-Antoine","Boyer \u002f Beaubien","Rousselot \u002f Jarry","Fabre \u002f St-Zotique","Boyer \u002f Beaubien","Lajeunesse \u002f Villeray (place Tap\u00e9o)","St-Dominique \u002f St-Viateur","de la Commune \u002f Place Jacques-Cartier","Wilson \u002f Sherbrooke","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Chapleau \u002f du Mont-Royal","des \u00c9rables \u002f Gauthier","de la Gaucheti\u00e8re \u002f de Bleury","Valois \u002f Ontario","de la Salle \u002f Ste-Catherine","Square St-Louis (du Square St-Louis \u002f Laval)","du Tricentenaire \u002f Prince-Albert","Parc Outremont (Bloomfield \u002f St-Viateur)","March\u00e9 Maisonneuve","Argyle \u002f Sherbrooke","Augustin-Cantin \u002f Shearer","Tupper \u002f du Fort","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","du Pr\u00e9sident-Kennedy \u002f Jeanne-Mance","Sanguinet \u002f de Maisonneuve","Biblioth\u00e8que Jean-Corbeil (Goncourt \u002f Place Montrichard)","Louis H\u00e9mon \u002f Rosemont","Lusignan \u002f St-Jacques","1\u00e8re avenue \u002f Beaubien","Waverly \u002f St-Viateur","De La Roche \u002f Everett","Tolhurst \u002f Fleury","Napol\u00e9on \u002f  St-Dominique","7e avenue \u002f St-Joseph","Ste-Famille \u002f des Pins","Metcalfe \u002f de Maisonneuve","Ottawa \u002f William","Coloniale \u002f du Mont-Royal","St-Mathieu \u002fSte-Catherine","de la Commune \u002f McGill","Fullum \u002f St-Joseph","Cartier \u002f St-Joseph","St-Andr\u00e9 \u002f Laurier","Bordeaux \u002f Masson","de la Commune \u002f Place Jacques-Cartier","Clark \u002f de Li\u00e8ge","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","de Kent \u002f de la C\u00f4te-des-Neiges","Sanguinet \u002f Ontario","Cartier \u002f B\u00e9langer","Complexe sportif Claude-Robillard","Palm \u002f St-Remi","Chabot \u002f St-Zotique","Swail \u002f Decelles","des Soeurs-Grises \u002f Marguerite-d'Youville","Tupper \u002f Atwater","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","Parthenais \u002f Ste-Catherine","Marmier \u002f St-Denis","Montcalm \u002f de Maisonneuve","Resther \u002f du Mont-Royal","St-Antoine \u002f de la Montagne","Ste-Famille \u002f des Pins","Viger\u002fJeanne Mance","Mozart \u002f St-Laurent","McGill \u002f Place d'Youville","2e avenue \u002f Wellington","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","Lambert-Closse \u002f Tupper","des Bassins \u002f du S\u00e9minaire","Marquette \u002f du Mont-Royal","Calixa-Lavall\u00e9e \u002f Rachel","Natatorium (LaSalle \u002f Rolland)","des \u00c9rables \u002f Gauthier","Notre-Dame \u002f Gauvin","Waverly \u002f St-Viateur","Parc Turin (Chambord \u002f Jean-Talon)","Bernard \u002f Jeanne-Mance","Waverly \u002f Van Horne","Trans Island \u002f Queen-Mary","McGill \u002f Place d'Youville","St-Dominique \u002f Jean-Talon","Pratt \u002f Van Horne","de Gasp\u00e9 \u002f St-Viateur","Viger\u002fJeanne Mance","St-Charles \u002f St-Sylvestre","Duvernay \u002f Charlevoix","10e avenue \u002f Masson","M\u00e9tro Monk (Allard \u002f Beaulieu)","Hillside \u002f Ste-Catherine","Roy \u002f St-Christophe","Bloomfield \u002f Bernard","Lesp\u00e9rance \u002f de Rouen","Stuart \u002f de la C\u00f4te-Ste-Catherine","de Chambly \u002f Rachel","Cadillac \u002f Sherbrooke","du Souvenir \u002f Chomedey","Henri-Julien \u002f Carmel","Garnier \u002f St-Joseph","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Marquette \u002f Jean-Talon","Aylmer \u002f Prince-Arthur","Greene \u002f St-Ambroise","Quesnel \u002f Vinet","Ryde \u002f Charlevoix","Durocher \u002f St-Viateur","Louis-H\u00e9bert \u002f St-Zotique","Grand Trunk \u002f d'Hibernia","Laporte \u002f Guay","M\u00e9tro Frontenac (Ontario \u002f du Havre)","Harvard \u002f de Monkland","St-Jacques \u002f Guy","de l'Arcade \u002f Clark","Langelier \u002f Marie-Victorin","St-Nicolas \u002f Place d'Youville","Jeanne-Mance \u002f Ren\u00e9-L\u00e9vesque","16e avenue \u002f St-Joseph","9e avenue \u002f Dandurand","Dandurand \u002f de Lorimier","Desjardins \u002f Hochelaga","Casgrain \u002f St-Viateur","M\u00e9tro Cadillac (de Cadillac \u002f Sherbrooke)","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","M\u00e9tro Villa Maria (D\u00e9carie \u002f Monkland)","de Bellechasse \u002f de St-Vallier","Hutchison\u002f Prince-Arthur","Aylmer\u002f Sainte-Catherine","Rockland \u002f Lajoie","Lionel-Groulx \u002f George-Vanier","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","de Maisonneuve \u002f Greene","St-Charles \u002f Ch\u00e2teauguay","Cartier \u002f Marie-Anne","M\u00e9tro Beaubien (de Chateaubriand \u002f Beaubien)","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Boyer \u002f Jarry","Hutchison \u002f des Pins","M\u00e9tro Frontenac (Ontario \u002f du Havre)","Guilbault \u002f Clark","Hamel \u002f Sauv\u00e9","de la Commune \u002f McGill","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","Durocher \u002f Bernard","Marcel-Laurin \u002f Beaudet","St-Germain \u002f Hochelaga","du Mont-Royal \u002f Laurendeau","CHSLD \u00c9loria-Lepage (de la P\u00e9pini\u00e8re \u002f de Marseille)","LaSalle \u002f Crawford","Leman \u002f de Chateaubriand","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Sanguinet \u002f de Maisonneuve","Clark \u002f St-Viateur","Ste-Catherine \u002f D\u00e9z\u00e9ry","M\u00e9tro Jolicoeur (de S\u00e8ve \u002f Drake)","St-Jacques \u002f Guy","Garnier \u002f du Mont-Royal","Laurier \u002f de Br\u00e9beuf","de la Gaucheti\u00e8re \u002f Robert-Bourassa","30e avenue \u002f St-Zotique","Parc Georges St-Pierre (Upper Lachine \u002f Oxford)","Br\u00e9beuf \u002f St-Gr\u00e9goire","Dunlop \u002f Van Horne","Ville-Marie \u002f Ste-Catherine","Resther \u002f St-Joseph","Brittany \u002f Ainsley","Alexandre-de-S\u00e8ve \u002f la Fontaine","Georges-Vanier \u002f Notre-Dame","Glengarry \u002f Graham","Parc J.-Arthur-Champagne (Chambly \u002f du Mont-Royal)","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","Montcalm \u002f Ontario","Garnier \u002f Everett","Island \u002f Centre","de Gasp\u00e9 \u002f Marmier","Drummond \u002f Sherbrooke","de Vend\u00f4me \u002f de Maisonneuve","Papineau \u002f \u00c9mile-Journault","Chambord \u002f Laurier","Louis-H\u00e9bert \u002f St-Zotique","Omer-Lavall\u00e9e \u002f du Midway","Larivi\u00e8re \u002f de Lorimier","Parc Rosemont (Dandurand \u002f d'Iberville)","Wolfe \u002f Robin","Boyer \u002f Beaubien","Young \u002f Wellington","de l'H\u00f4tel-de-Ville \u002f Roy","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","M\u00e9tro Snowdon (de Westbury \u002f Queen-Mary)","Swail \u002f Decelles","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Parthenais \u002f du Mont-Royal","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","de Lanaudi\u00e8re \u002f Marie-Anne","de Bordeaux \u002f Masson","M\u00e9tro Cr\u00e9mazie (Cr\u00e9mazie \u002f Lajeunesse)","de Champlain \u002f Ontario","Queen \u002f Wellington","Hutchison \u002f Van Horne","Kirkfield \u002f de Chambois","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","St-Charles \u002f Charlotte","de Bordeaux \u002f Rachel","M\u00e9tro Langelier (Sherbrooke \u002f Langelier)","de l'Esplanade \u002f Fairmount","de Bordeaux \u002f Gilford","1\u00e8re avenue \u002f Rosemont","d'Orl\u00e9ans \u002f Masson","Cartier \u002f Masson","Sanguinet \u002f Ste-Catherine","M\u00e9tro C\u00f4te-Vertu (Gohier \u002f Edouard-Laurin)","Natatorium (LaSalle \u002f Rolland)","Ville-Marie \u002f Ste-Catherine","Grand Boulevard \u002f Sherbrooke","Guizot \u002f St-Denis","Duvernay \u002f Charlevoix","M\u00e9tro Parc  (Hutchison \u002f Ogilvy)","35e avenue \u002f Beaubien","de Kent \u002f de la C\u00f4te-des-Neiges","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","Harvard \u002f Monkland","7e avenue \u002f St-Joseph","Messier \u002f Hochelaga","Shannon \u002f Ottawa","du Rosaire \u002f St-Hubert","Sainte-Catherine \u002f Union","de Ville-Marie \u002f Ste-Catherine","Chapleau \u002f Terrasse Mercure","Faillon \u002f St-Denis","Mansfield \u002f Ste-Catherine","de Lisieux \u002f Jean-Talon","Casgrain \u002f Mozart","Champlain \u002f Ontario","des Seigneurs \u002f Notre-Dame","Fullum \u002f Jean Langlois","M\u00e9tro \u00c9douard-Montpetit (du Mont-Royal \u002f Vincent-d'Indy)","Ste-Catherine \u002f St-Denis","St-Jacques \u002f St-Pierre","Laval \u002f Rachel","Beaudry \u002f Sherbrooke","Dollard \u002f Van Horne","St-Jacques \u002f St-Laurent","Henri-Julien \u002f du Carmel","de Maisonneuve \u002f Greene","du Mont-Royal \u002f Augustin-Frigon","Louis-Colin \u002f McKenna","de l'\u00c9p\u00e9e \u002f Jean-Talon","Berri \u002f Cherrier","St-Dominique \u002f Jean-Talon","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","Br\u00e9beuf \u002f St-Gr\u00e9goire","McGill \u002f Place d'Youville","Square St-Louis (du Square St-Louis \u002f Laval)","Metcalfe \u002f Square Dorchester","Durocher \u002f Van Horne","Messier \u002f St-Joseph","de la Gaucheti\u00e8re \u002f Robert-Bourassa","Laval \u002f du Mont-Royal","Boyer \u002f Rosemont","Cote St-Paul \u002f St-Ambroise","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Chabanel \u002f de l'Esplanade","St-Dominique \u002f St-Zotique","Place de la paix ( Pl. du march\u00e9 \u002f St-Dominique )","de Maisonneuve \u002f Robert-Bourassa","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Parc J.-Arthur-Champagne (de Chambly \u002f du Mont-Royal)","Querbes \u002f Ball","Square St-Louis (du Square St-Louis \u002f Laval)","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Alexandre-de-S\u00e8ve \u002f la Fontaine","Argyle \u002f Sherbrooke","Calixa-Lavall\u00e9e \u002f Rachel","Marquette \u002f Fleury","Drolet \u002f Faillon","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Union \u002f Ren\u00e9-L\u00e9vesque","Chauveau \u002f de l'Assomption","de Lanaudi\u00e8re \u002f B\u00e9langer","St-Jacques \u002f St-Pierre","Logan \u002f de Champlain","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","Cartier \u002f St-Joseph","Grand Trunk \u002f Hibernia","Bercy \u002f Ontario","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Viger \u002f Chenneville","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Leman \u002f de Chateaubriand","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","6e avenue \u002f Rosemont","St-Jacques \u002f Gauvin","Dixie \u002f 53e avenue","du Rosaire \u002f St-Hubert","Casgrain \u002f de Bellechasse","H\u00f4pital g\u00e9n\u00e9ral juif (de la C\u00f4te Ste-Catherine \u002f L\u00e9gar\u00e9)","Victoria Hall","de la Roche \u002f  de Bellechasse","Richmond \u002f Notre-Dame","Fleury \u002f Lajeunesse","de Lanaudi\u00e8re \u002f Gilford","Greene \u002f Workman","Ste-Catherine \u002f D\u00e9z\u00e9ry","M\u00e9tro \u00c9douard-Montpetit (du Mont-Royal \u002f Vincent-d'Indy)","CHSLD St-Michel (Jarry \u002f 8e avenue)","de Bordeaux \u002f St-Zotique","du Parc-La Fontaine \u002f Duluth","de Gasp\u00e9 \u002f Villeray","Ernest-Gendreau \u002f du Mont-Royal","de Maisonneuve \u002f Greene","Mackay \u002f de Maisonneuve","Marquette \u002f Laurier","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","St-Dominique \u002f Gounod","1\u00e8re avenue \u002f Beaubien","Terminus Le Carrefour (Terry-Fox \u002f Le Carrefour)","Bourget \u002f St-Jacques","des \u00c9rables \u002f du Mont-Royal","Gascon \u002f Rachel","Parc Saint-Laurent (Dutrisac)","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","Maguire \u002f St-Laurent","Boyer \u002f St-Zotique","St-Dominique \u002f Napol\u00e9on","Lincoln \u002f du Fort","Hogan \u002f Ontario","Chabot \u002f de Bellechasse","Berri \u002f St-Gr\u00e9goire","Marmier \u002f St-Denis","Stuart \u002f St-Viateur","Bernard \u002f Jeanne-Mance","William \u002f Guy","St-Denis \u002f de Maisonneuve","M\u00e9tro Laurier (Rivard \u002f Laurier)","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","St-Viateur\u002fParc","D\u00e9z\u00e9ry \u002f Ontario","Darling \u002f Sherbrooke","Henri-Julien \u002f Jarry","Regina \u002f de Verdun","de St-Vallier \u002f St-Zotique","Sherbrooke \u002f Frontenac","Chambord \u002f Laurier","Basile-Routhier \u002f Gouin","Parc de Bullion (de Bullion \u002f Prince-Arthur)","St-Jacques \u002f St-Laurent","de la Cath\u00e9drale \u002f Ren\u00e9-L\u00e9vesque","de Gasp\u00e9 \u002f Fairmount","Gordon \u002f Wellington","St-Christophe \u002f Cherrier","Plessis \u002f Ontario","Ste-Claire \u002f Pierre-T\u00e9treault","de Kent \u002f de la C\u00f4te-des-Neiges","Milton \u002f University","Ste-Famille \u002f Sherbrooke","de Chambly \u002f St-Joseph","Benny \u002f de Monkland","de Bordeaux \u002f Jean-Talon","Wilderton  \u002f Van Horne","Chapleau \u002f Terrasse Mercure","St-Nicolas \u002f Place d'Youville","H\u00f4pital Sacr\u00e9-Coeur (Forbes)","Fabre \u002f Beaubien","Dandurand \u002f Papineau","Francis \u002f Fleury","Villeneuve \u002f St-Urbain","de Gasp\u00e9 \u002f Marmier","de Maisonneuve \u002f Greene","M\u00e9tro Atwater (Atwater \u002f Ste-Catherine)","de Maisonneuve \u002f Mansfield (ouest)","Marmier \u002f St-Denis","M\u00e9tro de la Savane (de Sorel \u002f Bougainville)","de la Roche \u002f St-Joseph","du Mont-Royal \u002f Clark","1\u00e8re  avenue \u002f St-Zotique","Br\u00e9beuf \u002f Laurier","Villeray \u002f de Normanville","Louis-H\u00e9bert \u002f Belllechasse","Henri-Julien \u002f du Carmel","35e avenue \u002f Beaubien","Dandurand \u002f Papineau","Parc du Bois-Franc (de l'\u00c9quateur \u002f des Andes)","Parthenais \u002f Laurier","Chabot \u002f du Mont-Royal","Hillside \u002f Ste-Catherine","de Bordeaux \u002f Beaubien","Montcalm \u002f de Maisonneuve","Lajeunesse \u002f Villeray","Place du Commerce","Gary-Carter \u002f St-Laurent","Quai de la navette fluviale","Union \u002f Ren\u00e9-L\u00e9vesque","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","St-Andr\u00e9 \u002f Ste-Catherine","Hamel \u002f Sauv\u00e9","Chabot \u002f Everett","de Hampton \u002f de Monkland","Parc Tino-Rossi (18e avenue \u002f Everett)","Ball \u002f Querbes","McLynn \u002f Plamondon","Crescent \u002f Ren\u00e9-L\u00e9vesque","Hamel \u002f Sauv\u00e9","de Mentana \u002f Rachel","Ste-Catherine \u002f McGill College","March\u00e9 Atwater","Notre-Dame \u002f de la Montagne","Valois \u002f Ontario","Dubois \u002f Eadie","\u00c9douard-Montpetit (Universit\u00e9 de Montr\u00e9al)","Mansfield \u002f Ste-Catherine","du Mont-Royal \u002f Vincent-d'Indy","St-Jacques \u002f St-Laurent","Parc des Rapides (LaSalle \u002f 6e avenue)","Wolfe \u002f Robin","Chambord \u002f Jean-Talon","Notre-Dame \u002f St-Gabriel","Place Longueuil","de Maisonneuve \u002f Mansfield (nord)","Crescent \u002f de Maisonneuve","Ste-Famille \u002f des Pins","de Bullion \u002f St-Joseph","Clark \u002f Evans","Cartier \u002f Rosemont","CIUSSS (du Trianon \u002f Pierre-de-Coubertin)","Legrand \u002f Cartier","Fullum \u002f Sherbrooke","M\u00e9tro Laurier (Rivard \u002f Laurier)","Henri-Julien \u002f Jean-Talon","Place Longueuil","Fleury \u002f Lajeunesse","Biblioth\u00e8que de Verdun (Brown \u002f Bannantyne)","Louis-H\u00e9mon \u002f Villeray","8e avenue \u002f Beaubien","Parc Jean-Drapeau","Jardin Botanique (Pie-IX \u002f Sherbrooke)","Fabre \u002f St-Zotique","Villeneuve \u002f du Parc","Aylwin \u002f Ontario","de Maisonneuve \u002f Aylmer","McKenna \u002f \u00c9douard-Montpetit","Notre-Dame \u002f Chatham","M\u00e9tro Monk (Allard \u002f Beaulieu)","Dandurand \u002f de Lorimier","Clark \u002f Evans","Parc Jeanne Mance (monument \u00e0 sir George-\u00c9tienne Cartier)","Evans \u002f Clark","M\u00e9tro Beaubien (de Chateaubriand \u002f Beaubien)","St-Andr\u00e9 \u002f de Maisonneuve","Ellendale \u002f C\u00f4te-des-Neiges","Omer-Lavall\u00e9e \u002f du Midway","Marquette \u002f Villeray","Ste-Catherine \u002f McGill College","Palm \u002f St-R\u00e9mi","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","St-Urbain \u002f Beaubien","Rivard \u002f Rachel","Benny \u002f de Monkland","Notre-Dame \u002f de la Montagne","16e avenue \u002f St-Joseph","Chabot \u002f du Mont-Royal","d'Orl\u00e9ans \u002f Masson","Logan \u002f d'Iberville","Ellendale \u002f de la C\u00f4te-des-Neiges","Ryde \u002f Charlevoix","de la Salle \u002f Ste-Catherine","Gloria \u002f Dollard","Clark \u002f Fleury","Drake \u002f de S\u00e8ve","Maguire \u002f Henri-Julien","3e avenue \u002f Dandurand","Berri \u002f de Maisonneuve","Molson \u002f Beaubien","Drummond \u002f Ste-Catherine","du Parc-La Fontaine \u002f Roy","de Bordeaux \u002f Jean-Talon","St-Dominique \u002f Bernard","Chabot \u002f Beaubien","Thimens \u002f Alexis-Nihon","Parc Maisonneuve (Viau \u002f Sherbrooke)","Lincoln \u002f du Fort","Laval \u002f Duluth","Lucien L'Allier \u002f St-Jacques","Dandurand \u002f de Lorimier","du Parc-La Fontaine \u002f Roy","Alexandra \u002f Jean-Talon","Marseille \u002f Viau","Wolfe \u002f Robin","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","d'Outremont \u002f Ogilvy","Queen \u002f Wellington","de la Roche \u002f Everett","Tupper \u002f du Fort","Hampton \u002f Monkland","B\u00e9langer \u002f St-Denis","4e avenue \u002f de Verdun","de Bordeaux \u002f Beaubien","Prince-Arthur \u002f St-Urbain","Ontario \u002f du Havre","des \u00c9cores \u002f Jean-Talon","Guizot \u002f St-Laurent","Ernest-Gendreau \u002f du Mont-Royal","Marquette \u002f Jean-Talon","du President-Kennedy \u002f Union","Fullum \u002f Sherbrooke ","7e avenue \u002f St-Joseph Rosemont","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Pontiac \u002f Gilford","Paul Boutet \u002f des Regrattiers","Notre-Dame \u002f Li\u00e9bert","Eadie \u002f Dubois","Bellerive \u002f Duchesneau","Bishop \u002f de Maisonneuve","de Chambly \u002f Rachel","Boyer \u002f St-Zotique","Joseph-Manceau \u002f Ren\u00e9-L\u00e9vesque","Le Caron \u002f Marc-Sauvalle ","16e avenue \u002f St-Joseph","Bishop \u002f Ste-Catherine","de Grosbois \u002f Ducheneau","Marquette \u002f Villeray","McGill \u002f des R\u00e9collets","19e avenue \u002f St-Zotique","Gascon \u002f Rachel","Christophe-Colomb \u002f St-Joseph","Bennett \u002f Ste-Catherine","Louis-H\u00e9mon \u002f Villeray","Lambert-Closse \u002f Tupper","H\u00f4pital g\u00e9n\u00e9ral juif","Coll\u00e8ge \u00c9douard-Montpetit (de Gentilly \u002f de Normandie)","M\u00e9tro Cartier (des Laurentides \u002f Cartier)","Duluth \u002f St-Laurent","Victoria \u002f 18e avenue Lachine","Dorion \u002f Ontario","Chambord \u002f Jean-Talon","Legrand \u002f Laurier","Peel \u002f avenue des Canadiens de Montr\u00e9al","Mackay \u002f Ste-Catherine","Berri \u002f Sherbrooke","Jeanne-Mance \u002f St-Viateur","Louis-H\u00e9bert \u002f Beaubien","St-Charles \u002f Ch\u00e2teauguay","Victoria \u002f de Maisonneuve","Marie-Anne \u002f de la Roche","University \u002f Prince-Arthur","Prince-Arthur \u002f St-Urbain","Poupart \u002f Ontario","Letourneaux \u002f Ste-Catherine","Gilford \u002f de Br\u00e9beuf","Ren\u00e9-L\u00e9vesque \u002f 9e avenue","35e avenue \u002f Beaubien","Place de la paix ( Pl. du march\u00e9 \u002f St-Dominique )","Lajeunesse \u002f Cr\u00e9mazie","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","M\u00e9tro Laurier (Rivard \u002f Laurier)","Aylmer \u002f Prince-Arthur","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","M\u00e9tro Frontenac (Ontario \u002f du Havre)","Elgar \u002f de l'\u00cele-des-S\u0153urs","de la Grande-All\u00e9e \u002f Ste-Catherine","Notre-Dame \u002f Murray","Laval \u002f Rachel","de l'Esplanade \u002f Laurier","Basile-Routhier \u002f Chabanel","Garnier \u002f des Carri\u00e8res","Resther \u002f du Mont-Royal","Sanguinet \u002f Ste-Catherine","Gauvin \u002f Notre-Dame","Regina \u002f Verdun","de la Roche \u002f B\u00e9langer","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Ontario \u002f Sicard","Milton \u002f University","LaSalle \u002f Crawford","de Lanaudi\u00e8re \u002f Marie-Anne","des \u00c9rables \u002f du Mont-Royal","de la Commune \u002f King","Everett \u002f 8e avenue","Hillside \u002f Ste-Catherine","St-Andr\u00e9 \u002f Cherrier","d'Outremont \u002f Ogilvy","de Castelnau \u002f Lajeunesse","Chabot \u002f de Bellechasse","Terrasse Mercure \u002f Fullum","Shearer \u002f Centre","St-Charles \u002f Charlotte","Newman \u002f Senkus","Fullum \u002f St-Joseph","Drolet \u002f Beaubien","Champagneur \u002f Jean-Talon","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Jeanne-d'Arc \u002f Ontario","Poirier \u002f O'Brien","Bordeaux \u002f St-Zotique","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Desjardins \u002f Ontario","Poupart \u002f Ste-Catherine","de Bordeaux \u002f Rachel","de Maisonneuve \u002f Robert-Bourassa","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","du Pr\u00e9sident-Kennedy \u002f McGill College","Boucher \u002f St-Denis","Drummond \u002f de Maisonneuve","Place de la paix ( Pl. du march\u00e9 \u002f St-Dominique )","St-Andr\u00e9 \u002f Laurier","Marquette \u002f Rachel","Island \u002f Centre","du Rosaire \u002f St-Hubert","de la Salle \u002f Ste-Catherine","Mansfield \u002f Ste-Catherine","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","Pontiac \u002f Gilford","Rockland \u002f Lajoie","de l'Eglise \u002f St-Germain","Clark \u002f Fleury","Papineau \u002f Ren\u00e9-L\u00e9vesque","Messier \u002f du Mont-Royal","University \u002f Prince-Arthur","de la Salle \u002f Hochelaga","Pontiac \u002f Gilford","12e avenue \u002f St-Zotique","de la Commune \u002f St-Sulpice","Queen \u002f Wellington","Basile-Routhier \u002f Chabanel","de Bleury \u002f Mayor","Leman \u002f de Chateaubriand","H\u00f4pital g\u00e9n\u00e9ral juif","Valli\u00e8res \u002f St-Laurent","Rockland \u002f Lajoie","4e avenue \u002f de Verdun","Union \u002f Ren\u00e9-L\u00e9vesque","Notre-Dame \u002f Chatham","Terminus Le Carrefour (Terry-Fox \u002f Le Carr)","Calixa-Lavall\u00e9e \u002f Sherbrooke","Ontario \u002f Viau","Parc Highlands (Highlands \u002f LaSalle)","35e avenue \u002f Beaubien","1\u00e8re avenue \u002f Rosemont","des \u00c9rables \u002f B\u00e9langer","Joseph-Manseau \u002f Ren\u00e9-L\u00e9vesque","Angus (Molson \u002f William-Tremblay)","M\u00e9tro Champ-de-Mars (Viger \u002f Sanguinet)","de Maisonneuve \u002f Aylmer (est)","de la Roche \u002f du Mont-Royal","Hudson \u002f de Kent","CHU Ste-Justine (de la C\u00f4te Ste-Catherine \u002f Hudson)","Parc Labelle (Lasalle\u002f Henri Duhamel)","Ren\u00e9-L\u00e9vesque \u002f Papineau","Hillside \u002f Ste-Catherine","Bernard \u002f Clark","St-Andr\u00e9 \u002f Robin","Villeneuve \u002f du Parc","Bel Air \u002f St-Antoine","St-Mathieu \u002fSte-Catherine","McEachran \u002f Van Horne","de Mentana \u002f Laurier","Notre-Dame \u002f St-Gabriel","Park Row O \u002f Sherbrooke","Dumesnil \u002f B\u00e9langer","Dandurand \u002f de Lorimier","Quesnel \u002f Vinet","Henri-Julien \u002f Jean-Talon","Louis-H\u00e9bert \u002f de Bellechasse","Place du Commerce","Duluth \u002f St-Denis","Robert-Bourassa \u002f Ste-Catherine","Ontario \u002f Sicard","de Maisonneuve \u002f de Bleury","Clark \u002f Rachel","Alexandre-de-S\u00e8ve \u002f la Fontaine","Parc P\u00e8re-Marquette (de Lanaudi\u00e8re \u002f de Bellechasse)","de Lanaudi\u00e8re \u002f Laurier","Gauthier \u002f De Lorimier","Bourbonni\u00e8re \u002f du Mont-Royal","de la C\u00f4te St-Antoine \u002f Royal","Durocher \u002f Beaumont","Waverly \u002f St-Viateur","St-Dominique \u002f Villeray","Champagneur \u002f Jean-Talon","Dorion \u002f Ontario","de Maisonneuve \u002f Mansfield (sud)","Ontario \u002f de Lorimier","Grand Boulevard \u002f Sherbrooke","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Chambly \u002f Rachel","Gounod \u002f St-Denis","Louis-Colin \u002f McKenna","de la Commune \u002f Berri","Bercy \u002f Ontario","de St-Vallier \u002f Rosemont","Mansfield \u002f Ren\u00e9-L\u00e9vesque","Bloomfield \u002f Bernard","Bel Air \u002f St-Antoine","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","March\u00e9 Maisonneuve","Laforest \u002f Dudemaine","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","des \u00c9rables \u002f Rachel","Marie-Anne \u002f de la Roche","Papineau \u002f Ren\u00e9-L\u00e9vesque","LaSalle \u002f Godin","C\u00e9gep Andr\u00e9-Laurendeau","De la Commune \u002f King","Chabot \u002f Everett","Beatty \u002f de Verdun","Calixa-Lavall\u00e9e \u002f Rachel","Ottawa \u002f St-Thomas","10e avenue \u002f Masson","Victoria \u002f 18e avenue Lachine","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Louis-H\u00e9bert \u002f Jean-Talon","Berri \u002f de Maisonneuve","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","Lajeunesse \u002f Jean-Talon","du Champ-de-Mars \u002f Gosford","Eadie \u002f Dubois","St-Dominique \u002f Rachel","Ste-Famille \u002f des Pins","Larivi\u00e8re \u002f de Lorimier","Charlevoix \u002f Lionel-Groulx","Boyer \u002f B\u00e9langer","du Mont-Royal \u002f Clark","Hutchison \u002f Sherbrooke","Elsdale\u002f Louis-Hebert","Drolet \u002f St-Zotique","Ropery \u002f Augustin-Cantin","Chapleau \u002f du Mont-Royal","Georges-Baril \u002f Fleury","Ball \u002f Querbes","Porte-de-Qu\u00e9bec \u002f St-Hubert","Pierre-de-Coubertin \u002f Langelier","Bel Air \u002f St-Antoine","Hochelaga \u002f Chapleau","Cartier \u002f Masson","Bloomfield \u002f Van Horne","Chabot \u002f Beaubien","de Bordeaux \u002f Sherbrooke","Casgrain \u002f St-Viateur","Place \u00c9milie-Gamelin (St-Hubert \u002f de Maisonneuve)","St-Joseph \u002f 8e avenue Lachine","Square St-Louis","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","Ren\u00e9-L\u00e9vesque \u002f Papineau","de Br\u00e9beuf \u002f Laurier","Gascon \u002f Rachel","5e avenue \u002f Rosemont","de Laprairie \u002f Centre","St-Jacques \u002f des Seigneurs","St-Alexandre \u002f Ste-Catherine","Fleury \u002f Lajeunesse","Parc Rosemont (Dandurand \u002f d'Iberville)","Metcalfe \u002f Ste-Catherine","Coolbrook \u002f Queen-Mary","Stinson \u002f Montpellier","Dollard \u002f Bernard","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Coll\u00e8ge \u00c9douard-Montpetit","Union \u002f Ren\u00e9-L\u00e9vesque","Jeanne-Mance \u002f Bernard","M\u00e9tro Namur (des Jockeys \u002f D\u00e9carie)","de Br\u00e9beuf \u002f du Mont-Royal","Guy \u002f Notre-Dame","du Havre \u002f de Rouen","de Gasp\u00e9 \u002f Jarry","Fairmount \u002f St-Dominique","d'Orl\u00e9ans \u002f Masson","Cartier \u002f Rosemont","2e avenue \u002f Centrale","Dorion \u002f Ontario","Roy \u002f St-Christophe","Fullum \u002f Gilford","March\u00e9 Maisonneuve","St-Hubert \u002f Laurier","Bishop\u002f Ren\u00e9-L\u00e9vesque","de Maisonneuve \u002f Mansfield (sud)","Benny \u002f Monkland","de l'\u00c9glise \u002f Bannantyne","Lajeunesse \u002f Jarry","Waverly \u002f St-Zotique","Augustin-Cantin \u002f Shearer","Gordon \u002f Wellington","Grand Boulevard \u002f de Terrebonne","West Broadway \u002f Sherbrooke","Guizot \u002f St-Denis","Roy \u002f St-Christophe","Place du Commerce","Wilson \u002f Sherbrooke","St-Viateur\u002fParc","Willibrord \u002f Verdun","3e avenue \u002f Holt","Hutchison \u002f des Pins","Ste-Catherine \u002f Labelle","Metcalfe \u002f St-Catherine","Duluth \u002f St-Laurent","Plessis \u002f Ren\u00e9-L\u00e9vesque","Duke \u002f Brennan","Chambord \u002f Fleury","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","Lesp\u00e9rance \u002f de Rouen","de Kent \u002f Victoria","St-Mathieu \u002f Ste-Catherine","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Boyer \u002f du Mont-Royal","LaSalle \u002f Crawford","de Gasp\u00e9 \u002f Jarry","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Marmier \u002f St-Denis","D\u00e9z\u00e9ry \u002f Ontario","Mozart \u002f St-Laurent","Mansfield \u002f Sherbrooke","Cartier \u002f St-Joseph","Ste-Catherine \u002f Labelle","Parc St-Paul (Le Caron \u002f Marc-Sauvalle)","St-Maurice \u002f Robert-Bourassa","Guizot \u002f St-Laurent","Drolet \u002f Villeray","St-Andr\u00e9 \u002f Cherrier","de l'Esplanade \u002f Fairmount","de la Commune \u002f Place Jacques-Cartier","Hamel \u002f Sauv\u00e9","Bel Air \u002f St-Antoine","Nicolet \u002f Sherbrooke","M\u00e9tro Lasalle (de Rushbrooke \u002f Caisse)","Maguire \u002f Henri-Julien","Boyer \u002f Jean-Talon","Biblioth\u00e8que de Verdun (Brown \u002f Bannantyne)","Francis \u002f Fleury","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","M\u00e9tro Georges-Vanier (St-Antoine \u002f Canning)","des \u00c9rables \u002f Sherbrooke","St-Dominique \u002f Ren\u00e9-L\u00e9vesque","Marie-Anne \u002f de la Roche","de Bordeaux \u002f Sherbrooke","Faillon \u002f St-Hubert","8e avenue \u002f Notre-Dame Lachine","Alexandra \u002f Waverly","St-Dominique \u002f Rachel","Marquette \u002f St-Gr\u00e9goire","Casgrain \u002f Maguire","Henri-Julien \u002f de Castelneau","St-Jacques \u002f des Seigneurs","15e avenue \u002f Masson","Clark \u002f Prince-Arthur","Waverly \u002f St-Zotique","Parc Beaulac (rue \u00c9lizabeth)","de Mentana \u002f Marie-Anne","Basile-Routhier \u002f Gouin","Duvernay \u002f Charlevoix","Bourgeoys \u002f Favard","Louis-H\u00e9mon \u002f Villeray","Lionel-Groulx \u002f George-Vanier","Lacombe \u002f Victoria","Gascon \u002f Rachel","de la Roche \u002f Everett","Parc Ottawa (de Belleville \u002f Fleury)","de la Roche \u002f du Mont-Royal","Rachel \u002f de Br\u00e9beuf","Hickson \u002f Wellington","Basin \u002f Richmond","de Monkland \u002f Girouard","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","1\u00e8re  avenue \u002f St-Zotique","Quesnel \u002f Vinet","Egan \u002f Laurendeau","McTavish \u002f des Pins","Duluth \u002f de l'Esplanade","Ste-Catherine \u002f St-Marc","Duvernay \u002f Charlevoix","C\u00f4te St-Antoine \u002f Clarke","Sherbrooke \u002f Frontenac","M\u00e9tro Laurier (Rivard \u002f Laurier)","Darling \u002f Sherbrooke","Dunlop \u002f Van Horne","de la Gaucheti\u00e8re \u002f Mansfield","Union \u002f Ren\u00e9-L\u00e9vesque","Rushbrooke \u002f  Caisse","McTavish \u002f des Pins","Square Ahmerst \u002f Wolfe","Rockland \u002f Lajoie","St-Viateur \u002f St-Laurent","St-Dominique \u002f Ren\u00e9-L\u00e9vesque","St-Jacques \u002f McGill","Gary-Carter \u002f St-Laurent","12e avenue \u002f St-Zotique","U. Concordia - Campus Loyola (Sherbrooke \u002f West Broadway)","Milton \u002f du Parc","Hogan \u002f Ontario","Lajeunesse \u002f Jarry","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Paul Boutet \u002f des Regrattiers","Ontario \u002f Sicard","University \u002f Prince-Arthur","Berri \u002f Cherrier","de Lanaudi\u00e8re \u002f B\u00e9langer","Complexe sportif Claude-Robillard","12e avenue \u002f St-Zotique","Briand \u002f le Caron","Boyer \u002f Jarry","Parc St-Laurent (Salk \u002f Thomas-Chapais)","Roy \u002f St-Denis","Parc St-Laurent (Dutrisac)","Jogues \u002f Allard","Bernard \u002f Jeanne-Mance","St-Andr\u00e9 \u002f Robin","Peel \u002f avenue des Canadiens de Montr\u00e9al","Sanguinet \u002f de Maisonneuve","Marquette \u002f du Mont-Royal","Henri-Julien \u002f de Castelneau","Berri \u002f St-Gr\u00e9goire","M\u00e9tro du Coll\u00e8ge (Cartier \u002f D\u00e9carie)","Molson \u002f William-Tremblay","Parthenais \u002f Laurier","Augustin-Cantin \u002f Shearer","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Duluth \u002f de l'Esplanade","Prince-Arthur \u002f Ste-Famille","Place du Canada (Peel \u002f de la Gaucheti\u00e8re)","Parc Jeanne Mance (monument sir George-Etienne Cartier)","Alexandre-de-S\u00e8ve \u002f la Fontaine","Desjardins \u002f Hochelaga","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","de Bellechasse \u002f de St-Vallier","Desjardins \u002f Ontario","Drolet \u002f Laurier","Fortune\u002fwellington","Lionel-Groulx \u002f George-Vanier","Duluth \u002f de l'Esplanade","Ste-\u00c9milie \u002f Sir-Georges-Etienne-Cartier","Rockland \u002f Lajoie","Marmier \u002f St-Denis","St-Dominique \u002f Laurier","Fabre \u002f Beaubien","Cartier \u002f Gauthier","Garnier \u002f du Mont-Royal","Milton \u002f Clark","William-Tremblay \u002f Molson","des Seigneurs \u002f Notre-Dame","St-Andr\u00e9 \u002f Laurier","St-Dominique \u002f Maguire","Cartier \u002f B\u00e9langer","Guizot \u002f St-Laurent","Poupart \u002f Ste-Catherine","Lucien L'Allier \u002f St-Jacques","LaSalle \u002f 4e avenue","Legendre \u002f Henri-Julien","Milton \u002f Clark","Darling \u002f Ste-Catherine","Bercy \u002f Ontario","de Maisonneuve \u002f Stanley","Peel \u002f avenue des Canadiens de Montr\u00e9al","Ontario \u002f du Havre","Lesp\u00e9rance \u002f Rouen","Fullum \u002f St-Joseph","Terrasse Guindon \u002f Fullum","Marquette \u002f Fleury","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","Boyer \u002f Jarry","Nicolet \u002f Ste-Catherine","Girouard \u002f de Terrebonne","Waverly \u002f St-Viateur","Bishop \u002f Ste-Catherine","Boyer \u002f B\u00e9langer","M\u00e9tro Rosemont (de St-Vallier \u002f Rosemont)","M\u00e9tro Sauv\u00e9 (Berri \u002f Sauv\u00e9)","Nazareth \u002f William","Metcalfe \u002f Square Dorchester","de la C\u00f4te St-Paul \u002f St-Ambroise","Square Sir-Georges-\u00c9tienne-Cartier \u002f St-Ambroise","de Kent \u002f de la C\u00f4te-des-Neiges","des \u00c9rables \u002f B\u00e9langer","St-Andr\u00e9 \u002f Ontario","Resther \u002f du Mont-Royal","16e avenue \u002f St-Joseph","M\u00e9tro Lasalle (de Rushbrooke \u002f Caisse)","4e avenue \u002f Masson","Bourgeoys \u002f Favard","Parc Jeanne Mance (monument George-\u00c9tienne Cartier)","Hutchison \u002f Van Horne","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Quai de la navette fluviale","de l'Esplanade \u002f Laurier","Parc Henri-Julien (Legendre \u002f Henri-Julien)","Angus (Molson \u002f William-Tremblay)","Parc riverain de Lachine (des Iroquois)","Calixa-Lavall\u00e9e \u002f Rachel","Tupper \u002f Atwater","Parc Kent (de Kent \u002f Hudson)","Du Quesne \u002f Pierre-de-Coubertin","Ste-Catherine \u002f Sanguinet","Marquette \u002f Fleury","Valli\u00e8res \u002f St-Dominique","Fleury \u002f Lajeunesse","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","de l'Esplanade \u002f Fairmount","M\u00e9tro Cadillac (Sherbrooke \u002f de Cadillac)","Villeneuve \u002f St-Urbain","du Mont-Royal \u002f Augustin-Frigon","Desjardins \u002f Ontario","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","Hamilton \u002f Jolicoeur","1\u00e8re avenue \u002f Rosemont","Victoria \u002f de Maisonneuve","de Bleury \u002f de Maisonneuve","Hutchison \u002f Van Horne","Waverly \u002f Van Horne","Bernard \u002f Jeanne-Mance","Bernard \u002f Clark","16e avenue \u002f Beaubien","Logan \u002f d'Iberville","Island \u002f Centre","Parc Plage","M\u00e9tro Laurier (Berri \u002f St-Joseph)","M\u00e9tro Snowdon (de Westbury \u002f Queen-Mary)","St-Hubert \u002f Roy","Guizot \u002f St-Denis","Garnier \u002f St-Joseph","Clark \u002f Rachel","des Seigneurs \u002f Notre-Dame","Notre-Dame \u002f Chatham","Omer-Lavall\u00e9e \u002f Midway","Monkland \u002f Girouard","Gare d'autocars de Montr\u00e9al (Berri \u002f Ontario)","Chabot \u002f Everett","Mentana \u002f Marie-Anne","du Mont-Royal \u002f Clark","de Champlain \u002f Ontario","des \u00c9rables \u002f Sherbrooke","2e avenue \u002f Jean-Rivard","Boyer \u002f B\u00e9langer","Villeneuve \u002f St-Denis","Girouard \u002f de Terrebonne","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Coloniale \u002f du Mont-Royal","St-Andr\u00e9 \u002f Duluth","St-Jacques \u002f Gauvin","Parc St-Claude (7e avenue \u002f 8e rue)","du Havre \u002f Hochelaga","Parthenais \u002f St-Joseph","Gounod \u002f St-Denis","Highlands \u002f LaSalle","de Kent \u002f Victoria","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","Georges-Baril \u002f Fleury","de la Salle \u002f Ste-Catherine","Marcil \u002f Sherbrooke","Boyer \u002f Jarry","Alma \u002f Beaubien","Jean Langlois \u002f Fullum","Alma \u002f Beaubien","de la Gaucheti\u00e8re \u002f Robert-Bourassa","St-Jacques \u002f St-Pierre","Palm \u002f St-Remi","Louis-H\u00e9mon \u002f Villeray","Garnier \u002f St-Gr\u00e9goire","St-Jean-Baptiste \u002f Ren\u00e9-L\u00e9vesque","de Maisonneuve \u002f Stanley","Boyer \u002f St-Zotique","Argyle \u002f de Verdun","4e avenue \u002f Masson","de Chateaubriand \u002f Beaubien","Jeanne-Mance \u002f St-Viateur","Roy \u002f St-Denis","Ste-Famille \u002f Sherbrooke","Wilderton  \u002f Van Horne","3e avenue \u002f Dandurand","Sanguinet \u002f de Maisonneuve","Fullum \u002f Sherbrooke ","Ontario \u002f Sicard","Beurling \u002f Godin","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Logan \u002f d'Iberville","Montcalm \u002f de Maisonneuve","2e avenue \u002f Jean-Rivard","Robin \u002f de la Visitation","Hogan \u002f Ontario","Stuart \u002f de la C\u00f4te-Ste-Catherine","Boyer \u002f Beaubien","Pierre-de-Coubertin \u002f Aird","10e avenue \u002f Masson","St-Charles \u002f St-Sylvestre","Ville-Marie \u002f Ste-Catherine","Parc Jeanne Mance (monument sir George-Etienne Cartier)","du Rosaire \u002f St-Hubert","St-Catherine \u002f St-Laurent","de la Commune \u002f Berri","Robert-Bourassa \u002f de Maisonneuve","Fullum \u002f Gilford","La Ronde","de Lanaudi\u00e8re \u002f Gilford","Boyer \u002f B\u00e9langer","Francis \u002f Fleury","de Repentigny \u002f Sherbrooke","Villeneuve \u002f St-Laurent","du Pr\u00e9sident-Kennedy \u002f Jeanne-Mance","Terrasse Guindon \u002f Fullum","Marcil \u002f Sherbrooke","des Seigneurs \u002f Notre-Dame","Cartier \u002f B\u00e9langer","Notre-Dame \u002f Peel","Durocher \u002f Bernard","Coloniale \u002f du Mont-Royal","Logan \u002f Fullum","de Bordeaux \u002f Sherbrooke","Aylmer \u002f Prince-Arthur","Gounod \u002f St-Denis","Rosemont \u002f Viau","Parc Howard (de l'\u00c9p\u00e9e \u002f de Li\u00e8ge)","Sherbrooke \u002f Frontenac","de la Rotonde \u002f Jacques-Le Ber","1\u00e8re avenue \u002f Rosemont","De la Commune \u002f King","Prince-Arthur \u002f du Parc","Chabot \u002f Everett","Boyer \u002f B\u00e9langer","Atwater \u002f Greene","Parc Luigi-Pirandello (de Bretagne \u002f Jean-Rivard)","de Maisonneuve \u002f Aylmer","Drolet \u002f Gounod","de la Gaucheti\u00e8re \u002f de Bleury","Pierre-de-Coubertin \u002f Louis-Veuillot","Berlioz \u002f de l'\u00cele des Soeurs","Smith \u002f Peel","LaSalle \u002f S\u00e9n\u00e9cal","Sherbrooke \u002f Frontenac","Parc Plage","Prince-Arthur \u002f Ste-Famille","de Bordeaux \u002f Marie-Anne","Ste-\u00c9milie \u002f Sir-Georges-Etienne-Cartier","Duvernay \u002f Charlevoix","Angus (Molson \u002f William-Tremblay)","Marie-Anne \u002f de la Roche","Boyer \u002f Rosemont","M\u00e9tro C\u00f4te-des-Neiges (Jean-Brillant \u002f de la C\u00f4te-des-Neiges)","Vezina \u002f Victoria","Ann \u002f William","de Bordeaux \u002f Rachel","Stanley \u002f du Docteur-Penfield","Angus (Molson \u002f William-Tremblay)","St-Andr\u00e9 \u002f Duluth","Drake \u002f de S\u00e8ve","Ste-Catherine \u002f St-Denis","18e avenue \u002f Rosemont","Muray \u002f Notre-Dame","Mansfield \u002f Ste-Catherine","Casgrain \u002f de Bellechasse","Chabot \u002f du Mont-Royal","1\u00e8re avenue \u002f Masson","Parc Noel-Nord (Toupin \u002f Keller)","Hogan \u002f Ontario","Calixa-Lavall\u00e9e \u002f Rachel","Notre-Dame \u002f St-Gabriel","Atwater \u002f Greene","Ste-Catherine \u002f St-Laurent","18e avenue \u002f Rosemont","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","M\u00e9tro Beaubien (de Chateaubriand \u002f Beaubien)","Hogan \u002f Ontario","Wilderton  \u002f Van Horne","Richardson \u002f de Montmorency","M\u00e9tro Sauv\u00e9 (Sauv\u00e9 \u002f Berri)","Drolet \u002f Beaubien","Aylwin \u002f Ontario","de Gasp\u00e9 \u002f Jarry","Henri-Julien \u002f de Castelneau","St-Hubert \u002f Rachel","Guizot \u002f St-Denis","Square Phillips 2","M\u00e9tro Angrignon (Lamont \u002f  des Trinitaires)","Louis H\u00e9mon \u002f Rosemont","du Pr\u00e9sident-Kennedy \u002f McGill College","de Maisonneuve \u002f Greene","41e avenue \u002f Victoria","16e avenue \u002f Beaubien","Metcalfe \u002f Ste-Catherine","St-Jacques \u002f Guy","Crescent \u002f Ste-Catherine","Cypress \u002f Peel","Laporte \u002f Guay","Ontario \u002f Viau","Milton \u002f Durocher","Bernard \u002f Clark","des Ormeaux \u002f de Grosbois","Clark \u002f St-Viateur","Hamel \u002f Sauv\u00e9","Notre-Dame \u002f de la Montagne","10e avenue \u002f Masson","Sanguinet \u002f Ontario","Louis-H\u00e9bert \u002f St-Zotique","Drolet \u002f St-Zotique","Metcalfe \u002f St-Catherine","St-Dominique \u002f Jean-Talon","Casgrain \u002f Maguire","des \u00c9rables \u002f Sherbrooke","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Villeneuve \u002f du Parc","Place Rodolphe-Rousseau (Gohier \u002f \u00c9douard-Laurin)","Gary-Carter \u002f St-Laurent","Cartier \u002f Rosemont","M\u00e9tro Rosemont (Rosemont \u002f de St-Vallier)","Square Phillips","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","Hamilton \u002f Jolicoeur","du Havre \u002f de Rouen","Drolet \u002f Faillon","d'Orl\u00e9ans \u002f Hochelaga","M\u00e9tro de la Savane (de Sorel \u002f Bougainville)","Hutchison \u002f des Pins","St-Jacques \u002f St-Pierre","Ryde \u002f Charlevoix","Robin \u002f de la Visitation","M\u00e9tro Charlevoix (Centre \u002f Charlevoix)","Robin \u002f de la Visitation","St-Dominique \u002f St-Viateur","de la Commune \u002f Berri","Park Row O \u002f Sherbrooke","Parc Painter (Marcotte \u002f Quenneville)","Labadie \u002f du Parc","St-Dominique \u002f Bernard","M\u00e9tro Lionel-Groulx (Atwater \u002f Lionel-Groulx)","16e avenue \u002f Beaubien","Atwater \u002f Sherbrooke","de Bullion \u002f St-Joseph","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Wolfe \u002f Robin","Gary Carter \u002f St-Laurent","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Mansfield \u002f Sherbrooke","Chabot \u002f St-Joseph","de Montmorency \u002f Richardson","M\u00e9tro C\u00f4te-des-Neiges (Jean-Brillant \u002f de la C\u00f4te-des-Neiges)","Hutchison \u002f Sherbrooke","Villeneuve \u002f du Parc","Durocher \u002f Bernard","Robin \u002f de la Visitation","de Bullion \u002f du Mont-Royal","Rivard \u002f Rachel","Roy \u002f St-Denis","Berri \u002f St-Gr\u00e9goire","de la Salle \u002f Hochelaga","Cartier \u002f Rosemont","du Parc-La Fontaine \u002f Roy","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","St-Hubert \u002f Laurier","de Kent \u002f de la C\u00f4te-des-Neiges","Quesnel \u002f Vinet","Place Longueuil","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","Duke \u002f Brennan","Duluth \u002f St-Denis","Cathcart \u002f Union","Gagne \u002f LaSalle","de Gasp\u00e9 \u002f St-Viateur","Biblioth\u00e8que du Bois\u00e9 (Thimens \u002f Todd)","M\u00e9tro Vend\u00f4me (de Marlowe \u002f de Maisonneuve)","LaSalle \u002f Crawford","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Clark \u002f de Li\u00e8ge","Parthenais \u002f Laurier","Mansfield \u002f Sherbrooke","M\u00e9tro Lionel-Groulx (Lionel-Groulx \u002f Charlevoix)","Drolet \u002f Jarry","Park Row O \u002f Sherbrooke","Bloomfield \u002f Van Horne","de Bellechasse \u002f de St-Vallier","Fran\u00e7ois-Boivin \u002f B\u00e9langer","Square Sir-Georges-\u00c9tienne-Cartier \u002f L\u00e9a Roback","Montcalm \u002f de Maisonneuve","Ellendale \u002f de la C\u00f4te-des-Neiges","de Gasp\u00e9 \u002f Jarry","la Fontaine \u002f Alexandre-de-S\u00e8ve","Parc du P\u00e9lican (1\u00e8re avenue \u002f Masson)","Clark \u002f Rachel","Parc Plage","5e avenue \u002f Bannantyne","Hutchison \u002f Beaumont","Pierre-de-Coubertin \u002f Langelier","Marquette \u002f Gilford","10e avenue \u002f Masson","Parc Wilfrid-Bastien (Lacordaire \u002f des Galets)","Metcalfe \u002f de Maisonneuve","Parc Jean-Brillant (Swail \u002f Decelles)","M\u00e9tro Lionel-Groulx (Atwater \u002f Lionel-Groulx)","Lucien L'Allier \u002f St-Jacques","Chabot \u002f du Mont-Royal","McGill \u002f des R\u00e9collets","Marquette \u002f Laurier","Boyer \u002f Jean-Talon","Cit\u00e9 des Arts du Cirque (Paul Boutet \u002f des Regrattiers)","5e avenue \u002f Bannantyne","St-Denis \u002f de Maisonneuve","Tolhurst \u002f Fleury","10e avenue \u002f Holt","Marcil \u002f Sherbrooke","Fortune \u002f Wellington","Gilford \u002f des \u00c9rables","St-Zotique \u002f 39e avenue","Casgrain \u002f Mozart","Mansfield \u002f Ren\u00e9-L\u00e9vesque","de Bordeaux \u002f Beaubien","Rousselot \u002f Villeray","St-Viateur \u002f St-Laurent","Fabre \u002f Beaubien","Louis-H\u00e9bert \u002f B\u00e9langer","Laval \u002f du Mont-Royal","Fairmount \u002f St-Dominique","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","Joliette \u002f Ste-Catherine","Chabanel \u002f du Parc","Jean Langlois \u002f Fullum","St-Jacques \u002f St-Pierre","Coll\u00e8ge \u00c9douard-Montpetit","Le Caron \u002f Marc-Sauvalle ","de Maisonneuve \u002f Mansfield","Georges-Vanier \u002f Notre-Dame","University \u002f Prince-Arthur","St-Zotique \u002f Clark","Georges-Vanier \u002f Notre-Dame","Parc Rosemont (Dandurand \u002f d'Iberville)","de l'\u00c9p\u00e9e \u002f Jean-Talon","Ross \u002f de l'\u00c9glise","D\u00e9z\u00e9ry \u002f Ontario","Logan \u002f Fullum","de la Commune \u002f Berri","Maguire \u002f St-Laurent","Clark \u002f Rachel","de Bordeaux \u002f Sherbrooke","Waverly \u002f Van Horne","Graham \u002f Wicksteed","Roy \u002f St-Laurent","Parc Marlborough (Beauz\u00e8le \u002f Robichaud)","Marie-Anne \u002f de la Roche","Villeneuve \u002f St-Laurent","St-Dominique \u002f Gounod","de l'\u00c9glise \u002f de Verdun","St-Charles \u002f Grant","de Gasp\u00e9 \u002f Fairmount","Laurier \u002f de Bordeaux","Montcalm \u002f Ontario","St-Joseph \u002f 21e avenue Lachine","du Mont-Royal \u002f du Parc","Valli\u00e8res \u002f St-Dominique","de Lanaudi\u00e8re \u002f B\u00e9langer","Marcil \u002f Sherbrooke","Ste-Catherine \u002f St-Hubert","St-Mathieu \u002f Sherbrooke","Duvernay \u002f Charlevoix","Wolfe \u002f Robin","de Maisonneuve \u002f de Bleury","Garnier \u002f du Mont-Royal","Hutchison \u002f Prince-Arthur","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","5e avenue \u002f Bannantyne","Prince-Arthur \u002f St-Urbain","St-Andr\u00e9 \u002f Ste-Catherine","Young \u002f Wellington","St-Andr\u00e9 \u002f St-Gr\u00e9goire","H\u00f4pital Santa Cabrini (St-Zotique \u002f Jeanne-Jugan)","Marquette \u002f Jean-Talon","Li\u00e9bert \u002f Hochelaga","\u00c9douard-Montpetit \u002f de Stirling","M\u00e9tro George-Vanier (St-Antoine \u002f Canning)","Berri \u002f de Maisonneuve","Drummond \u002f Ste-Catherine","du Mont-Royal \u002f Augustin-Frigon","Poupart \u002f Ontario","M\u00e9tro Lucien-L'Allier (Lucien l'Allier \u002f Argyle)","Ontario \u002f du Havre","St-Andr\u00e9 \u002f Ontario","LaSalle \u002f 37e avenue","M\u00e9tro Place-des-Arts (de Maisonneuve \u002f de Bleury)","Elsdale\u002f Louis-Hebert","Gilford \u002f Br\u00e9beuf","de Bellechasse \u002f de St-Vallier","Bordeaux \u002fGilford","Par\u00e9 \u002f Mountain Sights","St-Denis \u002f de Maisonneuve","de Li\u00e8ge \u002f Lajeunesse","Bourgeoys \u002f Favard","Bloomfield \u002f Bernard","McGill College \u002f Ste-Catherine","Aylmer \u002f Prince-Arthur","Parc Summerlea (53e avenue \u002f St-Joseph)","Parthenais \u002f Ste-Catherine","Terrasse Mercure \u002f Fullum","Eadie\u002fDubois","Logan \u002f de Champlain","Parc Outremont (Bloomfield \u002f Elmwood)","M\u00e9tro \u00c9douard-Montpetit (du Mont-Royal \u002f Vincent-d'Indy)","du Mont-Royal \u002f Clark","Guilbault \u002f Clark","de Lanaudi\u00e8re \u002f Gilford","CHU Sainte-Justine (de la C\u00f4te Sainte-Catherine \u002f Hudson)","Laval \u002f du Mont-Royal","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Fortune \u002f Wellington","St-Charles \u002f Montarville","de Maisonneuve \u002f Stanley","6e avenue \u002f Rosemont","Hochelaga \u002f Chapleau","M\u00e9tro Monk (Allard \u002f Beaulieu)","Hudson \u002f de Kent","Mackay \u002f Ste-Catherine","Waverly \u002f Van Horne","Fullum \u002f Sherbrooke ","Mozart \u002f St-Laurent","Travaux publics LaSalle (Cordner \u002f L\u00e9ger)","Casgrain \u002f Mozart","H\u00f4pital g\u00e9n\u00e9ral juif","St-Urbain \u002f de la Gaucheti\u00e8re","de Gasp\u00e9 \u002f de Li\u00e8ge","Quai de la navette fluviale","March\u00e9 Maisonneuve","St-Maurice \u002f Robert-Bourassa","Marquette \u002f des Carri\u00e8res","de Gasp\u00e9 \u002f Dante","Rousselot \u002f Jarry","Berri \u002f Cherrier","Fortune \u002f Wellington","Ryde \u002f Charlevoix","West Broadway \u002f Sherbrooke","Queen \u002f Ottawa","Guy \u002f Notre-Dame","de Bordeaux \u002f Sherbrooke","Sanguinet \u002f Ste-Catherine","LaSalle \u002f 90e avenue","St-Ambroise \u002f Louis-Cyr","Omer-Lavall\u00e9e \u002f du Midway","de Gasp\u00e9 \u002f de Li\u00e8ge","18e avenue \u002f Rosemont","Jacques-Le Ber \u002f de la Pointe Nord","Grand Trunk \u002f Hibernia","Jardin Botanique (Pie-IX \u002f Sherbrooke)","de Maisonneuve \u002f Aylmer","Sanguinet \u002f Ste-Catherine","du Mont-Royal \u002f Clark","28e avenue \u002f Rosemont","Rouen \u002f Fullum","M\u00e9tro Radisson (Sherbrooke \u002f des Groseilliers)","Clark \u002f Guilbault","St-Nicolas \u002f Place d'Youville","Belmont \u002f Beaver Hall","Garnier \u002f St-Joseph","Bishop \u002f de Maisonneuve","Louis-H\u00e9bert \u002f B\u00e9langer","Boyer \u002f Rosemont","Ste-Catherine \u002f St-Marc","St-Mathieu \u002fSte-Catherine","Benny \u002f Sherbrooke","Faillon \u002f St-Hubert","M\u00e9tro Jean-Talon (de Chateaubriand \u002f Jean-Talon)","Valois \u002f Ontario","Queen \u002f Ottawa","Eadie \u002f Dubois","Peel \u002f de la Gauchetiere","Desjardins \u002f Ontario","Wilderton  \u002f Van Horne","8e avenue \u002f Beaubien","du Souvenir \u002f Chomedey","1\u00e8re avenue \u002f Rosemont","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Cartier \u002f Rosemont","Natatorium (LaSalle \u002f Rolland)","d'Orl\u00e9ans \u002f Masson","Maguire \u002f Henri-Julien","Nicolet \u002f Ste-Catherine","Lincoln \u002f du Fort","Clark \u002f St-Zotique","M\u00e9tro Champ-de-Mars (Viger \u002f Sanguinet)","Chambord \u002f Jean-Talon","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","Bourbonni\u00e8re \u002f du Mont-Royal","March\u00e9 Jean-Talon (Henri-Julien \u002f Jean-Talon)","St-Andr\u00e9 \u002f St-Antoine","Boyer \u002f du Mont-Royal","Bossuet \u002f Pierre-de-Coubertin","de Normanville \u002f Villeray","Metcalfe \u002f de Maisonneuve","CHSLD Benjamin-Victor-Rousselot (Dickson \u002f Sherbrooke)","de Belleville \u002f Fleury","Mozart \u002f St-Laurent","Rachel \u002f de Br\u00e9beuf","St-Dominique \u002f St-Zotique","30e avenue \u002f St-Zotique","Drolet \u002f Beaubien","de St-Vallier \u002f St-Zotique","Thimens \u002f Alexis-Nihon","de la Salle \u002f Ste-Catherine","Roy \u002f St-Denis","Messier \u002f du Mont-Royal","Monsabr\u00e9 \u002f Boileau","Napol\u00e9on \u002f  St-Dominique","Metcalfe \u002f de Maisonneuve","St-Andr\u00e9 \u002f St-Gr\u00e9goire","Young \u002f Wellington","Cartier \u002f B\u00e9langer","18e avenue \u002f Rosemont","Desjardins \u002f Hochelaga","Benny \u002f Monkland","Marquette \u002f Villeray","Jacques-Le Ber \u002f de la Pointe Nord","10e Avenue \u002f Rosemont","Casino de Montr\u00e9al","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Robin \u002f de la Visitation","Larivi\u00e8re \u002f de Lorimier","Willibrord \u002f Verdun","St-Viateur \u002f St-Laurent","Cathcart \u002f Place Phillips","Jardin Botanique (Pie-IX \u002f Sherbrooke)","M\u00e9tro Georges-Vanier (St-Antoine \u002f Canning)","Villeneuve \u002f St-Urbain","Gauthier \u002f Parthenais","Terrasse Mercure \u002f Fullum","Gilford \u002f de Lanaudi\u00e8re","Chapleau \u002f du Mont-Royal","Marquette \u002f St-Gr\u00e9goire","Parc d'Antioche (Place d'Antioche \u002f St-Zotique)","1\u00e8re avenue \u002f Masson","Square Victoria","St-Zotique \u002f Clark","Basin \u002f Richmond","M\u00e9tro Snowdon (de Westbury \u002f Queen-Mary)","Jacques-Le Ber \u002f de la Pointe Nord","M\u00e9tro Place-des-Arts (du Pr\u00e9sident-Kennedy \u002f Jeanne-Mance)","Villeneuve \u002f du Parc","Marquette \u002f Laurier","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","M\u00e9tro Atwater (Atwater \u002f Ste-Catherine)","M\u00e9tro Villa-Maria (D\u00e9carie \u002f de Monkland)","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Grand Boulevard \u002f de Terrebonne","29e avenue \u002f St-Zotique","Grand Boulevard \u002f Sherbrooke","de l'Esplanade \u002f Laurier","M\u00e9tro Sauv\u00e9 (Sauv\u00e9 \u002f Berri)","Parc des Hirondelles (Fleury \u002f Andr\u00e9-Jobin)","Argyle \u002f Bannantyne","Argyle \u002f Bannantyne","Ste-Catherine \u002f St-Hubert","Davaar \u002f de la C\u00f4te-Ste-Catherine","Beaudry \u002f Sherbrooke","Guizot \u002f St-Laurent","Milton \u002f Durocher","Valli\u00e8res \u002f St-Laurent","Darling \u002f Sherbrooke","Notre-Dame \u002f Peel","26e avenue \u002f Beaubien","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","Chomedey \u002f de Maisonneuve","Gauthier \u002f de Lorimier","M\u00e9tro Snowdon (Dornal \u002f Westbury)","Elsdale \u002f Louis-H\u00e9bert","de Kent \u002f de la C\u00f4te-des-Neiges","Pontiac \u002f Gilford","M\u00e9tro Verdun (Willibrord \u002f de Verdun)","Montcalm \u002f Ontario","de St-Vallier \u002f St-Zotique","Garnier \u002f du Mont-Royal","Ottawa \u002f William","Gauthier \u002f De Lorimier","Resther \u002f St-Joseph","Duluth \u002f St-Laurent","Hutchison \u002f Van Horne","Augustin-Cantin \u002f Shearer","Place de la paix (Pl. du march\u00e9 \u002f St-Dominique)","de Gasp\u00e9 \u002f Jean-Talon","Aylwin \u002f Ontario","M\u00e9tro Joliette  (Joliette \u002f Hochelaga)","M\u00e9tro Villa-Maria (D\u00e9carie \u002f de Monkland)","de Soissons \u002f de Darlington","Fleury \u002f Lajeunesse","Cypress \u002f Peel","Lambert-Closse \u002f Tupper","Parc St-Henri (Laporte \u002f Guay)","de l'Esplanade \u002f du Mont-Royal","Drolet \u002f St-Zotique","M\u00e9tro Laurier (Rivard \u002f Laurier)","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","de Maisonneuve \u002f Greene","du Parc-La Fontaine \u002f Duluth","de la Commune \u002f McGill","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","de St-Vallier \u002f St-Zotique","Notre-Dame \u002f Peel","de Li\u00e8ge \u002f Lajeunesse","St-Dominique \u002f Villeray","M\u00e9tro Place-des-Arts (de Maisonneuve \u002f de Bleury)","Logan \u002f d'Iberville","de la Roche \u002f du Mont-Royal","Rouen \u002f Fullum","de Chateaubriand \u002f Jarry","41e avenue \u002f Rosemont","de l'Esplanade \u002f Duluth","St-Mathieu \u002fSte-Catherine","28e avenue \u002f Rosemont","St-Andr\u00e9 \u002f Laurier","Ste-Catherine \u002f Parthenais","Parc Beaubien (Stuart \u002f St-Viateur)","d'Orl\u00e9ans \u002f Masson","Drolet \u002f St-Zotique","Peel \u002f avenue des Canadiens de Montr\u00e9al","St-Andr\u00e9 \u002f St-Antoine","Square Victoria","Louis-H\u00e9bert \u002f B\u00e9langer","Park Row O \u002f Sherbrooke","des \u00c9rables \u002f Gauthier","15e avenue \u002f Beaubien","Lajeunesse \u002f de Castelnau","Chomedey \u002f Lincoln","Milton \u002f Durocher","de Maisonneuve \u002f Greene","Harvard \u002f Monkland","de la Roche \u002f St-Joseph","8e avenue \u002f Beaubien","Parc de la Conf\u00e9d\u00e9ration (Fielding \u002f West Hill)","Beatty \u002f Verdun","D\u00e9z\u00e9ry \u002f Ontario","Chambord \u002f Laurier","Basile-Routhier \u002f Gouin","Parc Outremont (Bloomfield \u002f Elmwood)","Pontiac \u002f Gilford","Roy \u002f St-Laurent","des \u00c9rables \u002f Gauthier","M\u00e9tro Monk (Allard \u002f Beaulieu)","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Querbes \u002f Laurier","Square Victoria","Stuart \u002f de la C\u00f4te-Ste-Catherine","Valois \u002f Ontario","Letourneux \u002f Ste-Catherine","Fairmount \u002f St-Dominique","Marquette \u002f Rachel","1\u00e8re  avenue \u002fSt-Zotique","Rousselot \u002f Villeray","Rushbrooke \u002f  Caisse","Parc Grovehill (St-Antoine \u002f Ivan-Franko)","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f de la Salle)","Chambord \u002f Jean-Talon","de l'H\u00f4tel-de-Ville \u002f Roy","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Logan \u002f Fullum","Bishop \u002f Ren\u00e9-L\u00e9vesque","Lesp\u00e9rance \u002f Rouen","Gascon \u002f Rachel","Clark \u002f Fleury","Square Chaboillez (St-Jacques \u002f de la Cath\u00e9drale)","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","Labelle \u002f Ste-Catherine","Drummond \u002f Ste-Catherine","Clark \u002f Prince-Arthur","Palm \u002f St-Remi","M\u00e9tro Lasalle (de Rushbrooke \u002f Caisse)","West Broadway \u002f Sherbrooke","Gordon \u002f Wellington","Valois \u002f Ontario","Milton \u002f Clark","Fullum \u002f de Rouen","M\u00e9tro Jolicoeur (Drake \u002f de S\u00e8ve)","Chambord \u002f Beaubien","5e avenue \u002f Verdun","de la Roche \u002f  de Bellechasse","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Louis-H\u00e9bert \u002f de Bellechasse","Laval \u002f Rachel","Hutchison \u002f Fairmount","de Montmorency \u002f Richardson","Square Ahmerst \u002f Wolfe","Louis-H\u00e9bert \u002f de Bellechasse","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","Marquette \u002f du Mont-Royal","St-Jacques \u002f St-Pierre","De Lanaudi\u00e8re \u002f du Mont-Royal","St-Mathieu \u002f Sherbrooke","Fran\u00e7ois-Perrault \u002f L.-O.-David","Shearer \u002f Centre","Louis-H\u00e9bert \u002f Beaubien","Henri-Julien \u002f de Castelnau","St-Dominique \u002f Gounod","M\u00e9tro Honor\u00e9-Beaugrand (Sherbrooke \u002f Honor\u00e9-Beaugrand)","Benny \u002f Sherbrooke","St-Alexandre \u002f Ste-Catherine","7e avenue \u002f St-Joseph","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","Clark \u002f Prince-Arthur","Montcalm \u002f de Maisonneuve","M\u00e9tro Villa-Maria (de Monkland \u002f D\u00e9carie)","Roy \u002f St-Laurent","Poupart \u002f Ontario","Valli\u00e8res \u002f St-Laurent","Gilford \u002f de Br\u00e9beuf","Lambert-Closse \u002f Tupper","Bel Air \u002f St-Antoine","Stuart \u002f de la C\u00f4te-Ste-Catherine","Lincoln \u002f du Fort","15e avenue \u002f Masson","Marie-Anne \u002f St-Hubert","Parc des Rapides (LaSalle \u002f 6e avenue)","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","de Bordeaux \u002f Rachel","Messier \u002f du Mont-Royal","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Henri-Julien \u002f du Carmel","\u00c9mile-Journault \u002f de Chateaubriand","de Maisonneuve \u002f Aylmer (ouest)","Berri \u002f Rachel","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Square Victoria (Viger \u002f du Square-Victoria)","Calixa-Lavall\u00e9e \u002f Sherbrooke","Place du Commerce","de Gasp\u00e9 \u002f Dante","Towers \u002f Ste-Catherine","Coloniale \u002f du Mont-Royal","Palm \u002f St-Remi","Guizot \u002f Clark","Bernard \u002f Jeanne-Mance","28e avenue \u002f Rosemont","M\u00e9tro Vend\u00f4me","St-Joseph \u002f 34e avenue Lachine","B\u00e9langer \u002f St-Denis","de Gasp\u00e9 \u002f Dante","Duluth \u002f de l'Esplanade","Laval \u002f Rachel","1\u00e8re avenue \u002f Beaubien","St-Urbain \u002f Beaubien","Louis-H\u00e9bert \u002f B\u00e9langer","Parc Joyce (Rockland \u002f Lajoie)","10e Avenue \u002f Rosemont","Laurier \u002f de Br\u00e9beuf","4e avenue \u002f Masson","6e avenue \u002f B\u00e9langer","CIUSSS (du Trianon \u002f Pierre-de-Coubertin)","St-Marc \u002f Sherbrooke","Chabot \u002f Laurier","du President-Kennedy \u002f Robert Bourassa","de la Commune \u002f Place Jacques-Cartier","Roy \u002f St-Christophe","5e avenue \u002f Bannantyne","L\u00e9a-Roback \u002f Sir-Georges-Etienne-Cartier","1\u00e8re avenue \u002f Masson","Parc Rosemont (Dandurand \u002f d'Iberville)","Jacques-Cassault \u002f Christophe-Colomb","St-Zotique \u002f Clark","Basile-Routhier \u002f Chabanel","Ontario \u002f Viau","Plessis \u002f Ren\u00e9-L\u00e9vesque","Clark \u002f Prince-Arthur","March\u00e9 Atwater","St-Jacques \u002f St-Laurent","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Dollard \u002f Bernard","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","BAnQ (Berri \u002f de Maisonneuve)","Marie-Anne \u002f St-Hubert","St-Andr\u00e9 \u002f Cherrier","Chomedey \u002f de Maisonneuve","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Chabot \u002f Everett","Wolfe \u002f Ren\u00e9-L\u00e9vesque","13e avenue \u002f St-Zotique","de Bullion \u002f du Mont-Royal","M\u00e9tro Jarry (Lajeunesse \u002f Jarry)","Prince-Arthur \u002f St-Urbain","St-Marc \u002f Sherbrooke","Desjardins \u002f Hochelaga","5e avenue \u002f Masson","35e avenue \u002f Beaubien","Maguire \u002f St-Laurent","Square Victoria (Viger \u002f du Square-Victoria)","Coloniale \u002f du Mont-Royal","M\u00e9tro Mont-Royal (Rivard \u002f du Mont-Royal)","de Repentigny \u002f Sherbrooke","M\u00e9tro de l'\u00c9glise (Ross \u002f de l'\u00c9glise)","Casgrain \u002f St-Viateur","d'Orl\u00e9ans \u002f du Mont-Royal","de la Roche \u002f du Mont-Royal","M\u00e9tro Assomption (Chauveau \u002f de l'Assomption)","Bercy \u002f Rachel","de Montmorency \u002f Richardson","Jeanne-d'Arc \u002f Ontario","U. Concordia - Campus Loyola (Sherbrooke \u002f West Broadway)","Messier \u002f du Mont-Royal","M\u00e9tro Henri-Bourassa (Henri-Bourassa \u002f Millen)","St-Joseph \u002f 34e avenue Lachine","M\u00e9tro Villa Maria (D\u00e9carie \u002f Monkland)","Berlioz \u002f Boul de l'Ile des Soeurs","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","St-Dominique \u002f St-Zotique","Aylwin \u002f Ontario","Basile-Routhier \u002f Gouin","Guy \u002f Notre-Dame","Parc LaSalle (10e avenue \u002f Victoria)","Br\u00e9beuf \u002f Laurier","d'Oxford \u002f de Monkland","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","St-Dominique \u002f Rachel","Mansfield \u002f Ren\u00e9-L\u00e9vesque","Bloomfield \u002f Van Horne","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","de St-Vallier \u002f Rosemont","Briand \u002f Carron","de Ville-Marie \u002f Ste-Catherine","Labadie \u002f du Parc","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","d'Outremont \u002f Ogilvy","Square Phillips","Desjardins \u002f de Rouen","de St-Firmin \u002f Fleury","de la Sucrerie \u002f Centre","\u00c9mile-Duploy\u00e9 \u002f Sherbrooke","Ontario \u002f Sicard","Jacques-Le Ber \u002f de la Pointe Nord","15e avenue \u002f Masson","Logan \u002f de Champlain","de Maisonneuve \u002f Mansfield","Fairmount \u002f St-Dominique","des \u00c9rables \u002f Rachel","Boyer \u002f B\u00e9langer","St-Viateur\u002fParc","M\u00e9tro Pr\u00e9fontaine (Moreau \u002f Hochelaga)","Parc MacDonald (Clanranald \u002f Isabella)","Ste-Catherine \u002f Clark","\u00c9douard-Montpetit \u002f Stirling","Champlain \u002f Ontario","Bordeaux \u002f St-Zotique","Berri \u002f St-Gr\u00e9goire","Hutchison\u002f Prince-Arthur","Gare d'autocars de Montr\u00e9al (Berri \u002f Ontario)","de Kent \u002f Victoria","Canning \u002f Notre-Dame","M\u00e9tro de l'\u00c9glise (Ross \u002f de l'\u00c9glise)","M\u00e9tro Radisson (Sherbrooke \u002f des Groseilliers)","Drummond \u002f Ste-Catherine","Chabot \u002f du Mont-Royal","Parc Jean-Brillant (Swail \u002f Decelles)","de Br\u00e9beuf \u002f St-Gr\u00e9goire","M\u00e9tro Longueuil - Universit\u00e9 de Sherbrooke","Rousselot \u002f Villeray","de la Commune \u002f King","Jean Langlois \u002f Fullum","Quai de la navette fluviale","Roy \u002f St-Laurent","Chambord \u002f B\u00e9langer","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","du Parc-La Fontaine \u002f Duluth","Guy \u002f Notre-Dame","Laval \u002f du Mont-Royal","McGill \u002f Place d'Youville","Nicolet \u002f Ste-Catherine","de Bordeaux \u002f Sherbrooke","de la Roche \u002f du Mont-Royal","Hutchison \u002f Sherbrooke","du Mont-Royal \u002f du Parc","C\u00f4te St-Antoine \u002f Clarke","Turgeon \u002f Notre-Dame","Argyle \u002f Bannantyne","Parc Pratt (Dunlop \u002f Van Horne)","Canning \u002f Notre-Dame","Drolet \u002f Jarry","des \u00c9rables \u002f B\u00e9langer","de Bordeaux \u002f Masson","Parc Outremont (Bloomfield \u002f Elmwood)","George-Baril \u002f Sauv\u00e9","Louis-H\u00e9bert \u002f Beaubien","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","St-Christophe \u002f Cherrier","Pontiac \u002f Gilford","Metcalfe \u002f de Maisonneuve","Gary-Carter \u002f St-Laurent","de l'H\u00f4tel-de-Ville \u002f Ste-Catherine","Lajeunesse \u002f de Li\u00e8ge","Ontario \u002f Viau","St-Zotique \u002f Clark","C\u00e9gep Marie-Victorin","Metcalfe \u002f Square Dorchester","Ann \u002f William","Parc Stoney-Point (St-Joseph \u002f 47e avenue)","d'Orl\u00e9ans \u002f Masson","St-Jacques \u002f Gauvin","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","Viger \u002f Chenneville","Bernard \u002f Clark","M\u00e9tro Villa Maria (D\u00e9carie \u002f Monkland)","Ontario \u002f Viau","Lajeunesse \u002f Villeray (place Tap\u00e9o)","Mairie d'arrondissement Montr\u00e9al-Nord (H\u00e9bert \u002f de Charleroi)","8e avenue \u002f Beaubien","Basile-Routhier \u002f Chabanel","Querbes \u002f Laurier","Chabanel \u002f de l'Esplanade","du Parc-La Fontaine \u002f Duluth","McTavish \u002f Sherbrooke","Villeneuve \u002f St-Laurent","19e avenue \u002f St-Zotique","M\u00e9tro St-Laurent (de Maisonneuve \u002f St-Laurent)","St-Hubert \u002f St-Joseph","Fleury \u002f Lajeunesse","M\u00e9tro Guy-Concordia (Guy \u002f St-Catherine)","St-Charles \u002f St-Sylvestre","Mentana \u002f Marie-Anne","Square Phillips","Fortune\u002fwellington","de Belleville \u002f Fleury","Hillside \u002f Ste-Catherine","Duluth \u002f St-Denis","de Lisieux \u002f Jean-Talon","de la C\u00f4te St-Antoine \u002f Royal","de la P\u00e9pini\u00e8re \u002f Pierre-de-Coubertin","Prince-Arthur \u002f du Parc","Gauthier \u002f De Lorimier","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","d'Orl\u00e9ans \u002f Masson","Cartier \u002f B\u00e9langer","Stanley \u002f Sherbrooke","Chapleau \u002f du Mont-Royal","Marseille \u002f Viau","Gagne \u002f LaSalle","March\u00e9 Maisonneuve","Bloomfield \u002f Van Horne","Square St-Louis","Place du Commerce","Clark \u002f Prince-Arthur","Ste-Famille \u002f Sherbrooke","University \u002f des Pins","Hutchison \u002f des Pins","St-Jacques \u002f St-Pierre","Natatorium (LaSalle \u002f Rolland)","Mansfield \u002f Ste-Catherine","M\u00e9tro Montmorency (de l'Avenir \u002f Jacques-T\u00e9treault)","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","des \u00c9rables \u002f Laurier","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Union \u002f Ren\u00e9-L\u00e9vesque","Boyer \u002f Jean-Talon","Wolfe \u002f Ren\u00e9-L\u00e9vesque","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","Sicard \u002f Adam","de Mentana \u002f Gilford","Milton \u002f University","de Gasp\u00e9 \u002f de Li\u00e8ge","Robin \u002f 15e rue","Milton \u002f du Parc","Lincoln \u002f du Fort","Chambord \u002f Beaubien","26e avenue \u002f Beaubien","Valois \u002f Ste-Catherine","Hamel \u002f Sauv\u00e9","Monk \u002f de Biencourt","M\u00e9tro Champ-de-Mars (Sanguinet \u002f Viger)","Crescent \u002f de Maisonneuve","de la Commune \u002f St-Sulpice","Belmore \u002f Sherbrooke","Desjardins \u002f Hochelaga","de l\u2019h\u00f4tel de ville \u002f Saint-Joseph","Fullum \u002f Gilford","de la Commune \u002f St-Sulpice","St-Hubert \u002f Ren\u00e9-L\u00e9vesque","Poupart \u002f Ontario","Poupart \u002f Ontario","Complexe sportif Claude-Robillard (\u00c9mile-Journault)","Square St-Louis","Sanguinet \u002f Ontario","Chapleau \u002f du Mont-Royal","Crescent \u002f Ren\u00e9-L\u00e9vesque","M\u00e9tro Place-d'Armes (Viger \u002f St-Urbain)","Milton \u002f Universit\u00e9","de la Montagne \u002f Sherbrooke","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","de Gasp\u00e9 \u002f St-Viateur","Victoria \u002f de Maisonneuve","Hutchison \u002f Fairmount","Wolfe \u002f Ren\u00e9-L\u00e9vesque","Graham \u002f Kindersley","Atwater \u002f Greene","Drummond \u002f de Maisonneuve","de Bullion \u002f St-Joseph","Gatineau \u002f Swail","Roy \u002f St-Andr\u00e9","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Wurtele \u002f de Rouen","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","de la Commune \u002f Place Jacques-Cartier","de Gasp\u00e9 \u002f Marmier","Calixa-Lavall\u00e9e \u002f Sherbrooke","Wilderton  \u002f Van Horne","St-Antoine \u002f St-Fran\u00e7ois-Xavier","H\u00f4tel-de-Ville (du Champs-de-Mars \u002f Gosford)","Logan \u002f d'Iberville","5e avenue \u002f Rosemont","St-Urbain \u002f de la Gaucheti\u00e8re","La Ronde","Notre-Dame \u002f Chatham","de Gasp\u00e9 \u002f Jarry","Berri \u002f Cherrier","Louis H\u00e9mon \u002f Rosemont","Jeanne-Mance \u002f Beaubien","de Chambly \u002f Rachel","Gilford \u002f de Lanaudi\u00e8re","St-Urbain \u002f Ren\u00e9-L\u00e9vesque","Metcalfe \u002f du Square Dorchester","Rivard \u002f Rachel","Marquette \u002f Fleury","Basin \u002f Richmond","de la Roche \u002f St-Joseph","Victoria \u002f de Maisonneuve","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f Pie-IX)","St-Alexandre \u002f Ste-Catherine","des \u00c9rables \u002f Rachel","Parc Outremont (Bloomfield \u002f Elmwood)","Prince-Arthur \u002f St-Urbain","Beatty \u002f de Verdun","M\u00e9tro Beaudry (Montcalm \u002f de Maisonneuve)","de la Montagne \u002f Sherbrooke","St-Andr\u00e9 \u002f Cherrier","Place \u00c9milie-Gamelin (de Maisonneuve \u002f Berri)","de Maisonneuve \u002f de Bleury","de Vend\u00f4me \u002f de Maisonneuve","St-Joseph \u002f 8e avenue Lachine","Duke \u002f Brennan","Pontiac \u002f Gilford","Marquette \u002f Laurier","Campus MIL (Outremont \u002f Th\u00e9r\u00e8se-Lavoie Roux)","Argyle \u002f de Verdun","Louis-H\u00e9mon \u002f Villeray","Darling \u002f Sherbrooke","M\u00e9tro Jean-Drapeau","Faillon \u002f St-Denis","Alfred Lalibert\u00e9 \u002f de Poutrincourt","Chabot \u002f de Bellechasse","Berri \u002f Sherbrooke","Ste-Catherine \u002f Ste-\u00c9lisabeth","Berri \u002f Jean-Talon","Parc de Beaus\u00e9jour (Gouin \u002f Henri-d'Arles)","Laval \u002f Rachel","Waverly \u002f St-Zotique","Robert Bourassa \u002f Sainte-Catherine","St-Marc \u002f Sherbrooke","La Ronde","Alfred Lalibert\u00e9 \u002f de Poutrincourt","St-Denis \u002f de Maisonneuve","Papineau \u002f Ren\u00e9-L\u00e9vesque","28e avenue \u002f Rosemont","St-Alexandre \u002f Ste-Catherine","de la Roche \u002f Fleury","M\u00e9tro Place St-Henri (St-Ferdinand \u002f St-Jacques)","M\u00e9tro Cr\u00e9mazie (Cr\u00e9mazie \u002f Lajeunesse)","des \u00c9cores \u002f Jean-Talon","Place Jean-Paul Riopelle (Viger \u002f de Bleury)","Milton \u002f Clark","M\u00e9tro Angrignon (Lamont \u002f  des Trinitaires)","de Lanaudi\u00e8re \u002f Gilford","CCI Outremont (Dollard \u002f Durcharme)","Evans \u002f Clark","Victoria Hall","Notre-Dame-de-Gr\u00e2ce \u002f Westmount","Gare Bois-de-Boulogne","St-Dominique \u002f Napol\u00e9on","Eleanor \u002f Ottawa","Lesp\u00e9rance \u002f Rouen","M\u00e9tro Lionel-Groulx (Lionel-Groulx \u002f Charlevoix)","M\u00e9tro Lionel-Groulx (Walker \u002f Saint-Jacques)","de Maisonneuve \u002f Robert-Bourassa","Guy \u002f Notre-Dame","Casgrain \u002f de Bellechasse","M\u00e9tro Jean-Drapeau","de la Cath\u00e9drale \u002f Ren\u00e9-L\u00e9vesque","de Gasp\u00e9 \u002f de Li\u00e8ge","St-Zotique \u002f Clark","Gilford \u002f Br\u00e9beuf","St-Andr\u00e9 \u002f Ontario","19e avenue \u002f St-Zotique","M\u00e9tro Acadie (de l'Acadie \u002f Beaumont)","Cit\u00e9 des Arts du Cirque (Paul Boutet \u002f des Regrattiers)","Victoria Hall","Rockland \u002f Lajoie","Cartier \u002f Marie-Anne","de Bordeaux \u002f Jean-Talon","Rachel \u002f de Br\u00e9beuf","Quai de la navette fluviale","9e avenue \u002f Dandurand","de Maisonneuve \u002f Peel","University \u002f Milton","Hutchison \u002f des Pins","de la Gaucheti\u00e8re \u002f Robert-Bourassa","de Gasp\u00e9 \u002f Fairmount","Place Longueuil","de la Montagne \u002f Sherbrooke","Bishop \u002f Ste-Catherine","Regina \u002f de Verdun","Plessis \u002f Ontario","Parc Ludger-Duvernay (St-Andr\u00e9 \u002f St-Hubert)","Place Jean-Paul Riopelle (Viger \u002f de Bleury)","Marquette \u002f St-Gr\u00e9goire","Sherbrooke \u002f Vignal","Belmont \u002f Beaver Hall","Parc Maurice B\u00e9langer (Hebert \u002f Forest)","H\u00f4tel-de-Ville 2 (du Champs-de-Mars \u002f Gosford)","Chambord \u002f Beaubien","Messier \u002f du Mont-Royal","M\u00e9tro Monk (Allard \u002f Beaulieu)","Harvard \u002f Monkland","Bloomfield \u002f Van Horne","Hutchison \u002f des Pins","Bannantyne \u002f de l'\u00c9glise","Laurier \u002f Parisi","Garnier \u002f du Mont-Royal","Ellendale \u002f de la C\u00f4te-des-Neiges","Bordeaux \u002fGilford","18e avenue \u002f Rosemont","Ste-Catherine \u002f McGill College","Rouen \u002f Fullum","Mansfield \u002f Cathcart","\u00c9mile-Journault \u002f de Chateaubriand","Hutchison \u002f Edouard Charles","M\u00e9tro Parc  (Ogilvy \u002f Hutchison)","St-Andr\u00e9 \u002f Cherrier","Alexandre-DeS\u00e8ve \u002f la Fontaine","Hutchison \u002f Sherbrooke","de Maisonneuve\u002f Mansfield (est)","Victoria Hall","Ste-Catherine \u002f D\u00e9z\u00e9ry","Atwater \u002f Sherbrooke","Chambord \u002f Jean-Talon","Mackay \u002f Ren\u00e9-L\u00e9vesque","Laval \u002f du Mont-Royal","Villeneuve \u002f de l'H\u00f4tel-de-Ville","St-Germain \u002f Hochelaga","St-Dominique \u002f St-Viateur","Gauthier \u002f Papineau","Rivard \u002f Rachel","St-Hubert \u002f Duluth","Bloomfield \u002f Jean-Talon","Park Row O \u002f Sherbrooke","Victoria Hall","Ryde \u002f Charlevoix","de Bordeaux \u002f St-Zotique","de la Salle \u002f Ste-Catherine","Aylwin \u002f Ontario","Berri \u002f Jean-Talon","C\u00f4te St-Antoine \u002f Royal","La Ronde","St-Hubert \u002f Rachel","H\u00e9l\u00e8ne-Baillargeon \u002f St-Denis","Lionel-Groulx \u002f George-Vanier","Ste-Catherine \u002f Dezery","Towers \u002f Ste-Catherine","Jacques-Le Ber \u002f de la Pointe Nord","Sherbrooke \u002f Frontenac","Marquette \u002f Rachel","St-Charles \u002f Thomas-Keefer","Centre des loisirs (Tass\u00e9 \u002f Grenet)","4e avenue \u002f Masson","St-Charles \u002f Montarville","Nicolet \u002f Sherbrooke","Chauveau \u002f de l'Assomption","Francis \u002f Fleury","M\u00e9tro Papineau (Cartier \u002f Ste-Catherine)","Prince-Arthur \u002f Ste-Famille","Quesnel \u002f Vinet","des Seigneurs \u002f Notre-Dame","Sainte-Catherine \u002f Parthenais","St-Andr\u00e9 \u002f Laurier","Graham \u002f Brookfield","Crescent \u002f de Maisonneuve","Berri \u002f Duluth","Laurier \u002f 15e avenue","St-Alexandre \u002f Ste-Catherine","Marie-Anne \u002f de la Roche","Greene \u002f St-Ambroise","De la Commune \u002f King","M\u00e9tro du Coll\u00e8ge (Cartier \u002f D\u00e9carie)","de Mentana \u002f Laurier","St-Charles \u002f Grant","Lajeunesse \u002f de Castelnau","Metcalfe \u002f du Square-Dorchester","Louis-H\u00e9mon \u002f Rosemont","Pierre-de-Coubertin \u002f Aird","Jacques-Casault \u002f Christophe-Colomb","30e avenue \u002f St-Zotique","de Maisonneuve \u002f Aylmer (est)","Hutchison \u002f Sherbrooke","Jacques-Casault \u002f Christophe-Colomb","d'Orl\u00e9ans \u002f Hochelaga","de Lanaudi\u00e8re \u002f Marie-Anne","26e avenue \u002f Beaubien","Hudson \u002f de Kent","St-Jacques \u002f St-Laurent","Sanguinet \u002f Sainte-Catherine","Parc J.-Arthur-Champagne (Chambly \u002f du Mont-Royal)","University \u002f des Pins","Hamel \u002f Sauv\u00e9","\u00c9douard-Montpetit \u002f de Stirling","Hutchison \u002f des Pins","Union \u002f Ren\u00e9-L\u00e9vesque","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","16e avenue \u002f Beaubien","Parc Maurice B\u00e9langer (Forest \u002f de l'H\u00f4tel-de-Ville)","Boyer \u002f B\u00e9langer","de Bordeaux \u002f Marie-Anne","19e avenue \u002f St-Joseph Lachine","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","Plessis \u002f Ontario","Victoria Hall","Panet \u002f Ste-Catherine","Ernest-Gendreau \u002f du Mont-Royal","de Kent \u002f Victoria","Jean-Brillant \u002f McKenna","Pierre-de-Coubertin \u002f Louis-Veuillot","Laurier \u002f 15e avenue","de Maisonneuve \u002f Robert-Bourassa","Square Phillips","Turgeon \u002f Notre-Dame","Marlowe \u002f de Maisonneuve","M\u00e9tro Plamondon (de Kent \u002f Victoria)","Joseph Manceau \u002f Ren\u00e9-L\u00e9vesque","Berri \u002f St-Joseph","M\u00e9tro Honor\u00e9-Beaugrand (Sherbrooke \u002f Honor\u00e9-Beaugrand)","de Darlington \u002f de la C\u00f4te-Ste-Catherine","Drolet \u002f Laurier","Milton \u002f Durocher","Parthenais \u002f Ste-Catherine","de Lanaudi\u00e8re \u002f Marie-Anne","de Melrose \u002f Sherbrooke","Cartier \u002f Rosemont","St-Dominique \u002f St-Viateur","Villeneuve \u002f St-Urbain","University \u002f Prince-Arthur","de l'Esplanade \u002f du Mont-Royal","Ren\u00e9-L\u00e9vesque \u002f Papineau","Marie-Anne \u002f St-Hubert","Georges-Vanier \u002f Notre-Dame","Marie-Anne \u002f St-Hubert","M\u00e9tro Lucien-L'Allier  (Lucien l'Allier \u002f Argyle)","LaSalle \u002f S\u00e9n\u00e9cal","City Councillors \u002f du President-Kennedy","Fullum \u002f St-Joseph","3e avenue \u002f Dandurand","St-Andr\u00e9 \u002f St-Gr\u00e9goire","Casino de Montr\u00e9al","CHSLD St-Michel (Jarry \u002f 8e avenue)","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Mackay \u002fde Maisonneuve (Sud)","Messier \u002f du Mont-Royal","M\u00e9tro George-Vanier (St-Antoine \u002f Canning)","de Vend\u00f4me \u002f de Maisonneuve","M\u00e9tro Atwater (Atwater \u002f Ste-Catherine)","de Li\u00e8ge \u002f Lajeunesse","16e avenue \u002f St-Joseph","Villeray \u002f St-Michel","de Bordeaux \u002f Masson","Notre-Dame \u002f St-Gabriel","Kirkfield \u002f de Chambois","de la Commune \u002f King","12e avenue \u002f St-Zotique","Villeneuve \u002f St-Laurent","St-Jacques \u002f Gauvin","St-Dominique \u002f du Mont-Royal","Drummond \u002f Ste-Catherine","Boyer \u002f Jean-Talon","Bordeaux \u002f St-Zotique","Hutchison \u002f Beaubien","Hamilton \u002f Jolicoeur","LaSalle \u002f Crawford","Faillon \u002f St-Hubert","Augustin-Cantin \u002f Shearer","Gilford \u002f de Br\u00e9beuf","Ste-Croix \u002f Tass\u00e9","Chambord \u002f Beaubien","Mansfield \u002f Sherbrooke","de Bordeaux \u002f Gilford","de Bellechasse \u002f de St-Vallier","Laval \u002f Duluth","de l'H\u00f4tel-de-Ville \u002f St-Joseph","St-Andr\u00e9 \u002f Ontario","de Maisonneuve \u002f Aylmer","St-Antoine \u002f St-Fran\u00e7ois-Xavier","Notre-Dame \u002f Li\u00e9bert","Queen \u002f Ottawa","de la Commune \u002f King","de l'H\u00f4tel-de-Ville \u002f Roy","16e avenue \u002f Beaubien","St-Charles \u002f St-Jean","Notre-Dame \u002f de la Montagne","8e avenue \u002f Beaubien","Trans Island \u002f Queen-Mary","Stuart \u002f de la C\u00f4te-Ste-Catherine","Fairmount \u002f St-Dominique","Parc Dixie (54e avenue \u002f Dixie)","Beaudry \u002f Ontario","Louis-H\u00e9bert \u002f B\u00e9langer","de l'\u00c9glise \u002f de Verdun","Chomedey \u002f de Maisonneuve","Gauthier \u002f Papineau","Duluth \u002f St-Denis","16e avenue \u002f Jean-Talon","Poupart \u002f Ontario","du Mont-Royal \u002f du Parc","Chapleau \u002f du Mont-Royal","St-Zotique \u002f Clark","du Mont-Royal \u002f Vincent-d'Indy","des R\u00e9collets \u002f Martial","Milton \u002f du Parc","Bernard \u002f Jeanne-Mance","Poupart \u002f Ste-Catherine","Beaucourt \u002f de la C\u00f4te-Ste-Catherine","de la Gaucheti\u00e8re \u002f Robert-Bourassa","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","Parc Plage","Waverly \u002f St-Zotique","de Bullion \u002f St-Joseph","Tellier \u002f des Futailles","La Ronde","Fran\u00e7ois-Perrault \u002f L.-O.-David","Lajeunesse \u002f Jarry","Marquette \u002f St-Gr\u00e9goire","3e avenue \u002f Dandurand","Lajeunesse \u002f Jarry","Berri \u002f Gilford","de Hampton \u002f de Monkland","Atateken \u002f du Square Amherst","Davaar \u002f de la C\u00f4te-Ste-Catherine","St-Andr\u00e9 \u002f Cherrier","de Bordeaux \u002f Rachel","Casgrain \u002f Mozart","Guizot \u002f St-Laurent","M\u00e9tro Assomption (Chauveau \u002f de l'Assomption)","Francis \u002f Fleury","des Seigneurs \u002f Workman","City Councillors \u002f du President-Kennedy","Ste-Catherine \u002f St-Denis","Francis \u002f Fleury","St-Andr\u00e9 \u002f Duluth","St-Marc \u002f Sherbrooke","de la C\u00f4te St-Antoine \u002f Argyle","C\u00f4te St-Antoine \u002f Clarke","Biblioth\u00e8que de Verdun (Brown \u002f Bannantyne)","Grand Boulevard \u002f Sherbrooke","Resther \u002f St-Joseph","M\u00e9tro Sherbrooke (de Rigaud \u002f St-Denis)","Fabre \u002f St-Zotique","Drummond \u002f de Maisonneuve","Mackay \u002f Ren\u00e9-L\u00e9vesque","St-Andr\u00e9 \u002f Duluth","Champagneur \u002f Jean-Talon","Park Row O \u002f Sherbrooke","Laval \u002f Rachel","Resther \u002f St-Joseph","Jogues \u002f Allard","Cypress \u002f Peel","Basile-Routhier \u002f Chabanel","Marmier","\u00c9douard-Montpetit \u002f Stirling","du President-Kennedy \u002f Union","Monkland \u002f Girouard","M\u00e9tro Sauv\u00e9 (Berri \u002f Sauv\u00e9)","Lucien L'Allier \u002f St-Jacques","La Ronde","27e avenue \u002f Rosemont","M\u00e9tro Angrignon","Molson \u002f Beaubien","Stanley \u002f Sherbrooke","Calixa-Lavall\u00e9e \u002f Sherbrooke","Davaar \u002f de la C\u00f4te-Ste-Catherine","Casgrain \u002f de Bellechasse","Beaudry \u002f Sherbrooke","Biblioth\u00e8que d'Ahuntsic (Lajeunesse \u002f Fleury)","Parc H\u00e9bert (Provencher \u002f Buies)","de la Roche \u002f Everett","5e avenue \u002f Verdun","M\u00e9tro Henri-Bourassa (Henri-Bourassa \u002f Millen)","du Mont-Royal \u002f Clark","Milton \u002f Clark","de Chateaubriand \u002f Jarry","Marquette \u002f Rachel","H\u00f4pital Maisonneuve-Rosemont (Rosemont \u002f Chatelain)","de la Commune \u002f Place Jacques-Cartier","9e avenue \u002f Dandurand","Louis-H\u00e9bert \u002f B\u00e9langer","M\u00e9tro Universit\u00e9 de Montr\u00e9al (\u00c9douard-Montpetit \u002f Louis-Collin)","Fleury \u002f Lajeunesse","Parc P\u00e8re-Marquette (Chambord \u002f Rosemont)","de L\u00e9ry \u002f Sherbrooke","de l'H\u00f4tel-de-Ville \u002f Sherbrooke","St-Dominique \u002f St-Zotique","de l'H\u00f4tel-de-Ville \u002f Roy","Sanguinet \u002f Ontario","Ste-Famille \u002f Sherbrooke","M\u00e9tro Peel (de Maisonneuve \u002f Stanley)","de la Roche \u002f St-Joseph","Grey \u002f Sherbrooke","Notre-Dame-de-Gr\u00e2ce \u002f D\u00e9carie","Villeneuve \u002f de l'H\u00f4tel-de-Ville","Waverly \u002f Van Horne","President-Kennedy \u002f Robert Bourassa","Br\u00e9beuf \u002f St-Gr\u00e9goire","BAnQ (Berri \u002f de Maisonneuve)","Lionel-Groulx \u002f George-Vanier","Fran\u00e7ois-Perrault \u002f L.-O.-David","Briand \u002f Carron","Berri \u002f Cherrier","Garnier \u002f St-Joseph","Parc du P\u00e9lican (2e avenue \u002f St-Joseph)","Beaudet \u002f Marcel-Laurin","de St-Vallier \u002f St-Zotique","St-Cuthbert \u002f St-Urbain","de Bullion \u002f du Mont-Royal","Cartier \u002f Dandurand","M\u00e9tro Montmorency","de Bullion \u002f St-Joseph","M\u00e9tro C\u00f4te-des-Neiges (Lacombe \u002f de la C\u00f4te-des-Neiges)","Bernard \u002f Jeanne-Mance","du Tricentenaire \u002f de Montigny","M\u00e9tro Sauv\u00e9 (Sauv\u00e9 \u002f Berri)","Dumesnil \u002f B\u00e9langer","Napol\u00e9on \u002f  St-Dominique","des \u00c9cores \u002f Jean-Talon","Metcalfe \u002f Ste-Catherine","Queen \u002f Wellington","Gare Canora","Aylwin \u002f Ontario","Milton \u002f du Parc","des Jockeys \u002f D\u00e9carie","Sherbrooke \u002f Frontenac","Parc Champdor\u00e9 (Champdor\u00e9 \u002f de Lille)","M\u00e9tro Outremont (Wiseman \u002f Van Horne)","Parc Benny (Benny \u002f de Monkland)","Maguire \u002f St-Laurent","Parc Kent (de Kent \u002f Hudson)","du Mont-Royal \u002f Augustin-Frigon","de Mentana \u002f Laurier","Notre-Dame \u002f St-Gabriel","Tupper \u002f du Fort","Mentana \u002f Marie-Anne","Roy \u002f St-Denis","Casino de Montr\u00e9al","Square Sir-Georges-Etienne-Cartier \u002f Ste-\u00c9milie","McGill \u002f Place d'Youville","M\u00e9tro Pie-IX (Pierre-de-Coubertin \u002f De La Salle)","Bourgeoys \u002f Favard","Belmont \u002f du Beaver Hall","McKenna \u002f \u00c9douard-Montpetit","Cote St-Paul \u002f St-Ambroise","Louis-H\u00e9bert \u002f St-Zotique","Terrasse Guindon \u002f Fullum","M\u00e9tro St-Michel (Shaughnessy \u002f St-Michel)","Sanguinet \u002f Ste-Catherine","Boyer \u002f Rosemont","Gounod \u002f St-Denis","Berri \u002f Rachel","de la Cath\u00e9drale \u002f Ren\u00e9-L\u00e9vesque","Bloomfield \u002f Bernard","William \u002f Robert-Bourassa","St-Nicolas \u002f Place d'Youville","Marcil \u002f Sherbrooke","Alexandre-DeS\u00e8ve \u002f de Maisonneuve","Parc Henri-Durant (L\u00e9ger \u002f Henri-Durant)","M\u00e9tro Monk (Allard \u002f Beaulieu)","Place du Canada ( Peel \u002f de la Gaucheti\u00e8re)","Resther \u002f St-Joseph","Charlevoix \u002f Lionel-Groulx","Parc nature de l'\u00cele-de-la-Visitation (Gouin \u002f de Bruch\u00e9si)","Casino de Montr\u00e9al","Notre-Dame \u002f de la Montagne","6e avenue \u002f Rosemont","Drolet \u002f Marie-Anne","de Gasp\u00e9 \u002f Fairmount","Lajeunesse \u002f Jean-Talon","St-Urbain \u002f Beaubien","Laval \u002f Duluth","Regina \u002f Verdun","March\u00e9 Atwater","Plessis \u002f Ren\u00e9-L\u00e9vesque","Parc Baldwin (Fullum \u002f Sherbrooke)","Quai de la navette fluviale","Boyer \u002f du Mont-Royal"],"type":"scattermapbox"},{"lat":[45.506620675516224,45.51008192332268,45.52959165921788,45.489860965686276,45.49949077777778,45.44501535,45.51643190322581,45.45750458333333,45.51374675,45.52379394594595,45.55960628865979,45.54031107488987,45.54695723766816,45.47032542307692,45.5352761875,45.53439681557377,45.55216284713376,45.55489346666667,45.618468863636366,45.47121552272727,45.579983222222225,45.43372825925926,45.560692764705884,45.483445889570554,45.52538714840989],"lon":[-73.5733722861357,-73.55828854952077,-73.57600795251396,-73.56844755882354,-73.63037352525252,-73.59625698333333,-73.69177716129032,-73.57433014285714,-73.52965847499999,-73.61110413513514,-73.65257679381443,-73.6238831585903,-73.60117464573992,-73.62241678846154,-73.50999528125,-73.55627273770492,-73.57554287261146,-73.72038246666666,-73.51232613636364,-73.55234404545455,-73.57530688888889,-73.679672,-73.54237333088236,-73.58550157055215,-73.59197768551236],"mode":"text","text":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"],"textfont":{"size":25},"type":"scattermapbox"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"mapbox":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{"lat":45.5195951475085,"lon":-73.58518631370328},"accesstoken":"pk.eyJ1IjoiYW5kcmV3eWV3Y3kiLCJhIjoiY2xsM3RqZ3luMDhndjNmbW9peHdzd2dydyJ9.M3UDgfPKJrePL1QlPj8fmQ\n","zoom":10.7},"legend":{"tracegroupgap":0},"title":{"text":"Figure 12. Cluster Centers from Final Model Overlayed over Montreal","x":0.5,"y":0.05},"autosize":false,"width":1000,"height":1000,"showlegend":false},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2e6ca530-593a-48d8-8ba9-3b1a6bbf2dd7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Overlaying the cluster centers over an actual map of Montreal reveals that the most important cluster is centered around McGill University, likely due to the large amounts of bicycle traffic by students. Further detailed analysis for each cluster center with its surrounding landmarks was now possible, but not included in this article for brevity.

## Final Evaluation

For final reporting, the trained model was evaluated using the test set, which had been excluded throughout the entire model training process.


```python
# Extract the independent and dependent variables into X and y for readability
X_te = test_df[["stn_lat", "stn_lon"]]
y_te = test_df["pct_of_annual_rides"]

# Generate final predictions
final_predictions = final_model.predict(X_te)

# Calculate and display final RMSE error
final_rmse = mean_squared_error(y_te, final_predictions, squared = False)
print(f"Final RMSE: {np.round(final_rmse * 100,5)} %")
```

    Final RMSE: 0.05608 %


To better represent the confidence in a model's prediction, the 95% confidence interval calculated below.


```python
# Define confidence interval
confidence = 0.95

# Define squared errors
squared_errors = (final_predictions - y_te) **2

# Calculate a 95 % confidence interval
ci = np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc = squared_errors.mean(),
        scale = stats.sem(squared_errors)
    )
)*100

print(f"Final RMSE: {np.round(final_rmse * 100,5)} % with a 95% CI of {np.round(ci,5)} %.")
```

    Final RMSE: 0.05608 % with a 95% CI of [0.0477  0.06336] %.


The final model was able to predict the percentage of annual rides for each station given its latitude and longitude with a RMSE of 0.05608%. To better understand what this implies, the final error was analyzed against the 3 categories created in the stratification step.


```python
# Display range of "percentage of annual rides" for each station
print(f"The distribution of `percentage of annual rides` in % by Percentiles")
df.groupby(by = "pct_cat")["pct_of_annual_rides"].describe().iloc[:,3:].T*100
```

    The distribution of `percentage of annual rides` in % by Percentiles





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>pct_cat</th>
      <th>00 - 33% percentile</th>
      <th>34 - 66% percentile</th>
      <th>67 - 100% percentile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>0.000018</td>
      <td>0.091008</td>
      <td>0.197032</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.031090</td>
      <td>0.112905</td>
      <td>0.239177</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.055255</td>
      <td>0.136307</td>
      <td>0.301887</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.073105</td>
      <td>0.162927</td>
      <td>0.393963</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.091000</td>
      <td>0.196958</td>
      <td>1.453954</td>
    </tr>
  </tbody>
</table>
</div>



For the first category, the `00-33% percentile`, the final model's RMSE was larger than the median (0.05608% > 0.055255%), which means the model will likely perform poorly for stations with little to no annual usage.

As the second and third categories represent stations with greater usage, the final model's RMSE suggests it performs best when predicting the percentage of annual rides for stations with the higest number of rental trips.

Finally, to evaluate the model's ability to predict the number of Bixi trips across the years based on station location, the final model used to predict the percentage of annual rides for each station, which was then multiplied by the total rides for each year. The result was then summed and compared to the actual number of rides each year in a bar chart.


```python
# Store station location and latitudes into a neat variable X
X = df[["stn_lat","stn_lon"]]

# Generate predictions using the final model
with print_time():
    y_pr = final_model.predict(X)
```

    Time taken: 0.1011 seconds.



```python
# Store the final predictions in DataFrame
df["pred_pct_of_annual_rides"] = y_pr
df["pred_rides"] = df["pred_pct_of_annual_rides"] * df.groupby(by = ["year"])["rides"].transform("sum")
```


```python
# Create a pivot table summarizing the actual number of trips per year and predicted number of trips
plot_df = df.groupby(
    by = ["year"],
    as_index = False
).agg(
    actual_trips = ("rides","sum"),
    predicted_trips = ("pred_rides","sum")
)

# Define x-axis, bar width, and multiplier for counting bar position
x = plot_df["year"].unique()
width = 0.25
multiplier = 0

# Create a plot summarizing the actual number of trips per year and predicted number of trips
fig, ax = plt.subplots(
    layout = "constrained"
)

for attribute, measurement in plot_df[["actual_trips", "predicted_trips"]].items():
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(measurement/1_000_000,1), width, label=attribute)
    ax.bar_label(rects, padding=4, rotation = 90)
    multiplier += 1

# Tidy up the plot
ax.set_xlabel(f"Year")
ax.set_ylabel(f"Number of Trips (millions)")
ax.set_title(f"Figure 13. Predicted Rides for each Year", y = -0.2)
ax.legend(loc='upper left', ncols=3)

sns.despine()
plt.show()
```


    
![png](../assets/images/output_166_0.png){:class="img-responsive"}
    


Although the final model was observed to perform unsatisfactorily for stations with number of trips in the 0-33% quantile, the overall performance of the final model on an annual basis suggests otherwise. The model was able to predict the number of trips accurately within 0.1 million trips for the years 2014 to 2019, only deviating significantly by 0.7 million trips for the year 2021. This is likely due to the errors for stations with little trips being insignificant compared to more busy stations, thus offering little to no effect on the overall annual sum. The deviation in 2021 might be due to the effects of recovery from Covid-19, but further detailed analysis was required to verify this. In retrospect, the difference observed in 2021 might suggest that the model has not learned the one-time effect of the Covid-19 pandemic, which would make it more generizable to new data.


```python
# Final numerical check that the final model was actually making predictions
display(df.loc[0:5, ["year", "start_stn_code", "rides","pred_rides"]].round(0))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>start_stn_code</th>
      <th>rides</th>
      <th>pred_rides</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021</td>
      <td>10</td>
      <td>4478.0</td>
      <td>3855.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021</td>
      <td>100</td>
      <td>6871.0</td>
      <td>5768.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021</td>
      <td>1000</td>
      <td>1093.0</td>
      <td>1140.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021</td>
      <td>1002</td>
      <td>4469.0</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>1003</td>
      <td>3179.0</td>
      <td>3134.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021</td>
      <td>1004</td>
      <td>1383.0</td>
      <td>1035.0</td>
    </tr>
  </tbody>
</table>
</div>


## Saving the Final Model


```python
# The final model was saved in the models folder
joblib.dump(final_model, f"../13_models/{time.strftime('%Y%m%d-%H%M')}_pct_annual_trips_model.pkl")
```




    ['../13_models/20231018-1536_pct_annual_trips_model.pkl']



# Conclusion

In conclusion, the station location coordinates were used to train a random forest model to predict the percentage of annual rides at each station with a final RMSE of 0.05608% of annual rides and a 95% confidence interval of 0.04770% and 0.06336%, regardless of year. KMeans clustering was used to identify cluster centers among stations. Then, RBF Kernel was used to measure the similarity of each station to those cluster centers, which form the features which the random forest model uses to make predictions. The model was observed to perform best for stations with high annual trips, and more work is required to fine tune the model for stations with trips below the 33% percentile. 

# References and Acknowledgement

- [Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/), pg 39-101, Aurélien Géron, 2022
- [Stackoverflow: partially shade the histogram in Python](https://stackoverflow.com/questions/36525121/partially-shade-the-histogram-in-python), user707650, 2016-04-10
- [Scikit Learn API](https://scikit-learn.org/stable/modules/classes.html)
