# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

#Creating a table for Data Exploration
# File location and type
file_location = "/FileStore/tables/yellow_tripdata_2024_09.parquet"
file_type = "parquet"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "yellow_tripdata"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `yellow_tripdata`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "yellow_tripdata"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC --1)Data Exploration
# MAGIC --Exploring where Passenger count is greater than 0
# MAGIC
# MAGIC
# MAGIC select count(*) from yellow_tripdata where passenger_count >0;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --Exploring where the total_amount is <0
# MAGIC select count(*)  from yellow_tripdata where total_amount <'0';
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --Exploring the outliers for filtering the data
# MAGIC select count(*) from yellow_tripdata where total_amount>'350';

# COMMAND ----------

# MAGIC %sql
# MAGIC --Final data rowsfor  the project
# MAGIC SELECT COUNT(*) 
# MAGIC FROM yellow_tripdata 
# MAGIC WHERE total_amount >0 and total_amount<350;
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# COMMAND ----------

#Reading the Data
from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("NYC Taxi Fare Prediction") \
    .getOrCreate()

# Read the Parquet file
taxi_sep_2024 = spark.read.parquet("/FileStore/tables/yellow_tripdata_2024_09.parquet")

# Concatenate the DataFrame (single DataFrame in this case)
taxi_data = taxi_sep_2024

# Show the first few rows
taxi_data.show()

# Get the shape of the DataFrame
row_count = taxi_data.count()
column_count = len(taxi_data.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

#DATA EXPLORATION
# Get the list of column names
columns = taxi_data.columns
print(columns)

# Select specific columns
taxi_data = taxi_data.select(
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'passenger_count',
    'trip_distance',
    'RatecodeID',
    'PULocationID',
    'DOLocationID',
    'payment_type',
    'total_amount'
)

# Show the first few rows of the DataFrame
taxi_data.show()

# Get the shape of the DataFrame
row_count = taxi_data.count()
column_count = len(taxi_data.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

#Histograms of each Columns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder.appName("HistogramExample").getOrCreate()

# Assuming `df` is your PySpark DataFrame
# Replace "df" with the DataFrame you are working on
numeric_columns = [col for col, dtype in df.dtypes if dtype in ("int", "double", "float", "bigint")]

# Convert the numeric columns to Pandas for visualization
df_pandas = df.select(numeric_columns).toPandas()

# Plot histograms for all numeric columns
df_pandas.hist(figsize=(20, 10), bins=60)
plt.tight_layout()
plt.show()


# COMMAND ----------

#import matplotlib.pyplot as plt
#from pyspark.sql import SparkSession

# Create SparkSession
#spark = SparkSession.builder.appName("HistogramExample").getOrCreate()

# Assuming `df` is your PySpark DataFrame
# Replace "df" with the DataFrame you are working on
#numeric_columns = [col for col, dtype in df.dtypes if dtype in ("int", "double", "float", "bigint")]

# Convert the numeric columns to Pandas for visualization
#df_pandas = df.select(numeric_columns).toPandas()

# Plot histograms for all numeric columns
#df_pandas.hist(figsize=(20, 10), bins=60)
#plt.tight_layout()
#plt.show()


# COMMAND ----------

#scatter Ployts
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

# Create a SparkSession
spark = SparkSession.builder.appName("ScatterPlotExample").getOrCreate()

# Filter rows where total_amount < 1000
filtered_df = taxi_data.filter(taxi_data['total_amount'] < 1000)

# Add an index column (to simulate reset_index in Pandas)
df_with_index = filtered_df.withColumn("index", F.monotonically_increasing_id())

# Convert the PySpark DataFrame to a Pandas DataFrame
pandas_df = df_with_index.select("index", "total_amount").toPandas()

# Create the scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(pandas_df['index'], pandas_df['total_amount'], alpha=0.6)
plt.title("Scatter Plot of Total Amount vs Index")
plt.xlabel("Index")
plt.ylabel("Total Amount")
plt.grid(True)
plt.show()

# COMMAND ----------

# Count payment_type values where total_amount < 0
negative_total_amount = taxi_data.filter(taxi_data['total_amount'] < 0)
payment_type_counts = negative_total_amount.groupBy('payment_type').count()

# Show the result
payment_type_counts.show()

# Get the shape of the DataFrame where total_amount < 0
row_count = negative_total_amount.count()
column_count = len(negative_total_amount.columns)
print(f"Shape: ({row_count}, {column_count})")

# COMMAND ----------

#PySpark Code for Histogram of trip_distance for Negative total_amount
import matplotlib.pyplot as plt

# Filter rows where total_amount < 0
negative_total_amount = taxi_data.filter(taxi_data['total_amount'] < 0)

# Collect `trip_distance` data for plotting
trip_distance_data = negative_total_amount.select("trip_distance").rdd.flatMap(lambda x: x).collect()

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.hist(trip_distance_data, bins=60, color='blue', alpha=0.7)
plt.title("Histogram of Trip Distance for Negative Total Amounts", fontsize=14)
plt.xlabel("Trip Distance", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# COMMAND ----------

#data Cleaning
# Filter rows where total_amount > 350
greater_than_350 = taxi_data.filter(taxi_data['total_amount'] > 350)

# Get the shape of the DataFrame
row_count_gt_350 = greater_than_350.count()
column_count_gt_350 = len(greater_than_350.columns)
print(f"Shape where total_amount > 350: ({row_count_gt_350}, {column_count_gt_350})")
# Filter rows where total_amount == 0
equal_to_0 = taxi_data.filter(taxi_data['total_amount'] == 0)

# Get the shape of the DataFrame
row_count_eq_0 = equal_to_0.count()
column_count_eq_0 = len(equal_to_0.columns)
print(f"Shape where total_amount == 0: ({row_count_eq_0}, {column_count_eq_0})")

#ignoring the data which the amounts are greater than 350 and equal to zero


# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

# Create a SparkSession
spark = SparkSession.builder.appName("PySpark Operations").getOrCreate()

# Filter rows where total_amount > 0 and total_amount < 350
taxi_data_filtered = taxi_data.filter((taxi_data['total_amount'] > 0) & (taxi_data['total_amount'] < 350))

# Show the first few rows
taxi_data_filtered.show()

# Count null values in each column
null_counts = taxi_data_filtered.select(
    [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in taxi_data_filtered.columns]
)
print("Null values in each column:")
null_counts.show()

# Filter rows where passenger_count is null
null_passenger_count = taxi_data_filtered.filter(F.col('passenger_count').isNull())

# Add an index column (like reset_index in Pandas)
null_passenger_count = null_passenger_count.withColumn("index", F.monotonically_increasing_id())

# Convert the DataFrame to Pandas for plotting
pandas_df = null_passenger_count.select("index", "total_amount").toPandas()

# Create scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(pandas_df['index'], pandas_df['total_amount'], alpha=0.6)
plt.title("Scatter Plot of Total Amount for Null Passenger Count", fontsize=14)
plt.xlabel("Index", fontsize=12)
plt.ylabel("Total Amount", fontsize=12)
plt.grid(True)
plt.show()

# Drop rows with null values
taxi_data_filtered = taxi_data_filtered.na.drop()

# Get the shape of the DataFrame
row_count = taxi_data_filtered.count()
column_count = len(taxi_data_filtered.columns)
print(f"Shape after dropping nulls: ({row_count}, {column_count})")

# COMMAND ----------

##Data Preparation
from pyspark.sql.functions import col, year, month, dayofmonth, hour, to_date

# Make a copy of the DataFrame (taxi_data_filtered is already filtered)
taxi_data_prepared = taxi_data_filtered

# Convert specified columns to string type
taxi_data_prepared = taxi_data_prepared.withColumn("RatecodeID", col("RatecodeID").cast("string"))
taxi_data_prepared = taxi_data_prepared.withColumn("PULocationID", col("PULocationID").cast("string"))
taxi_data_prepared = taxi_data_prepared.withColumn("DOLocationID", col("DOLocationID").cast("string"))
taxi_data_prepared = taxi_data_prepared.withColumn("payment_type", col("payment_type").cast("string"))

# Add transaction_date (date only, without time)
taxi_data_prepared = taxi_data_prepared.withColumn("transaction_date", to_date(col("tpep_pickup_datetime")))

# Extract year, month, day, and hour from the pickup datetime
taxi_data_prepared = taxi_data_prepared.withColumn("transaction_year", year(col("tpep_pickup_datetime")))
taxi_data_prepared = taxi_data_prepared.withColumn("transaction_month", month(col("tpep_pickup_datetime")))
taxi_data_prepared = taxi_data_prepared.withColumn("transaction_day", dayofmonth(col("tpep_pickup_datetime")))
taxi_data_prepared = taxi_data_prepared.withColumn("transaction_hour", hour(col("tpep_pickup_datetime")))

# Show the first few rows of the prepared DataFrame
taxi_data_prepared.show()

# COMMAND ----------

#Filtering the Data columns for feature extraction
from pyspark.sql.functions import col, mean, count
from pyspark.sql import functions as F

# Filter rows for the year 2024
taxi_data_prepared = taxi_data_prepared.filter(col("transaction_year") == 2024)

# Filter rows for the month of September
taxi_data_prepared = taxi_data_prepared.filter(col("transaction_month") == 9)

# Get the shape of the filtered DataFrame
row_count = taxi_data_prepared.count()
column_count = len(taxi_data_prepared.columns)
print(f"Shape after filtering by year and month: ({row_count}, {column_count})")

# Filter rows where passenger_count > 0
taxi_data_prepared = taxi_data_prepared.filter(col("passenger_count") > 0)

# Get the new shape
row_count = taxi_data_prepared.count()
column_count = len(taxi_data_prepared.columns)
print(f"Shape after filtering by passenger_count > 0: ({row_count}, {column_count})")

# Define categorical and numerical columns
categorical_columns = ['PULocationID', 'transaction_date', 'transaction_month', 'transaction_day', 'transaction_hour']
numerical_columns = ['trip_distance', 'total_amount']
all_needed_columns = categorical_columns + numerical_columns

# Select the needed columns
main_taxi_df = taxi_data_prepared.select(*all_needed_columns)

# Get the shape of the main DataFrame
row_count = main_taxi_df.count()
column_count = len(main_taxi_df.columns)
print(f"Shape of the main DataFrame: ({row_count}, {column_count})")

# Show the first few rows
main_taxi_df.show()

# Group by categorical columns and calculate the mean of numerical columns
taxi_grouped_by_region = main_taxi_df.groupBy(categorical_columns).agg(
    *[mean(col).alias(f"mean_{col}") for col in numerical_columns]
)

# Add a column for the count of transactions
taxi_grouped_by_region = taxi_grouped_by_region.join(
    main_taxi_df.groupBy(categorical_columns).agg(count("*").alias("count_of_transactions")),
    on=categorical_columns,
    how="inner"
)

# Get the shape of the grouped DataFrame
row_count = taxi_grouped_by_region.count()
column_count = len(taxi_grouped_by_region.columns)
print(f"Shape of the grouped DataFrame: ({row_count}, {column_count})")

# Show the first few rows of the grouped DataFrame
taxi_grouped_by_region.show()

# COMMAND ----------

#Benchmark Model Building
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Define categorical and numerical features
categorical_features_benchmark = ['PULocationID', 'transaction_month', 'transaction_day', 'transaction_hour']
numerical_features = ['mean_trip_distance']  # Use mean_trip_distance instead of trip_distance
target_feature_benchmark = 'mean_total_amount'  # Use mean_total_amount as the target

# Filter out columns with only one distinct value
valid_categorical_features = [
    col for col in categorical_features_benchmark
    if taxi_grouped_by_region.select(col).distinct().count() > 1
]

# Index and encode valid categorical features
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in valid_categorical_features]
encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in valid_categorical_features]

# Assemble all valid features into a single vector
assembler_inputs = [f"{col}_encoded" for col in valid_categorical_features] + numerical_features
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Define the pipeline for transformation
pipeline = Pipeline(stages=indexers + encoders + [assembler])

# Transform the data
data_for_benchmark_model = pipeline.fit(taxi_grouped_by_region).transform(taxi_grouped_by_region)

# Select features and target for decision tree
final_data = data_for_benchmark_model.select("features", col(target_feature_benchmark).alias("label"))

# Split the data into training and test sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=10)

# Show the prepared training data (features and label)
train_data.show(5, truncate=False)
test_data.show(5, truncate=False)

# COMMAND ----------

#adding columns of what day it , it checks the day is holiday or not amd weekend or not
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, dayofweek, lit

# Initialize SparkSession
spark = SparkSession.builder.appName("Feature Engineering").getOrCreate()

# Create a sample DataFrame to simulate the original data
data = [
    ("2024-09-01",),
    ("2024-09-02",),
    ("2024-09-03",),
    ("2024-09-04",),
    ("2024-09-05",),
    ("2024-09-06",),
    ("2024-09-07",),
]
columns = ["transaction_date"]

data_with_new_features = spark.createDataFrame(data, columns)

# Correct the transaction_week_day calculation
data_with_new_features = data_with_new_features.withColumn(
    "transaction_week_day", ((dayofweek(col("transaction_date")) + 5) % 7)
)

# Add weekend column (True if transaction_week_day is 5 or 6, otherwise False)
data_with_new_features = data_with_new_features.withColumn(
    "weekend", when(col("transaction_week_day").isin([5, 6]), lit(True)).otherwise(lit(False))
)

# Manually define September 2024 holidays
september_2024_holidays = ["2024-09-02"]

# Add is_holiday column for September 2024
data_with_new_features = data_with_new_features.withColumn(
    "is_holiday",
    when(col("transaction_date").cast("string").isin(september_2024_holidays), lit(True)).otherwise(lit(False))
)

# Show the updated DataFrame with new features
data_with_new_features.select("transaction_date", "transaction_week_day", "weekend", "is_holiday").show()

# COMMAND ----------


#Feature Engineering- adding a new data
from pyspark.sql.functions import col

# Read the CSV file from DBFS
zone_lookup = spark.read.csv("/FileStore/tables/taxi_zone_lookup.csv", header=True, inferSchema=True)

# Select only the required columns
zone_lookup = zone_lookup.select("LocationID", "Borough")

# Convert LocationID to string type
zone_lookup = zone_lookup.withColumn("LocationID", col("LocationID").cast("string"))

# Show the first few rows
zone_lookup.show()

# COMMAND ----------

##Looking up which has most number of pickups
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import plotly.express as px

# Initialize Spark session
spark = SparkSession.builder.appName("BoroughAnalysis").getOrCreate()

# Create PySpark DataFrame
data = [
    ('Manhattan', 41057),
    ('Brooklyn', 14587),
    ('Queens', 14480),
    ('Bronx', 4700),
    ('Unknown', 1301),
    ('EWR', 273),
    ('Staten Island', 83)
]
columns = ['Borough', 'Pickup_Count']
spark_df = spark.createDataFrame(data, columns)

# Calculate value counts (already done in the provided data here)
# If starting from raw data, you can group and count as follows:
# spark_df = raw_data.groupBy("Borough").agg(F.count("*").alias("Pickup_Count"))

# Convert to Pandas DataFrame for Plotly
pandas_df = spark_df.toPandas()

# Create the bar chart with Plotly
fig = px.bar(
    pandas_df,
    x='Borough',
    y='Pickup_Count',
    title='Most Picked-Up Locations by Borough',
    text='Pickup_Count',
    color='Pickup_Count',  # Add color to differentiate values
    color_continuous_scale='Viridis'  # Use a visually appealing color scale
)

# Customize the layout
fig.update_traces(
    textposition='outside',
    marker_line_width=1.5,
    marker_line_color='black'
)
fig.update_layout(
    xaxis_title='Borough',
    yaxis_title='Number of Pickups',
    title_x=0.5,
    title_font_size=20,
    xaxis_tickangle=-45,  # Rotate x-axis labels for better visibility
    height=600,  # Increase the height of the chart
    width=1000,  # Increase the width of the chart
    font=dict(size=14),
    margin=dict(l=50, r=50, t=80, b=150),  # Adjust margins for readability
)

# Show the chart
fig.show()

# COMMAND ----------

#Loading a new dataset and attaching to the old table
from pyspark.sql.functions import col, regexp_replace, trim

# Read the CSV file with the specified encoding from DBFS
nyc_weather = spark.read.csv("/FileStore/tables/sep_weather_data_cleaned.csv", header=True, inferSchema=True, encoding='ISO-8859-1')

# Clean the dataset by removing problematic characters (Ê and its variations) from all string columns
string_columns = [field.name for field in nyc_weather.schema.fields if str(field.dataType) == "StringType"]

for col_name in string_columns:
    nyc_weather = nyc_weather.withColumn(
        col_name, trim(regexp_replace(col(col_name), r"[Ê\u00CA\u00C2\u202F\u00A0]", ""))
    )

# Display the cleaned dataset
nyc_weather.show()

# COMMAND ----------

#Exploring the data and dividing it to features
from pyspark.sql.functions import col, when

# Ensure the Precipitation column is numeric (convert invalid values to null)
nyc_weather = nyc_weather.withColumn(
    "Precipitation", col("Precipitation").cast("float")
)

# Fill NaN values in Precipitation with 0
nyc_weather = nyc_weather.fillna({"Precipitation": 0})

# Apply the weather type logic
nyc_weather = nyc_weather.withColumn(
    "Weather_Type",
    when(col("Precipitation") ==   0.00, "sunny")
    .when(col("Precipitation") < 0.1, "cloudy")
    .otherwise("rainy")
)

# Show the results
nyc_weather.select("Precipitation", "Weather_Type").show()

# COMMAND ----------

##machine learning model and full features of feature engineering are available in the python file , as the pyspark functionality and limited resources(Clusters) of the Data bricks running for the large dataset, taking longer time >30 mins and sometimes its crashing , so we had to running from the beginning, with professors permission he told to submit the rest of the whole code in python file , please refer to that, this file is to demonstrate that we have used all the technologies to meet the project requirements(hive , pyspark,mlib and sql and big data engines) for full code refer to the ipnyb file
