
## NYC Taxi Fare Prediction Using Big Data Technologies
This project predicts NYC taxi fares using Big Data technologies, including Databricks and Spark MLlib. The dataset is sourced from the NYC Taxi Trip dataset.

## Dataset Link
NYC Taxi Trip Data (2024 September yellow taxi trip data subset). Link: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Prerequisites:
1. Databricks account
2. Python environment with necessary libraries installed (pyspark, pandas, numpy, etc.)
3. Download all the files from the Zip file 

## Setup Instructions(for the file named data bricks, used to meet the project requirements and done some basic tasks like data cleaning, filtering, and exploring and basic model building and half part of feature engineering  )
Step 1: Create a Cluster in Databricks
- Log in to your Databricks account.
- Go to the "Clusters" section and create a new cluster with your desired configurations (ensure it supports Spark).

Step 2: Upload the Dataset
- Download the NYC Taxi Trip dataset in a zip file format.
- Go to the "Data" tab in Databricks and upload the dataset to the DBFS (Databricks File System).
- Once uploaded, choose the option to "Create Table in Notebook."
- Change the dataset name to `yellow_tripdata` for uniformity.

Step 3: Import and Run the Databricks File
- Upload the provided Databricks notebook file into your workspace.
- You can run all cells.

## For Complete Results:(for the file named jupyter,- this is the whole source file which contains(a-z for this project)

You can upload the ipynb file (that is uploaded in our zip file) in the Jupyter notebook.

Have the below files in the local environment:
- Sep_weather_data_cleaned.csv
- taxi-zone-lookup.csv
- yellow_tripdata_2024-09.parquet

Once the ipynb file is uploaded along with the correct file paths to the above files you can run the codes to get results.(run all cells and replace file paths wherever necessary)

##You can go thorugh HTML file if you only want to view the code and results. 