from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, count, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Initialize Spark session
spark = SparkSession.builder \
    .appName("Crime Analysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# 2. Load the data
file_path = '/content/crime_datasheet.csv'
df = spark.read.option("header", "true").csv(file_path, inferSchema=True)

# 3. Clean the dataset (drop unnecessary columns and clean data)
columns_to_drop = ["unique_key", "case_number", "location",
                   "fbi_code", "x_coordinate", "y_coordinate", "updated_on",
                   "latitude", "longitude"]
df_cleaned = df.drop(*columns_to_drop).dropna()

# Convert date to datetime
df_cleaned = df_cleaned.withColumn("date", to_date(col("date"), "MM/dd/yyyy HH:mm:ss"))
df_cleaned = df_cleaned.withColumn("year", year(col("date"))).withColumn("month", month(col("date")))

# Filter data for the years 2022, 2023, 2024
df_recent = df_cleaned.filter(col("year").isin([2022, 2023, 2024]))


# Additional Analytics from Original Code
# 5. Crime Distribution by Year
crime_by_year = df_recent.groupBy("year").agg(count("year").alias("crime_count")).orderBy("year").toPandas()
plt.figure(figsize=(12, 8))
sns.lineplot(data=crime_by_year, x="year", y="crime_count", marker="o")
plt.title("Crime Distribution by Year")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.show()

# 6. Crime Distribution by Month
crime_by_month = df_recent.groupBy("month").agg(count("month").alias("crime_count")).orderBy("month").toPandas()
plt.figure(figsize=(12, 8))
sns.lineplot(data=crime_by_month, x="month", y="crime_count", marker="o")
plt.title("Crime Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.show()

# 7. Top Crime Types
top_crimes = df_recent.groupBy("primary_type") \
    .agg(count("primary_type").alias("crime_count")).orderBy(col("crime_count").desc()).toPandas()
plt.figure(figsize=(12, 8))
sns.barplot(data=top_crimes.head(10), x="crime_count", y="primary_type", palette="coolwarm")
plt.title("Top 10 Crime Types")
plt.xlabel("Number of Crimes")
plt.ylabel("Crime Type")
plt.show()

# 8. Graphs Based on Location (Number of Cases in Recent Months)
recent_months = df_recent.filter(col("date") >= to_date(lit("2023-06-01"), "yyyy-MM-dd"))

location_crimes = recent_months.groupBy("location_description").agg(count("date").alias("crime_count")).orderBy(col("crime_count").desc()).toPandas()

plt.figure(figsize=(12, 8))
sns.barplot(data=location_crimes.head(10), x="crime_count", y="location_description", palette="coolwarm")
plt.title("Top 10 Locations by Crime Incidents (Recent Months)")
plt.xlabel("Number of Crimes")
plt.ylabel("Location Description")
plt.show()

# 4. Feature Engineering
indexer = StringIndexer(inputCol="location_description", outputCol="location_index")
df_indexed = indexer.fit(df_recent).transform(df_recent)

# Aggregate crime data by location, year, and month
location_monthly_df = df_indexed.groupBy("location_description", "location_index", "year", "month") \
    .agg(count("date").alias("crime_count"))

# Prepare data for Random Forest
assembler = VectorAssembler(inputCols=["location_index", "year", "month"], outputCol="features")
location_features_df = assembler.transform(location_monthly_df)

# Split data into training and test sets
(train_data, test_data) = location_features_df.randomSplit([0.8, 0.2], seed=1234)

# Train Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="crime_count", numTrees=100, maxBins=40)
rf_model = rf.fit(train_data)

# Make predictions on the test data
predictions = rf_model.transform(test_data)
predictions.select("features", "crime_count", "prediction").show(10)

# Predict next 12 months for each location
future_months = spark.createDataFrame([(i,) for i in range(1, 13)], ["month"])
all_predictions = []

for location in df_indexed.select("location_description", "location_index").distinct().collect():
    location_str = location["location_description"]
    location_idx = location["location_index"]
    for year in [2024]:
        future_data = future_months.withColumn("location_index", lit(location_idx)).withColumn("year", lit(year))
        future_features = assembler.transform(future_data)
        future_predictions = rf_model.transform(future_features)
        future_predictions = future_predictions.withColumn("location_description", lit(location_str))
        all_predictions.append(future_predictions.select("month", "prediction", "location_description"))

predictions_df = all_predictions[0]
for i in range(1, len(all_predictions)):
    predictions_df = predictions_df.union(all_predictions[i])

# Convert to pandas for visualization
predictions_pd = predictions_df.toPandas()
predictions_pd["prediction"] = predictions_pd["prediction"].apply(lambda x: max(0, x))  # Ensuring no negative predictions

# Plot predictions
plt.figure(figsize=(12, 8))
sns.lineplot(data=predictions_pd, x="month", y="prediction", hue="location_description", marker="o")
plt.title("Crime Predictions by Location for Next Year")
plt.xlabel("Month")
plt.ylabel("Predicted Number of Crimes")
plt.legend(title='Location Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



# 9. Correlation Matrices
columns_to_correlate = ["arrest", "domestic", "year", "location_description", "description", "primary_type"]

# Convert selected columns to a Pandas DataFrame
correlation_df = df_recent.select(*columns_to_correlate).toPandas()

# Check if the DataFrame is successfully created
print(correlation_df.head())

# List of string columns to encode
columns_to_encode = ["location_description", "description", "primary_type"]

# Initialize a dictionary to store LabelEncoders
label_encoders = {}

# Apply Label Encoding to string columns
for col in columns_to_encode:
    le = LabelEncoder()
    correlation_df[col] = le.fit_transform(correlation_df[col])
    label_encoders[col] = le

# Calculate the correlation matrix
correlation_matrix = correlation_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
