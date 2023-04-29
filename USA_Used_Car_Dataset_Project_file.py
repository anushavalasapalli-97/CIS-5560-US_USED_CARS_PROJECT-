# Databricks notebook source
IS_DB = True # Run the code in Databricks

PYSPARK_CLI = False
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/used_cars_data_set.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------


temp_table_name = "used_cars_data_set_Csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

if PYSPARK_CLI:
   df = spark.read.csv('used_cars_data_set.csv', inferSchema=True, header=True)
else:
    df = spark.sql("SELECT * FROM used_cars_data_set_Csv")


#df.show(5)

# COMMAND ----------

#df.take(10)

# COMMAND ----------

#df.dtypes

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType,IntegerType
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit, CrossValidator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator



# COMMAND ----------

# DBTITLE 1,Typecast string to int or float
df2 = df.withColumn("year",col("year").cast("int")).withColumn("city_fuel_economy",col("city_fuel_economy").cast("float")).withColumn("price",col("price").cast("float")).withColumn("mileage",col("mileage").cast("float")).withColumn("owner_count",col("owner_count").cast("int")).withColumn("latitude",col("latitude").cast("float")).withColumn("highway_fuel_economy",col("highway_fuel_economy").cast("float"))

# COMMAND ----------

display(df2.select("year", "city_fuel_economy", "price", "mileage", "owner_count", "latitude", "highway_fuel_economy").limit(10))

# COMMAND ----------

# DBTITLE 1,Convert dataset to pandas
df3 = df2.toPandas()

# COMMAND ----------

#df3.dtypes

# COMMAND ----------

#display(df3)

# COMMAND ----------

data1=df2

# COMMAND ----------

#df2.describe

# COMMAND ----------

# DBTITLE 1,Remove the null values and filter the price to more than 100,000 
#remove nan values for price and year columns and store result in new df

df2.na.drop(subset=["price", "year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"])
df22 = df2.filter(col("price") < 100000)

# COMMAND ----------

# DBTITLE 1,Split the data into Train and Test datasets.
splits = df22.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("TRAINING ROWS:", train_rows, "TESTING ROWS:", test_rows)

# COMMAND ----------

# DBTITLE 1,Algorithm 1: Linear Regression
assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame with vector assembler
training = assembler.transform(train).select(col("features"), (col("price").cast("Int").alias("label")))
#training.show()

# COMMAND ----------

# DBTITLE 1,Train a Regression Model
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
model = lr.fit(training)
print("Model trained!")

# COMMAND ----------

# DBTITLE 1,Prepare the Testing Data
testing = assembler.transform(test).select(col("features"), col("price").cast("Int").alias("trueLabel"))
#testing.show()

# COMMAND ----------

# DBTITLE 1,Test the Model
prediction = model.transform(testing)
# does not need probability parameter of Logistic Regression
predicted = prediction.select("features", col("prediction").cast("Int"), "trueLabel")

predicted.show(10)

# COMMAND ----------

display(predicted.limit(1000));

# COMMAND ----------

trainingSummary = model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="trueLabel",metricName="r2")
print("R Squared (R2) on test data = %g" %
lr_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,GBTRegressor
gbt_assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame for GBT
training = gbt_assembler.transform(train)
#training.show()



# COMMAND ----------

# DBTITLE 1,sets up a model with GBTRegressor
gbt = GBTRegressor(labelCol="price") #,featuresCol="features",maxIter=10

# COMMAND ----------

# DBTITLE 1,create a parameter combination, which is to tune the model.
paramGrid = ParamGridBuilder()\
.addGrid(gbt.maxDepth, [2, 5])\
.addGrid(gbt.maxIter, [10, 20])\
.build()

# COMMAND ----------

# DBTITLE 1,create a evaluator to evaluate a model with R2
from pyspark.ml.evaluation import RegressionEvaluator
gbt_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="price",metricName="r2")

# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
pipeline = Pipeline(stages=[gbt_assembler, gbt]) 

# COMMAND ----------

# DBTITLE 1,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv = CrossValidator(estimator=pipeline, evaluator=gbt_evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of cross validator
model = cv.fit(train) 

# COMMAND ----------

# DBTITLE 1,calculate the feature importance of GBT algorithm
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "price") 
predicted.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
display(predicted.limit(1000))

# COMMAND ----------

# DBTITLE 1,To evaluate the prediction result R2 (Coefficient of Determination) and RMSE of the model with the test data:
print("R Squared (R2) on test data = %g" % gbt_evaluator.evaluate(prediction))
 
gbt_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % gbt_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,You observe that the coefficient, R2, is  0.675 which is not much closer to 1, comparing to the linear Regression: RMSE is 6728.73 In the previous model, we had : [R2: 0.436939, RMSE: 8837.227827]
from pyspark.ml.evaluation import RegressionEvaluator
 
gbt_evaluator1 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="price",metricName="r2")


# COMMAND ----------

# DBTITLE 1,use TrainValidationSplit instead of CrossValidator
cv1 = TrainValidationSplit(estimator=pipeline, evaluator=gbt_evaluator1, estimatorParamMaps=paramGrid, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model1 = cv1.fit(train)

# COMMAND ----------

# DBTITLE 1,calculate the feature importance of GBT algorithm using TrainValidationSplit
prediction = model1.transform(test)
predicted = prediction.select("features", "prediction", "price") 
predicted.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
display(predicted.limit(1000))

# COMMAND ----------

# DBTITLE 1,To evaluate the prediction result R2 (Coefficient of Determination) and RMSE of the model with the test data:
print("R Squared (R2) on test data = %g" % gbt_evaluator1.evaluate(prediction))
 
gbt_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % gbt_evaluator.evaluate(prediction))


##You observe that the coefficient, R2, is  0.675 which is some what closer to 1, comparing to the linear Regression: RMSE for train validation split is 6728.73 In the previous model, we had : [R2: 0.436939, RMSE: 8837.227827]

# COMMAND ----------

# And, you observe that the coefficient, R2, is 0.675 and RMSE is 6728.73 In the
# previous model, we have : [R2: 0.675, RMSE: 6728.73], which is same as in CrossValidator.
# TrainValidationSplit has the similar generalization to but much faster than CrossValidator

# COMMAND ----------

# DBTITLE 1,Random Forest Regressor
assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame for Random Forest Regressor
training = assembler.transform(train)
#training.show()

# COMMAND ----------

# DBTITLE 1,sets up a model with Random Forest Regressor
rf = RandomForestRegressor(labelCol='price') #,featuresCol="features", numTrees=10, maxDepth=5

# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
pipeline = Pipeline(stages=[assembler, rf])

# COMMAND ----------

# DBTITLE 1, Tune the model for the generalization. 
model = pipeline.fit(train)

# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe 
rfModel = model.stages[-1]
#print(rfModel.toDebugString)

# COMMAND ----------

# DBTITLE 1,show the importance of the features – columns – in the order of importance. NOTE: this code uses pandas library in standard Python not in PySpark.
import pandas as pd
featureImp = pd.DataFrame(list(zip(assembler.getInputCols(), rfModel.featureImportances)), columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending="False")

# COMMAND ----------

# DBTITLE 1,Build an RF model using pipeline with the paramters (numTrees=10, maxDepth=5) and train data set.
rf = RandomForestRegressor(labelCol="price", featuresCol="features", numTrees=10, maxDepth=5)

# combine stages into pipeline
pipeline = Pipeline(stages= [assembler, rf])
model = pipeline.fit(train)

# COMMAND ----------

#display(test)

# COMMAND ----------

# DBTITLE 1,Testing data frame on the model that we built. Then, the model predicts the value “prediction” that the price of used cars counts. It can be compared to “price” in the target column, the actual price of the used cars.
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "price")
predicted.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
display(predicted.limit(1000))

# COMMAND ----------

# DBTITLE 1,Evaluate the prediction result of the model with the test data:
from pyspark.ml.evaluation import RegressionEvaluator
rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                   labelCol="price",metricName="r2")

print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(prediction))

rf_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
print("RMSE: %f" % rf_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,Decision Tree Classifier
flightSchema = StructType([
  StructField("year", IntegerType(), False),
  StructField("price", IntegerType(), False),
  StructField("mileage", StringType(), False),
  StructField("owner_count", IntegerType(), False),
  StructField("seller_rating", IntegerType(), False),
  StructField("city_fuel_economy", IntegerType(), False),
  StructField("highway_fuel_economy", IntegerType(), False),
])

# COMMAND ----------

# DBTITLE 1,To display the selected columns
display(df2.select("year", "city_fuel_economy", col("price").alias("label"), "mileage", "owner_count", "latitude", "highway_fuel_economy").limit(10))


# COMMAND ----------

# DBTITLE 1,Data Conversion and Data Filtering
df2 = df2.filter(col("price") < 100000)
display(df2.limit(10))
df23 = df2.select("year", "city_fuel_economy", ((col("price") > 10000).cast("Double").alias("label")), "mileage", "owner_count", "highway_fuel_economy")


# COMMAND ----------

# DBTITLE 1,Drop the Unused Columns
df23.na.drop(subset=["label", "year", "city_fuel_economy", "mileage","owner_count", "highway_fuel_economy"])
#display(df23)


# COMMAND ----------

# DBTITLE 1,Split the Data
splits = df23.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# the following columns are categorical number such as ID so that it should be Category features
catVect = VectorAssembler(inputCols = ["mileage", "owner_count", "city_fuel_economy", "highway_fuel_economy"], outputCol="catFeatures").setHandleInvalid("skip")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures").setHandleInvalid("skip")

# COMMAND ----------

# number is meaningful so that it should be number features
numVect = VectorAssembler(inputCols = ["year"], outputCol="numFeatures")
# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")  #="features1")
# number vector is normalized: this changes the accuracy of precision a little
# minMax2 = MinMaxScaler(inputCol = featVect.getOutputCol(), outputCol="features")

#dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)

# Pipeline process the series of transformation above, which is 7 transformation
pipeline = Pipeline(stages=[catVect, catIdx, numVect, minMax, featVect, lr]) #minMax2, lr])

# COMMAND ----------

# DBTITLE 1,Tune Parameters
paramGrid = (ParamGridBuilder() \
             .addGrid(lr.regParam, [0.01, 0.5]) \
             .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
             .addGrid(lr.maxIter, [1, 5]) \
             .build())
#paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1, 0.01]).addGrid(lr.maxIter, [10, 5]).addGrid(lr.threshold, [0.35, 0.30]).build()

# COMMAND ----------

# DBTITLE 1,CrossValidator vs TrainValidationSplit

cv = TrainValidationSplit(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

model = cv.fit(train)

# COMMAND ----------

# DBTITLE 1,Test the Pipeline Model (The model produced by the pipeline is a transformer that will apply all of the stages in the pipeline to a specified DataFrame and apply the trained model to generate predictions. In this case, you will transform the test DataFrame using the pipeline to generate label predictions.)
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "probability", "trueLabel")

predicted.show(100, truncate=False)

# COMMAND ----------

# DBTITLE 1, Compute Confusion Matrix Metrics Classifiers are typically evaluated by creating a confusion matrix, which indicates the number of:  True Positives True Negatives False Positives False Negatives From these core measures, other evaluation metrics such as precision and recall can be calculated.
tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])

# COMMAND ----------

metrics.show() 

# COMMAND ----------

# DBTITLE 1,View the Raw Prediction and Probability The prediction is based on a raw prediction score that describes a labelled point in a logistic function. This raw prediction is then converted to a predicted label of 0 or 1 based on a probability vector that indicates the confidence for each possible label value (in this case, 0 and 1). The value with the highest confidence is selected as the prediction.
prediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate=False)

# COMMAND ----------

# DBTITLE 1,Note that there are now more True Positives and less False Negatives, and Recall has improved. By changing the discrimination threshold, the model now gets more predictions correct - though it’s worth noting that the number of False Positives has also increased.  Calculate the update AUC


# COMMAND ----------

# DBTITLE 1,Review the Area Under ROC Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a BinaryClassificationEvaluator class that you can use to compute this. The ROC curve shows the True Positive and False Positive rates plotted for varying thresholds. If AUC is 0.9227 (=92%), it is very high.
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print("AUC = ", auc)
