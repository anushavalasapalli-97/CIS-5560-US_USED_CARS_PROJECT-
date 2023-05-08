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

#display(df)
df.show()

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
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor, FMRegressor
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit, CrossValidator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


# COMMAND ----------

# DBTITLE 1,Typecast string to int or float
df2 = df.withColumn("year",col("year").cast("int")).withColumn("city_fuel_economy",col("city_fuel_economy").cast("float")).withColumn("price",col("price").cast("float")).withColumn("mileage",col("mileage").cast("float")).withColumn("owner_count",col("owner_count").cast("int")).withColumn("latitude",col("latitude").cast("float")).withColumn("highway_fuel_economy",col("highway_fuel_economy").cast("float"))

# COMMAND ----------

#display(df2.select("year", "city_fuel_economy", "price", "mileage", "owner_count", "latitude", "highway_fuel_economy").limit(10))
(df2.select("year", "city_fuel_economy", "price", "mileage", "owner_count", "latitude", "highway_fuel_economy").limit(10)).show()

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
df222 = df2.filter(col("price") < 100000)
df22 = df222.select("year", "city_fuel_economy", col("price").alias("label"), "mileage", "owner_count", "latitude", "highway_fuel_economy")

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
#assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

lr_assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame with vector assembler
#training = assembler.transform(train)
#training.show()

training = lr_assembler.transform(train)
training.show()

# COMMAND ----------

# DBTITLE 1,Train a Regression Model
# lr = LinearRegression(labelCol="price",featuresCol="features")
# model = lr.fit(training)
# print("Model trained!")

lr= LinearRegression(labelCol="label") #,featuresCol="features",maxIter=10

# COMMAND ----------

# DBTITLE 1,create a parameter combination, which is to tune the model.
# paramGrid_lr = ParamGridBuilder()\
# .addGrid(lr.maxIter, [10, 20])\
# .build()
paramGrid = ParamGridBuilder()\
.addGrid(lr.maxIter, [15, 20])\
.addGrid(lr.regParam, [0.5])\
.build()

# COMMAND ----------

# DBTITLE 1,create a evaluator to evaluate a model with R2
# from pyspark.ml.evaluation import RegressionEvaluator
# lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
# labelCol="price",metricName="r2")


from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")


# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
pipeline_lr = Pipeline(stages=[lr_assembler, lr]) 

# COMMAND ----------

# DBTITLE 1,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv_lr = CrossValidator(estimator=pipeline_lr, evaluator=lr_evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of cross validator
model_lr = cv_lr.fit(train) 

# COMMAND ----------

# DBTITLE 1,Predict price with actual price using Lr
prediction_lr = model_lr.transform(test)
predicted_lr = prediction.select("features", "prediction", "label") 
predicted_lr.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted_lr.limit(1000))
(predicted_lr.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,To evaluate the prediction result R2 (Coefficient of Determination) and RMSE of the model with the test data:using cross validator
#print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(prediction_lr))

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
 
lr_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 

print("r2: %f" % lr_evaluator_r2.evaluate(prediction_lr))
print("RMSE: %f" % lr_evaluator_rmse.evaluate(prediction_lr))

# COMMAND ----------

# DBTITLE 1,use TrainValidationSplit instead of CrossValidator
tv_lr = TrainValidationSplit(estimator=pipeline, evaluator=lr_evaluator_r2, estimatorParamMaps=paramGrid, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model1_tv = tv_lr.fit(train)

# COMMAND ----------

# DBTITLE 1,comparing the predicted results with actual results
prediction_tv_lr = model1_tv.transform(test)
predicted_tv_lr = prediction_tv_lr.select("features", "prediction", "label") 
predicted_tv_lr.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted_tv_lr.limit(1000))
(predicted_tv_lr.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,To evaluate the prediction result R2 (Coefficient of Determination) and RMSE of the model with the test data using testvalidationsplit:
 
#lr_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
#lr_evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
#print("RMSE: %f" % lr_evaluator_rmse.evaluate(prediction_tv_lr))
#print("R2: %f" % lr_evaluator_r2.evaluate(prediction_tv_lr))



from pyspark.ml.evaluation import RegressionEvaluator
lr_tv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
 
lr_tv_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 

print("r2: %f" % lr_tv_evaluator_r2.evaluate(prediction_tv_lr))
print("RMSE: %f" % lr_tv_evaluator_rmse.evaluate(prediction_tv_lr))


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,GBTRegressor
gbt_assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame for GBT
training = gbt_assembler.transform(train)
#training.show()



# COMMAND ----------

# DBTITLE 1,sets up a model with GBTRegressor
gbt = GBTRegressor(labelCol="label") #,featuresCol="features",maxIter=10

# COMMAND ----------

# DBTITLE 1,create a parameter combination, which is to tune the model.
paramGrid = ParamGridBuilder()\
.addGrid(gbt.maxDepth, [2, 5])\
.addGrid(gbt.maxIter, [10, 20])\
.addGrid(gbt.minInfoGain, [0,0])\
.build()

# COMMAND ----------

# DBTITLE 1,create a evaluator to evaluate a model with R2
from pyspark.ml.evaluation import RegressionEvaluator
gbt_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")

# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
pipeline = Pipeline(stages=[gbt_assembler, gbt]) 

# COMMAND ----------

# DBTITLE 1,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv = CrossValidator(estimator=pipeline, evaluator=gbt_evaluator, estimatorParamMaps=paramGrid, numFolds = 3)

# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of cross validator
model = cv.fit(train) 

# COMMAND ----------

# DBTITLE 1,calculate the feature importance of GBT algorithm
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "label") 
predicted.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted.limit(1000))
(predicted.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,GBT R2 & RMSE For Cross Validation
print("R Squared (R2) on test data = %g" % gbt_evaluator.evaluate(prediction))
 
gbt_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % gbt_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,GBT RMSE For Cross Validation
from pyspark.ml.evaluation import RegressionEvaluator
 
gbt_evaluator1 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("RMSE: %f" % gbt_evaluator1.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,use TrainValidationSplit instead of CrossValidator
cv1 = TrainValidationSplit(estimator=pipeline, evaluator=gbt_evaluator1, estimatorParamMaps=paramGrid, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model1 = cv1.fit(train)

# COMMAND ----------

# DBTITLE 1,calculate the feature importance of GBT algorithm using TrainValidationSplit
prediction = model1.transform(test)
predicted = prediction.select("features", "prediction", "label") 
predicted.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted.limit(1000))
#(predicted.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,GBT Train Validation Split RMSE & R2
print("R Squared (R2) on test data = %g" % gbt_evaluator1.evaluate(prediction))
 
gbt_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % gbt_evaluator.evaluate(prediction))


##You observe that the coefficient, R2, is  0.675 which is some what closer to 1, comparing to the linear Regression: RMSE for train validation split is 6728.73 In the previous model, we had : [R2: 0.436939, RMSE: 8837.227827]

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
 
gbt_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2: %f" % gbt_evaluator_r2.evaluate(prediction))

# COMMAND ----------

# And, you observe that the coefficient, R2, is 0.675 and RMSE is 6728.73 In the
# previous model, we have : [R2: 0.675, RMSE: 6728.73], which is same as in CrossValidator.
# TrainValidationSplit has the similar generalization to but much faster than CrossValidator

# COMMAND ----------

# DBTITLE 1,Random Forest Regressor
assembler_Rf = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame for Random Forest Regressor
training_Rf = assembler.transform(train)
#training.show()

# COMMAND ----------

# DBTITLE 1,sets up a model with Random Forest Regressor
rf = RandomForestRegressor(labelCol='label') #,featuresCol="features", numTrees=10, maxDepth=5

# COMMAND ----------

# DBTITLE 1,create a parameter combination, which is to tune the model.
paramGrid_Rf = ParamGridBuilder()\
.addGrid(rf.maxDepth, [2, 3])\
.addGrid(rf.minInfoGain, [0.0])\
.build()

# COMMAND ----------

# DBTITLE 1,create a evaluator to evaluate a model with R2
from pyspark.ml.evaluation import RegressionEvaluator
rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")

# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
pipeline = Pipeline(stages=[assembler_Rf, rf])

# COMMAND ----------

# DBTITLE 1,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv_rf = CrossValidator(estimator=pipeline, evaluator=rf_evaluator, estimatorParamMaps=paramGrid_Rf)

# COMMAND ----------

# DBTITLE 1, Tune the model for the generalization. 
model_rf = cv_rf.fit(train)

# COMMAND ----------

# DBTITLE 1,Predict price with actual price using RF
prediction_rf = model_rf.transform(test)
predicted_rf = prediction_rf.select("features", "prediction", "label") 
predicted_rf.show(10)

# COMMAND ----------

#display(prediction_rf.limit(1000))
#(prediction_rf.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,Random Forest Cross Validation R2 and RMSE
print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(prediction_rf))
 
rf_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % rf_evaluator.evaluate(prediction_rf))

# COMMAND ----------

# DBTITLE 1,R2 Evaluation
from pyspark.ml.evaluation import RegressionEvaluator
 
rf_evaluator1 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")

print("R2: %f" % rf_evaluator1.evaluate(prediction_rf))


# COMMAND ----------

# DBTITLE 1,use TrainValidationSplit instead of CrossValidator
Tv_rf = TrainValidationSplit(estimator=pipeline, evaluator=rf_evaluator1, estimatorParamMaps=paramGrid_Rf, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model_tv_rf = Tv_rf.fit(train)

# COMMAND ----------

# DBTITLE 1,comparing the predicted results with actual results
prediction_tv_rf = model_tv_rf.transform(test)
predicted_tv_rf = prediction_tv_rf.select("features", "prediction", "label") 
predicted_tv_rf.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(prediction_tv_rf.limit(1000))

# COMMAND ----------

# DBTITLE 1,Random Forest RMSE
 rf_tv_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % rf_tv_evaluator.evaluate(prediction_tv_rf))

# COMMAND ----------

# DBTITLE 1,R2 Evaluation for Random Forest train validation split
from pyspark.ml.evaluation import RegressionEvaluator
 
rf_tv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")

print("R2: %f" % rf_tv_evaluator_r2.evaluate(prediction_tv_rf))

# COMMAND ----------

model = pipeline.fit(train)

# COMMAND ----------

pipeline = Pipeline(stages=[assembler_Rf, rf])

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
rf = RandomForestRegressor(labelCol="label", featuresCol="features", numTrees=10, maxDepth=5)

# combine stages into pipeline
pipeline = Pipeline(stages= [assembler_Rf, rf])
model = pipeline.fit(train)

# COMMAND ----------

#display(test)

# COMMAND ----------

# DBTITLE 1,Testing data frame on the model that we built. Then, the model predicts the value “prediction” that the price of used cars counts. It can be compared to “price” in the target column, the actual price of the used cars.
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "label")
predicted.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted.limit(1000))

# COMMAND ----------

# DBTITLE 1,Evaluate the prediction result of the model with the test data:
from pyspark.ml.evaluation import RegressionEvaluator
rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                   labelCol="label",metricName="r2")

print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(prediction))

rf_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
print("RMSE: %f" % rf_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,ML 4:  Factorization Machine Regression
assembler_fm = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Create a training data frame for GBT
training_fm = assembler_fm.transform(train).select(col("features"))
#training.show()

# COMMAND ----------

# DBTITLE 1,sets up a model with FM Regressor
fm = FMRegressor(labelCol='label') #,featuresCol="features", numTrees=10, maxDepth=5

# COMMAND ----------

paramGrid_fm = ParamGridBuilder()\
  .addGrid(fm.factorSize, [2, 3]) \
  .addGrid(fm.stepSize, [5, 10]) \
  .addGrid(fm.regParam, [0.0]) \
.build()


# paramGridCV = ParamGridBuilder() \
#   .addGrid(fm.factorSize, [2, 3]) \
#   .addGrid(fm.stepSize, [5, 10]) \
#   .addGrid(fm.regParam, [0.0]) \
#   .build()

# COMMAND ----------

# DBTITLE 1,create a evaluator to evaluate a model with R2
from pyspark.ml.evaluation import RegressionEvaluator
fm_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")

# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
pipeline_fm = Pipeline(stages=[assembler_fm, fm]) 

# COMMAND ----------

# DBTITLE 1,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv_fm = CrossValidator(estimator=pipeline_fm, evaluator=fm_evaluator, estimatorParamMaps=paramGrid_fm)

# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of cross validator
model_fm = cv_fm.fit(train) 

# COMMAND ----------

# DBTITLE 1,Predict price with actual price using Lr
prediction_fm = model_fm.transform(test)
predicted_fm = prediction_fm.select("features", "prediction", "label") 
predicted_fm.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted_fm.limit(1000))

# COMMAND ----------

# DBTITLE 1,FM Cross Validator RMSE
#print("R Squared (R2) on test data = %g" % fm_evaluator.evaluate(prediction_fm))
 
fm_cv_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % fm_cv_evaluator_rmse.evaluate(prediction_fm))

# COMMAND ----------

# DBTITLE 1,R2
from pyspark.ml.evaluation import RegressionEvaluator
 
fm_cv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")

print("R2: %f" % fm_cv_evaluator_r2.evaluate(prediction_fm))

# COMMAND ----------

# DBTITLE 1,use TrainValidationSplit instead of CrossValidator
fm_tv = TrainValidationSplit(estimator=pipeline_fm, evaluator=fm_evaluator, estimatorParamMaps=paramGrid_fm, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model_fm = fm_tv.fit(train)

# COMMAND ----------

# DBTITLE 1,comparing the predicted results with actual results
prediction_fm = model_fm.transform(test)
predicted_fm = prediction_fm.select("features", "prediction", "label") 
predicted_fm.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: label and Prediction.
#display(predicted_fm.limit(1000))

# COMMAND ----------

# DBTITLE 1,To evaluate the prediction result R2 (Coefficient of Determination) and RMSE of the FM Train Validation Split
print("R Squared (R2) on test data = %g" % fm_evaluator.evaluate(prediction_fm))
 
fm_tv_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE: %f" % fm_tv_evaluator.evaluate(prediction_fm))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,ML5: Decision Tree Classifier
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
(df2.select("year", "city_fuel_economy", col("price").alias("label"), "mileage", "owner_count", "latitude", "highway_fuel_economy").limit(10)).show()


# COMMAND ----------

# DBTITLE 1,Data Conversion and Data Filtering
df2 = df2.filter(col("price") < 100000)
#display(df2.limit(10))
(df2.limit(10)).show()
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

#cv_fm = CrossValidator(estimator=pipeline_fm, evaluator=fm_evaluator, estimatorParamMaps=paramGrid_fm)
cv_decision = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid)

model_Cv_decision = cv_decision.fit(train)

# COMMAND ----------

prediction_cv_decision = model_Cv_decision.transform(test)
predicted_cv_decisison = prediction_cv_decision.select("features", "prediction", "probability", "trueLabel")

predicted_cv_decisison.show(100, truncate=False)

# COMMAND ----------

tp = float(predicted_cv_decisison.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted_cv_decisison.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted_cv_decisison.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted_cv_decisison.filter("prediction == 0.0 AND truelabel == 1").count())
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

prediction_cv_decision.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate=False)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_cv = evaluator.evaluate(prediction_cv_decision)
print("AUC = ", auc_cv)

# COMMAND ----------

# DBTITLE 1, TrainValidationSplit

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

# COMMAND ----------


