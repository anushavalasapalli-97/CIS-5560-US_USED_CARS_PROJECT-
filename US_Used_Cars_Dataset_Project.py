# Databricks notebook source
IS_SPARK_SUBMIT_CLI = False
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

spark.sparkContext.setLogLevel("WARN")

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/USA_USED_CARS_DATASET_FOR_PREDICTING_CAR_PRICE.csv"
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


temp_table_name = "USA_USED_CARS_DATASET_FOR_PREDICTING_CAR_PRICE_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
   df = spark.read.csv('USA_USED_CARS_DATASET_FOR_PREDICTING_CAR_PRICE.csv', inferSchema=True, header=True)
else:
    df = spark.sql("SELECT * FROM USA_USED_CARS_DATASET_FOR_PREDICTING_CAR_PRICE_csv")


df.show(5)

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
from pyspark.sql.functions import when, col

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder
import pandas as pd



# COMMAND ----------

# DBTITLE 1,Typecast string to int or float
df2 = df.withColumn("year",col("year").cast("int")).withColumn("city_fuel_economy",col("city_fuel_economy").cast("float")).withColumn("price",col("price").cast("float")).withColumn("mileage",col("mileage").cast("float")).withColumn("owner_count",col("owner_count").cast("int")).withColumn("latitude",col("latitude").cast("float")).withColumn("highway_fuel_economy",col("highway_fuel_economy").cast("float")).withColumn("daysonmarket",col("daysonmarket").cast("int")).withColumn("dealer_zip",col("dealer_zip").cast("int")).withColumn("engine_displacement",col("engine_displacement").cast("int")).withColumn("listing_id",col("listing_id").cast("int")).withColumn("savings_amount",col("savings_amount").cast("int")).withColumn("sp_id",col("sp_id").cast("int")).withColumn("seller_rating",col("seller_rating").cast("int"))


# COMMAND ----------

(df2.select("year", "city_fuel_economy", "price", "mileage", "owner_count", "latitude", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating", "sp_id","frame_damaged","theft_title", "has_accidents", "frame_damaged", "theft_title").limit(10)).show()

# COMMAND ----------

# DBTITLE 1,Convert dataset to pandas
df3 = df2.toPandas()

# COMMAND ----------

data1=df2

# COMMAND ----------

# DBTITLE 1,Remove the null values and filter the price to more than 100,000 
df2.na.drop(subset=["price", "year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating", "sp_id", "has_accidents", "frame_damaged", "theft_title"])
df222 = df2.filter(col("price") < 100000)
df22 = df222.select("year", "city_fuel_economy", col("price").alias("label"), "mileage", "owner_count", "latitude", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating", "sp_id", "has_accidents", "frame_damaged", "theft_title")
df22.show()



# COMMAND ----------

df22.show()

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
xlr_assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy", "engine_displacement", "savings_amount"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 1,Train a Regression Model
lr= LinearRegression(labelCol="label") #,featuresCol="features",maxIter=10

# COMMAND ----------

# DBTITLE 1,create a parameter combination, which is to tune the model.
paramGrid1_CV_LR = ParamGridBuilder()\
.addGrid(lr.maxIter, [10, 30])\
.addGrid(lr.regParam, [0, 0.1])\
.build()

# COMMAND ----------

# DBTITLE 1,create a evaluator to evaluate a model with R2
from pyspark.ml.evaluation import RegressionEvaluator
lr_CV_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")


# COMMAND ----------

# DBTITLE 1,create a pipeline, which is to sequence the tasks.
#we can use same pipeline_lr for both cv and tvs
pipeline_lr = Pipeline(stages=[lr_assembler, lr]) 

# COMMAND ----------

# DBTITLE 1,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv_lr = CrossValidator(estimator=pipeline_lr, evaluator=lr_CV_evaluator, estimatorParamMaps=paramGrid1_CV_LR)

# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of cross validator
model_CV_LR = cv_lr.fit(train) 

# COMMAND ----------

# DBTITLE 1,Predict price with actual price using Lr
prediction_CV_lR = model_CV_LR.transform(test)
predicted_CV_lR = prediction_CV_lR.select("features", "prediction", "label") 
predicted_CV_lR.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted_lr.limit(1000))
(predicted_CV_lR.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,R2 and RMSE For LR_CV
#print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(prediction_lr))

from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator_CV_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
 
lr_evaluator_CV_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
#R2 and RMSE For LR_CV
print("r2 For LR_CV: %.2f" % lr_evaluator_CV_r2.evaluate(prediction_CV_lR))
print("RMSE For LR_CV: %.2f" % lr_evaluator_CV_rmse.evaluate(prediction_CV_lR))

# COMMAND ----------

# DBTITLE 1,TRAIN VALIDATION LINERA REGRESSION
paramGrid2_TV_LR = ParamGridBuilder()\
.addGrid(lr.maxIter, [10, 40])\
.addGrid(lr.regParam, [0.1, 1.0])\
.build()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
lr_tv_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")

# COMMAND ----------

tv_lr = TrainValidationSplit(estimator=pipeline_lr, evaluator=lr_tv_evaluator, estimatorParamMaps=paramGrid2_TV_LR, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Data frame 
model1_tv_lr = tv_lr.fit(train)

# COMMAND ----------

# DBTITLE 1,comparing the predicted results with actual results
prediction_tv_lr = model1_tv_lr.transform(test)
predicted_tv_lr = prediction_tv_lr.select("features", "prediction", "label") 
predicted_tv_lr.show(10)

# COMMAND ----------

# DBTITLE 1,Display of Scatter Plot
#display(predicted_tv_lr.limit(1000))
(predicted_tv_lr.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,R2 and RMSE For LR_TV

from pyspark.ml.evaluation import RegressionEvaluator
lr_tv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
 
lr_tv_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
#R2 and RMSE For LR_TV
print("r2 For LR CV: %.2f" % lr_tv_evaluator_r2.evaluate(prediction_tv_lr))
print("RMSE for LT TV: %.2f" % lr_tv_evaluator_rmse.evaluate(prediction_tv_lr))


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,GBTRegressor_CV
gbt_assembler = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy", "engine_displacement", "savings_amount"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 0,sets up a model with GBTRegressor
gbt = GBTRegressor(labelCol="label") #,featuresCol="features",maxIter=10

# COMMAND ----------

# DBTITLE 0,create a parameter combination, which is to tune the model.
paramGrid_CV_GBT = ParamGridBuilder()\
.addGrid(gbt.maxDepth, [2, 5])\
.addGrid(gbt.maxIter, [10, 20])\
.addGrid(gbt.minInfoGain, [0,0])\
.build()

# COMMAND ----------

# DBTITLE 0,create a evaluator to evaluate a model with R2
gbt_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")

# COMMAND ----------

# DBTITLE 0,create a pipeline, which is to sequence the tasks.
pipeline_GBT = Pipeline(stages=[gbt_assembler, gbt]) 

# COMMAND ----------

# DBTITLE 0,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
GBT_CV = CrossValidator(estimator=pipeline_GBT, evaluator=gbt_evaluator, estimatorParamMaps=paramGrid_CV_GBT, numFolds = 3)

# COMMAND ----------

model_GBT = GBT_CV.fit(train) 

# COMMAND ----------

# DBTITLE 0,calculate the feature importance of GBT algorithm
prediction_GBT = model_GBT.transform(test)
predicted_GBT = prediction_GBT.select("features", "prediction", "label") 
predicted_GBT.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted.limit(1000))
(predicted_GBT.limit(1000)).show()

# COMMAND ----------

# DBTITLE 1,GBT R2 & RMSE For Cross Validation
gbt_evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE_GBT_CV: %.2f" % gbt_evaluator_r2.evaluate(prediction_GBT))

gbt_evaluator_rmse = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2_GBT_CV: %.2f" % gbt_evaluator_rmse.evaluate(prediction_GBT))


# COMMAND ----------

# DBTITLE 1,GBT Regressor_TV
paramGrid_TV_GBT = ParamGridBuilder()\
.addGrid(gbt.maxDepth, [2, 5])\
.addGrid(gbt.maxIter, [20, 40])\
.addGrid(gbt.maxBins, [15, 30])\
.addGrid(gbt.stepSize, [0.05, 0.1])\
.addGrid(gbt.minInfoGain, [0,0])\
.build()

# COMMAND ----------

GBT_TV = TrainValidationSplit(estimator=pipeline_GBT, evaluator=gbt_evaluator, estimatorParamMaps=paramGrid_TV_GBT, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model_GBT_TV = GBT_TV.fit(train)

# COMMAND ----------

# DBTITLE 1,calculate the feature importance of GBT algorithm using TrainValidationSplit
prediction_GBT = model_GBT_TV.transform(test)
predicted_GBT = prediction_GBT.select("features", "prediction", "label") 
predicted_GBT.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted.limit(1000))
predicted_GBT.limit(1000).show()

# COMMAND ----------

# DBTITLE 1,GBT Train Validation Split RMSE & R2
gbt_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE_GBT_TV: %.2f" % gbt_evaluator_rmse.evaluate(prediction_GBT))

 
gbt_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2_GBT_TV: %.2f" % gbt_evaluator_r2.evaluate(prediction_GBT))


# COMMAND ----------

# DBTITLE 1,Random Forest Regressor_CV
assembler_Rf = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy", "engine_displacement", "savings_amount"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 0,Create a training data frame for Random Forest Regressor
training_Rf = assembler_Rf.transform(train)
#training.show()

# COMMAND ----------

# DBTITLE 0,sets up a model with Random Forest Regressor
rf = RandomForestRegressor(labelCol='label') #,featuresCol="features", numTrees=10, maxDepth=5

# COMMAND ----------

# DBTITLE 0,create a parameter combination, which is to tune the model.
paramGrid_CV_Rf = ParamGridBuilder()\
    .addGrid(rf.maxDepth, [5, 10, 15])\
    .addGrid(rf.numTrees, [20, 50, 100])\
    .addGrid(rf.minInfoGain, [0.0])\
    .build()


# COMMAND ----------

# DBTITLE 0,create a evaluator to evaluate a model with R2
rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
labelCol="label",metricName="r2")

# COMMAND ----------

# DBTITLE 0,create a pipeline, which is to sequence the tasks.
pipeline_rf = Pipeline(stages=[assembler_Rf, rf])

# COMMAND ----------

# DBTITLE 0,create a cross validator, which is to tune the model for the generalization. The default folds are 3.
cv_rf = CrossValidator(estimator=pipeline_rf, evaluator=rf_evaluator, estimatorParamMaps=paramGrid_CV_Rf)

# COMMAND ----------

# DBTITLE 0, Tune the model for the generalization. 
model_CV_rf = cv_rf.fit(train)

# COMMAND ----------

# DBTITLE 0,Predict price with actual price using RF
prediction_CV_rf = model_CV_rf.transform(test)
predicted_CV_rf = prediction_CV_rf.select("features", "prediction", "label") 
predicted_CV_rf.show(10)

# COMMAND ----------

#display(prediction_rf.limit(1000))
prediction_CV_rf.limit(1000).show()

# COMMAND ----------

# DBTITLE 0,Random Forest Cross Validation R2 and RMSE
rf_evaluator_CV_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE for CV_RF: %.2f" % rf_evaluator_CV_rmse.evaluate(prediction_CV_rf))

 
rf_evaluator_CV_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2 for CV_RF: %.2f" % rf_evaluator_CV_r2.evaluate(prediction_CV_rf))

# COMMAND ----------

# DBTITLE 1,Random Forest Train Validation Split
paramGrid_TV_Rf = ParamGridBuilder()\
    .addGrid(rf.maxDepth, [5, 10, 15])\
    .addGrid(rf.numTrees, [20, 50, 100])\
    .addGrid(rf.featureSubsetStrategy, ['auto', 'sqrt', 'log2'])\
    .addGrid(rf.minInfoGain, [0.0])\
    .build()


# COMMAND ----------

# DBTITLE 0,use TrainValidationSplit instead of CrossValidator
Tv_rf = TrainValidationSplit(estimator=pipeline_rf, evaluator=rf_evaluator, estimatorParamMaps=paramGrid_TV_Rf, trainRatio=0.9)


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
prediction_tv_rf.limit(1000).show()

# COMMAND ----------

# DBTITLE 0,Random Forest RMSE
rf_tv_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE for RF_TV: %.2f" % rf_tv_evaluator_rmse.evaluate(prediction_tv_rf))

 
rf_tv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2 for RF_TV: %.2f" % rf_tv_evaluator_r2.evaluate(prediction_tv_rf))

# COMMAND ----------

model_RF = pipeline_rf.fit(train)

# COMMAND ----------

# DBTITLE 0, Build a model with the train Dataframe 
rfModel = model_RF.stages[-1]
#print(rfModel.toDebugString)

# COMMAND ----------

# DBTITLE 0,show the importance of the features – columns – in the order of importance. NOTE: this code uses pandas library in standard Python not in PySpark.
featureImp = pd.DataFrame(list(zip(assembler_Rf.getInputCols(), rfModel.featureImportances)), columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending="False")

# COMMAND ----------

# DBTITLE 0,Build an RF model using pipeline with the paramters (numTrees=10, maxDepth=5) and train data set.
rf = RandomForestRegressor(labelCol="label", featuresCol="features", numTrees=30, maxDepth=15)

# combine stages into pipeline
pipeline_RF_1 = Pipeline(stages= [assembler_Rf, rf])
model_rf = pipeline_RF_1.fit(train)

# COMMAND ----------

# DBTITLE 0,Testing data frame on the model that we built. Then, the model predicts the value “prediction” that the price of used cars counts. It can be compared to “price” in the target column, the actual price of the used cars.
prediction_rf1 = model_rf.transform(test)
predicted_rf1 = prediction_rf1.select("features", "prediction", "label")
predicted_rf1.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: price and Prediction.
#display(predicted.limit(1000))
predicted_rf1.limit(1000).show()

# COMMAND ----------

# DBTITLE 0,Evaluate the prediction result of the model with the test data:
rf_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE for only RF: %.2f" % rf_evaluator_rmse.evaluate(prediction_rf1))

 
rf_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2 for only RF: %.2f" % rf_evaluator_r2.evaluate(prediction_rf1))


# COMMAND ----------

# DBTITLE 1,ML 4:  Factorization Machine Regression_CV
assembler_fm = VectorAssembler(inputCols = ["year", "city_fuel_economy", "mileage", "owner_count", "highway_fuel_economy", "engine_displacement", "savings_amount"], outputCol="features").setHandleInvalid("skip")

# COMMAND ----------

# DBTITLE 0,Create a training data frame for GBT
training_fm = assembler_fm.transform(train).select(col("features"))
#training.show()

# COMMAND ----------

# DBTITLE 0,sets up a model with FM Regressor
fm = FMRegressor(labelCol='label') #,featuresCol="features", numTrees=10, maxDepth=5

# COMMAND ----------

# paramGrid_CV_fm = ParamGridBuilder()\
#   .addGrid(fm.factorSize, [4, 7]) \
#   .addGrid(fm.stepSize, [50, 90]) \
#   .addGrid(fm.regParam, [0.0]) \
# .build()


paramGrid_CV_fm = ParamGridBuilder() \
  .addGrid(fm.stepSize, [0.1, 1]) \
  .build()

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
cv_fm = CrossValidator(estimator=pipeline_fm, evaluator=fm_evaluator, estimatorParamMaps=paramGrid_CV_fm)

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

# DBTITLE 1,FM Cross Validator R2 and  RMSE
fm_cv_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE for FM_CV: %.2f" % fm_cv_evaluator_rmse.evaluate(prediction_fm))

fm_cv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2 for FM_CV: %.2f" % fm_cv_evaluator_r2.evaluate(prediction_fm))


# COMMAND ----------

# DBTITLE 1, FM_TrainValidationSplit 
paramGrid_TV_fm = ParamGridBuilder()\
  .addGrid(fm.factorSize, [8]) \
  .addGrid(fm.stepSize, [0.01, 0.1]) \
  .addGrid(fm.regParam, [0.01, 0.1]) \
.build()

# COMMAND ----------



# COMMAND ----------

fm_tv = TrainValidationSplit(estimator=pipeline_fm, evaluator=fm_evaluator, estimatorParamMaps=paramGrid_TV_fm, trainRatio=0.8)


# COMMAND ----------

# DBTITLE 1, Build a model with the train Dataframe of Train Validation Split
model_tv_fm = fm_tv.fit(train)

# COMMAND ----------

# DBTITLE 1,comparing the predicted results with actual results
prediction_tv_fm = model_tv_fm.transform(test)
predicted_tv_fm = prediction_tv_fm.select("features", "prediction", "label") 
predicted_tv_fm.show(10)

# COMMAND ----------

# DBTITLE 1,To compare the columns in scatter chart: label and Prediction.
display(predicted_tv_fm.limit(1000))

# COMMAND ----------

# DBTITLE 1,To evaluate the prediction result R2 (Coefficient of Determination) and RMSE of the FM Train Validation Split
fm_tv_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse") 
print("RMSE for FM_TV: %.2f" % fm_tv_evaluator_rmse.evaluate(prediction_tv_fm))

fm_tv_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", \
                labelCol="label",metricName="r2")
print("R2 for FM_CV: %.2f" % fm_tv_evaluator_r2.evaluate(prediction_tv_fm))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,ML5: Logistic Regression
flightSchema = StructType([
  StructField("year", IntegerType(), False),
  StructField("price", IntegerType(), False),
  StructField("mileage", StringType(), False),
  StructField("owner_count", IntegerType(), False),
  StructField("seller_rating", IntegerType(), False),
  StructField("city_fuel_economy", IntegerType(), False),
  StructField("highway_fuel_economy", IntegerType(), False),
  StructField("daysonmarket", IntegerType(), False),
  StructField("dealer_zip", IntegerType(), False),
  StructField("engine_displacement", IntegerType(), False),
  StructField("listing_id", IntegerType(), False),
  StructField("savings_amount", IntegerType(), False),
  StructField("seller_rating", IntegerType(), False),
  StructField("sp_id", IntegerType(), False),
    
])

# COMMAND ----------

# DBTITLE 1,To display the selected columns
(df2.select("year", "city_fuel_economy", col("price").alias("label"), "mileage", "owner_count", "latitude", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating","sp_id").limit(10)).show()


# COMMAND ----------

# DBTITLE 1,Data Conversion and Data Filtering
df2 = df2.filter(col("price") < 100000)
#display(df2.limit(10))
(df2.limit(10)).show()
df23 = df2.select("year", "city_fuel_economy", ((col("price") > 10000).cast("Double").alias("label")), "mileage", "owner_count", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating","sp_id")


# COMMAND ----------

# DBTITLE 1,Drop the Unused Columns
df23.na.drop(subset=["label", "year", "city_fuel_economy", "mileage","owner_count", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating","sp_id"])
df23.show()


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
catVect = VectorAssembler(inputCols = ["mileage","year", "owner_count", "city_fuel_economy", "highway_fuel_economy", "daysonmarket", "dealer_zip", "engine_displacement", "listing_id", "savings_amount", "seller_rating","sp_id"], outputCol="catFeatures").setHandleInvalid("skip")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures").setHandleInvalid("skip")

# COMMAND ----------

# number is meaningful so that it should be number features
numVect = VectorAssembler(inputCols = ["mileage"], outputCol="numFeatures")
# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")  #="features1")
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
cv_logr = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid)
model_Cv_logr = cv_decision.fit(train)

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
print("AUC = ","{:.2f}".format(auc))

# COMMAND ----------


