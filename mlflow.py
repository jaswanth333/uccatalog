from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient
import mlflow

# Define the input schema
# CHANGE THESE COLUMN NAMES AND DATA TYPES ACCORDINGLY
input_schema = Schema([
    ColSpec('long', "STORE_ID"),
    ColSpec('double', "SCHEDULED_LABOR"),
    ColSpec('long', "WEEK_ID"),
    ColSpec('long', "DAYOFWEEK"),
    ColSpec('long', "HOUR"),
    ColSpec('long', "MIN")
])

# Define the output schema
output_schema = Schema([
    ColSpec("double")
])

# Create the model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

model_version = 1
# Instead of model_version, you can also use model_stage as 'Production'
workspace_registry_uri = f'models://{scope}:{prefix}@databricks/{model_name}/{model_stage}'
loaded_model = mlflow.pyfunc.load_model(workspace_registry_uri)
inst=loaded_model.unwrap_python_model()


parent_run_id = None
with mlflow.start_run() as parent_run:
        # CHANGE THIS to your model falvor 
    mlflow.pyfunc.log_model(artifact_path = "model",python_model=inst,signature=signature)
    
    parent_run_id = parent_run.info.run_id

#2
mlflow.set_registry_uri("databricks-uc")
##PARENT RUN ID from ABOVE
mlflow.register_model(model_uri=f"runs:/{parent_run_id}/model",
                      name=f"feature_store.default.{model_name}_ws"
                      )

# Databricks notebook source
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import mlflow
import mlflow.spark

# Sample data
data = [(1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0), (5, 6.0)]
columns = ["feature", "label"]
df = spark.createDataFrame(data, columns)

# Prepare data for Linear Regression
assembler = VectorAssembler(inputCols=["feature"], outputCol="features")
assembled_df = assembler.transform(df)

# Train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(assembled_df)

# Log model in MLflow
mlflow.set_experiment("/Workspace/Users/databricksuc@kjaswanth4gmail.onmicrosoft.com/linear_regression_experiment")
with mlflow.start_run():
    mlflow.spark.log_model(lr_model, "linear_regression_model")

# COMMAND ----------

import mlflow
catalog = "main"
schema = "default"
model_name = "my_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/ad24fe1533ba4826915e85bf7285fa47/linear_regression_model", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

import mlflow.models.signature as signature
from mlflow.types.schema import Schema, ColSpec

# Define input and output schema
input_schema = Schema([ColSpec("double", "feature")])
output_schema = Schema([ColSpec("double")])

# Log model with signature in MLflow
mlflow.set_experiment("/Workspace/Users/databricksuc@kjaswanth4gmail.onmicrosoft.com/linear_regression_experiment_signatured")
with mlflow.start_run():
    mlflow.spark.log_model(lr_model, "linear_regression_model", signature=signature.ModelSignature(inputs=input_schema, outputs=output_schema))

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/databricks")

# COMMAND ----------

import mlflow
import os

# Check if the source artifact location exists
artifact_location = "dbfs:/databricks/mlflow-tracking/4031673903815799/ad24fe1533ba4826915e85bf7285fa47/artifacts/linear_regression_experiment_signatured"
if not os.path.exists(artifact_location):
    raise FileNotFoundError(f"Artifact location {artifact_location} does not exist.")

# Proceed with model registration
catalog = "main"
schema = "default"
model_name = "signatured_lr_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(f"runs:/ad24fe1533ba4826915e85bf7285fa47/linear_regression_experiment_signatured", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

import mlflow
catalog = "main"
schema = "default"
model_name = "signatured_lr_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/ad24fe1533ba4826915e85bf7285fa47/linear_regression_experiment_signatured", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# set the experiment id
mlflow.set_experiment(experiment_id="4031673903815802")

mlflow.autolog()
db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

# COMMAND ----------

import mlflow
catalog = "main"
schema = "default"
model_name = "my_model_signaute_run1"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri="runs:/75147035287f43eda6656e5fada0a9cd/model",
    name=f"{catalog}.{schema}.{model_name}"
)
