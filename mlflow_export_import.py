# Databricks notebook source
# DBTITLE 1,Create Model
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
mlflow.set_experiment("/Workspace/Users/databricksuc@kjaswanth4gmail.onmicrosoft.com/test_model")
with mlflow.start_run():
    mlflow.spark.log_model(lr_model, "test_model")

# COMMAND ----------

# DBTITLE 1,Register in Workspace Registry
# Register the model in the MLflow Model Registry
model_name = "test_model"
model_uri = "runs:/32a8328445aa45c9a6f3662eec44abcb/test_model"
mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# DBTITLE 1,Load the model/Add Signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks")

input_schema = Schema([ColSpec("string", "feature")])
output_schema = Schema([ColSpec("string", "label")])
# Create the model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)


model_name = "test_model"
model_uri = "models:/test_model/1"

loaded_model = mlflow.spark.load_model(model_uri)


parent_run_id = None
with mlflow.start_run() as parent_run:
    mlflow.spark.log_model(artifact_path="model", spark_model=loaded_model, signature=signature)
    
    parent_run_id = parent_run.info.run_id

# COMMAND ----------

tracking_uri = mlflow.get_tracking_uri()
display(tracking_uri)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
##PARENT RUN ID from ABOVE
mlflow.register_model(model_uri=f"runs:/{parent_run_id}/model",
                      name=f"main.default.{model_name}_uc"
                      )

# COMMAND ----------

from mlflow_export_import.model.export_model import export_model

model_name='main.default.test_model_uc'
vol='/Volumes/main/default/test_volume'

export_model(model_name,vol)

# COMMAND ----------

from mlflow_export_import.model.import_model import import_model

model_name='main.default.test_model_uc'
vol='/Volumes/main/default/test_volume'
experiment_name='/Users/databricksuc@kjaswanth4gmail.onmicrosoft.com/test_model_uc'

import_model(model_name=model_name,experiment_name=experiment_name,input_dir=vol)

# COMMAND ----------

# DBTITLE 1,RESOURCE_DOES_NOT_EXIST
from mlflow_export_import.model.export_model import export_model

model_name='main.default.random'
vol='/Volumes/main/default/test_volume'

export_model(model_name,vol)
