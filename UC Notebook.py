# Databricks notebook source
# MAGIC %sql
# MAGIC select current_catalog()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS aamp_dev

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog aamp_dev

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema db_work

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists aamp_dev.db_work.stores(id int);
# MAGIC insert into aamp_dev.db_work.stores values (1);

# COMMAND ----------

# MAGIC %sql
# MAGIC describe extended aamp_dev.db_work.stores

# COMMAND ----------

# MAGIC %sql
# MAGIC describe aamp_dev.db_work.stores
