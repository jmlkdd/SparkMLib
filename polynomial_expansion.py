from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors

sc = SparkContext('spark://node:7077', 'PCA')
spark = SparkSession(sc)

df = spark.createDataFrame([
    (Vectors.dense([2.0, 1.0]),),
    (Vectors.dense([0.0, 0.0]),),
    (Vectors.dense([3.0, -1.0]),)
], ['features'])

poly_expansion = PolynomialExpansion(degree=3, inputCol='features', outputCol='polyFeatures')

poly_df = poly_expansion.transform(df)

poly_df.show(truncate=False)
