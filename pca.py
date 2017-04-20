import numpy as np
import pandas as pd
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import Matrix

sc = SparkContext('spark://node:7077', 'PCA')
spark = SparkSession(sc)

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df_data = pd.DataFrame(np.random.randn(20, 5), columns=list('ABCDE'))
data1 = [(Vectors.dense(df_data.iloc[i, :]),) for x in range(df_data.shape[0])]
df1 = spark.createDataFrame(data1, ["features"])
df = spark.createDataFrame(data, ["features"])

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

df1.rdd.map(lambda x: (Vectors.sparse(x.asDict().values()),)).collect()