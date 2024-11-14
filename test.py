import torch
import pyspark

# Check PyTorch installation
x = torch.rand(5, 3)
print(x)

# Initialize Spark session
def __main__():
    spark = pyspark.sql.SparkSession.builder.appName("example").getOrCreate()
    df = spark.createDataFrame([(1, 'a'), (2, 'b')], ["id", "value"])
    df.show()