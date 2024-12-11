from pyspark import SparkContext
import os

# Initialize SparkContext
sc = SparkContext("spark://localhost:7077", "Dataset Splitter")
sc.setLogLevel("ERROR")

# Example Tensor dataset
tensor_dataset = [i for i in range(100)]  # Replace this with your actual dataset
number_of_workers = 3  # Set according to SPARK_WORKER_INSTANCES

# Split the dataset into chunks based on the number of workers
rdd = sc.parallelize(tensor_dataset, numSlices=number_of_workers)

# Function to process each chunk and save the data to a file
def save_chunk(chunk_iterator):
    chunk = list(chunk_iterator)  # Convert iterator to list
    worker_id = os.getenv("SPARK_EXECUTOR_ID", "unknown")  # Identify worker ID
    output_dir = "/home/michele/aiuto"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"worker_{worker_id}_chunk.txt")
    with open(file_path, "w") as f:
        for item in chunk:
            f.write(f"{item}\n")  # Write each element of the chunk on a new line
    return chunk

# Use mapPartitions to apply the function to each partition
chunk_results = rdd.mapPartitions(save_chunk).collect()

# Stop the Spark context
sc.stop()
