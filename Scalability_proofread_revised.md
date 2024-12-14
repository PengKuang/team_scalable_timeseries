## Scalability

The primary objective of this project is to utilize an ensemble approach when the dataset exceeds the capacity of a single machine. Beyond handling large-scale data, adopting an ensemble model offers several advantages. Notably, it enables the application of a distributed learning framework, where worker nodes compute simple metrics (resulting in low transfer costs) and transmit them to the master node. This approach ensures data privacy by keeping sensitive information localized.

We begin by outlining the ensemble approach. Consider a dataset $D$ with size $S_D$, and assume that a single node can feasibly train a model on data of size $S_1$ within a reasonable timeframe. If $S_D \leq S_1 $, there is no need to employ an ensemble, and a single model can be trained on the entire dataset. However, if $S_D > S_1 $, we partition the dataset as $ D = \bigcup_{i=1}^N D_i $, dividing it into chunks $D_i$ such that $|D_i| \leq S_1$, each of which fits within the capacity of a single node. Here, $N$ denotes the total number of nodes used.


The process proceeds by letting 
1. Training
    1. Node $i$ receives data $D_i$, $i=1,\dots,N$
    2. Node $i$ defines a model $m_i$, $i=1,\dots,N$
    3. Model $m_i$ is trained on $D_i$ until some criteria is met, $i=1,\dots,N$
2. Inference
    1. The master node is fed a set (possibly single element) $D_{test}$ to perform TASK (Anomaly detection)
    2. Assuming $|D_{test}| < S_1 $, we send the whole set to each node $i$, $i=1,\dots,N$
    3. Node $i$ evalutes and returns $f_{AD}(m_i(D_{test}))$ to master node, $i=1,\dots,N$, ($AD$ = Anomaly detection)
    4. Master node evaluates the mean (or anything else)
       $$
       Output = \frac{1}{N}\sum_{i=1}^{N} f_{AD}(m_i(D_{test}))
       $$
   
(Note that the output can either be a single value representing an inference for the entire dataset or be evaluated pointwise, providing individual inference values for each data point.)

This approach works for any type of data partitioning, constituting a type of distributed deep learning called Distributed Data Parallel (DDP). In our application with ECG data, we assume the data is managed by a centralized healthcare system, for instance, www.1177.se. The dataset contains the nation's patient records of ten years. It is too big for a single node or cluster at 1177 to train. Although vertical scaling at 1177 is possible, it can still take too long to complete the training or simply cost too much to upgrade the infrastructure. Therefore, the better option is that 1177 acts as a driver to distribute the training job to the hospitals that already have some available, existing infrastructure. Further, 1177 can easily prescribe and actualize that each hospital trains the model with its own ECG data. That is, the hospital can only access the ECG data originally submitted to 1177 by itself. This approach preserves patients' privacy. Meanwhile, hospitals benefit from sharing a model that is well-trained with all the data.

We would like to train a copy of the shared model at each hospital with its own data. Implemented by TorchDistributor, the gradients from each node will be averaged and synchronized through interprocess communications, and then used to update the local model replica. This enables each worker node (hospital) to process new data to evaluate anomalies locally during the inference stage. It again enforces privacy preservation.

In this report, we focus on time series data from a single ECG dataset. To develop a plausible model, we implement an ensemble approach on a local machine. Instead of assigning a GPU node to each model and its corresponding data partition, we utilize a single machine by distributing the workload across CPU cores. For this purpose, we use the PySpark TorchDistributor framework, which efficiently distributes tasks across multiple cores. While designed to support GPUs and multi-GPU setups at each node, this framework is also adaptable for parallel processing on CPU cores.

Below we describe how we implement TorchDistributor and how we use it:

More information can be found [databricks.com/TorchDistributor](https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html), and from the links within. Below we briefly explain the main structure, mostly cited from the databricks link.

The main functionality can be summarized in this code block.

```
from pyspark.ml.torch.distributor import TorchDistributor

result = TorchDistributor(
    num_processes= N,     # Number of "nodes" we use (actual tasks)
    local_mode=True,      # Determines if the master node is doing work or not
    use_gpu=False         # Use GPU or not
        ).run(
            <function>,   # The function call we want to distribute to each node
            <args>     )  # The arguments that should be passed to each node
```

In our setting, we will let the function, "training", be a function that trains one model with one partitioned data set. <args> will contain training instructions, such as epochs to run, learning rate and other hyperparameters we might want to tune. 

The data is partitioned in the preprocessing and each node is assigned one partition. The models are either created at this stage or initialized in the "training" function.

Running the ensemble on our own computer (laptop without GPUs `use_gpu = False`) , we see that we get best scaling by setting $N$ as the number of CPU cores. We set `local_mode = True`, otherwise the master node (i.e. the only node) will not work.

In our main training function (where each node is working) we make use `local_rank = int(os.environ["LOCAL_RANK"])`. This retrieves which node that is working and in a very simple way we can collect the correct partition and model. Similarly, the model parameters are saved based on their `local_rank` to know which node it belongs to.
