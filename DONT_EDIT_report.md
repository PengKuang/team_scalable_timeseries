> Project by [Kasper Bågmark](https://research.chalmers.se/person/bagmark), [Michele Di Sabato](https://www.umu.se/en/staff/michele-di-sabato/), [Erik Jansson](https://www.chalmers.se/en/persons/erikjans/), [Peng Kuang](https://portal.research.lu.se/en/persons/peng-kuang) and [Selma Tabakovic](https://www.chalmers.se/en/persons/selmat/)

# Anomaly detection in time series using autoencoders 
The goal of this project is to take a scalable approach to anomaly detection in time series. 
More specifically, we consider electrocardiogram data, i.e., _time series_ of heart rate sequences. The challenge is to reliably detect if such a time series is _anomalous_, or in other words, deviates from healthy heartbeat patterns. 

## Background on time series

A time series is a sequence of real data points indexed by time, i.e., 
$
(x_t,t \in \mathbb{T}), 
$
where $\mathbb{T}$ is an index set, for instance $\mathbb{T} = \{1,2,3,\ldots\}$. Famous examples include series of temperature data collected for instance daily, or the closing stock price. 
Time series modeling is an important application of the theory of stochastic processes. 
After fitting a stochastic process to the data, i.e., a sequence of random variables indexed by $\mathbb{T}$, the model can be used to answer several questions. 
For instance, one may extract trend lines or seasonality (relevant in for instance financial or climate modeling) and perhaps most importantly, forecast the future. 
Time series models are usually (compared to deep models) simple and easy to fit, with theoretically well understood techniques.  
In this project, we consider the problem of _anomaly detection_, i.e., to, given a previous sample (i.e., training data) of different time series, detect if a new series is anomalous in some way. 
This can be various things, for instance, if the series contains outlier points, if a subset of the series does not fit in what one would expect, or, as is the case for us, if the series is entirely different in some sense from what is expected. 

<figure>
    <img src="report_images/GA1.png" alt="anomaly detection" title="anomaly detection">
    <figcaption>Illustration of the anomaly detection procedure. The autoencoder is trained on the normal (green) data. The idea is that normal data will be accurately reconstructed, whereas anomalies are reconstructed poorly.</figcaption>
</figure>

One viable approach to this would be to fit a time series model to the training data, and using a statistical approach determine if the new series is different from what the model predicts. 
In this project, however, we will avoid the modeling step and instead take a fully data-driven approach using a deep learning technique known as _autoencoders_. 

## Autoencoders

Autoencoders are neural networks that are used to learn embeddings of unlabeled data. 
An autoencoder consists of two networks, the _encoder_ and the _decoder_. 
The encoder learns a representation of the input data into some latent space, and the decoder learns to reconstruct the input data from the representations. 
Formally, the autoencoder consists of a $4$-tuple $(\mathcal D,\mathcal E,\varphi_E,\varphi_D)$. $\mathcal D$ is the data space (in our case, the space of time series of a certain length), and $\mathcal E$ is the latent space of representations, in our case chosen to be a Euclidean space of dimension $n$. 
Further, $\varphi_E\colon \mathcal D \to \mathcal E$ is the _encoder_ and $\varphi_D\colon \mathcal E \to \mathcal D$ is the _decoder_. 

The mappings $\varphi_E$ and $\varphi_D$ are parametrized by neural networks. 
In our case, both networks have a simple architecture: they are standard multilayer perceptrons, each with two hidden layers. The latent space (i.e., output dimension of the encoder) is taken to be 16-dimensional. The architecture of the decoder is just the same as that of the encoder, but reversed.  


To train the autoencoder, it is assigned a task to solve. 
In practice, this means selecting a data fidelity measure on $\mathcal{D}$, i.e., a function $d\colon \mathcal{D} \times \mathcal{D} \to \mathbb R$.
Then, given the parametrized coders $\varphi_E^{\theta_E}$ and $\varphi_D^{\theta_D}$, where $\theta_E$ and $\theta_D$ are the parameters of the functions indicated by their subscript, the training problem is to solve $\min_{\theta_E,\theta_D} \sum_{i=1}^N d(x_i,\varphi_D^{\theta_D}\circ \varphi_E^{\theta_E}(x_i)) := \min_{\theta_E,\theta_D} \text{loss}(\theta_E,\theta_D)$ where $\{x_i, i = 1,\ldots N\} \subset \mathcal D$ are the $N$ samples of the training data, e.g., the $N$ time series used to learn what a normal time series should look like. 

<figure>
    <img src="report_images/network.png">
    <figcaption>Architecture of the autoencoder when selecting 10 components for the encoded features.</figcaption>
</figure>


## Time series anomaly detection using autoencoders

The autoncoder is trained on normal, non-anomalous time series, collected in the dataset $\mathcal D$. Each time series is represented in a latent space $\mathcal E$ by the encoder and then the decoder is used to map $\mathcal E$ back to the time series given as input to the model. The underlying assumption is that if the autoencder is fed an anomalous signal, then its reconstruction performance is poor, either because the latent representation of the anomalous signal differs from the one of the training samples or because the reconstruction loss of the abnormal signal is high. This assumption is motivated by the fact that the autoencoder is trained on normal signals.

At inference time, the autoencoder trained on normal signals collected in the dataset $\mathcal D$ is given as input a new time series $\tilde{x}$, that is then mapped to its latent representation and then reconstructed back to $\varphi_D ( \varphi_E(\tilde{x}))$. The technique that we use to understand if the given signal is anomalous or not works in this way. 
1.  We compute the distribution of the reconstruction losses on the training set, namely the distribution of $\{d(z, \varphi_D ( \varphi_E(z))) \ \forall z \in \mathcal D\}$. For ease of notation, we re-define $d(\cdot, \varphi_D ( \varphi_E(\cdot)))$ as $d(\cdot)$ and by $d(\mathcal D)$ we mean the collection of the reconstruction losses on the training set $\mathcal D$. Moreover, we indicate the distribution of $d(\mathcal D)$ with $\pi_{\mathcal D}$. Notice that the support of this distribution is the set of positive real numbers.

2. The Cumulative Density Function of $\pi_{\mathcal D}$ is computed. More precisely, given a new time series $x$, $1-\text{CDF}\_{\pi\_{\mathcal D}}(d(x)) = 1-\mathbb P\_{\pi\_{\mathcal D}}(d(z) \leq d(x)) = P\_{\pi\_{\mathcal D}}(d(z) \geq d(x))$ where $d(z)$ is a random variable such that $d(z) \sim \pi_{\mathcal D}$.

If $x$ is a normal signal, then $d(x)$ is a sample from $\pi_{\mathcal D}$. Consequently, we do not expect $1-\text{CDF}_{\pi_{\mathcal D}}(d(x))$ to be low, namely we do not expect $d(x)$ to be an outlier with respect to the distribution of the reconstruction losses of normal signals (i.e. $\pi_{\mathcal D}$). Therefore:

3. Set a (small) threshold $\alpha$. If $1-\text{CDF}_{\pi_{\mathcal D}}(d(x)) \leq \alpha$, then the new time series $x$ is flagged as an anomaly. 

Notice that this anomaly detection pipeline returns a number in the $[0,1]$ interval (namely, $1-\text{CDF}_{\pi_{\mathcal D}}(d(x))$), which could be regarded as the "probability" that the new signal is anomalous. Ideally, this would be the final output of the pipeline, letting experts in the field actually have the last word on whether the given time series is atypical or not. If the data have been collected in the medical field, this option might be safer than relying on the chosen threshold $\alpha$. We choose to select a threshold and to actually flag time series as normal or anomalous for the purpose of evaluating the performance of the pipeline.

Remark: An alternative to this approach might be to compare the new time series $x$ and the normal signals in the dataset $\mathcal D$ through the embdedd features learned by the autoencoder. 

## The dataset
The raw data are collected in [this](https://www.physionet.org/content/chfdb/1.0.0/) repository, and they have originally been collected in [this work](http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=3950244&dopt=Abstract). The raw dataset includes long-term ECG recordings from 15 subjects (11 men, aged 22 to 71, and 4 women, aged 54 to 63) with severe congestive heart failure. The data have been furhter analyzed and preprocessed in [this work](https://link.springer.com/article/10.1007/s10618-014-0388-4): each heartbeat was extracted and the heartbeat length was equalized using interpolation. After that, 5,000 heartbeats were randomly selected. The dataset contains 2079 normal heartbeats. Anomalous heartbeats are further separated into 4 classes each corresponding to a specific heart condition. For our purposes, all anomalies are collected in one single class, as we decided not to detect a specific heart condition but rather just abnormal ECGs. Unfortunately, as far as we know, the raw data consists in a single, long sequence of evaluations of electrocardiograms. This means that we do not have access to individual heartbeats, but only to the 5000 randomly sampled signals.
<figure>
    <img src="report_images/signals_plot.png" alt="anomaly detection" title="anomaly detection">
    <figcaption>Plot of the ECG data (green: normal signals; red: anomalies).</figcaption>
</figure>

## Scalability

The primary objective of this project is to utilize an ensemble approach when the dataset exceeds the capacity of a single machine. Beyond handling large-scale data, adopting an ensemble model offers several advantages. Notably, it enables the application of a federated learning framework, where local nodes compute simple metrics (resulting in low transfer costs) and transmit them to the master node. This approach ensures data privacy by keeping sensitive information localized.

We begin by outlining the ensemble approach. Consider a dataset $D$ with size $S_D$, and assume that a single node can feasibly train a model on data of size $S_1$ within a reasonable timeframe. If $S_1 \geq S_D $, there is no need to employ an ensemble, and a single model can be trained on the entire dataset. However, if $S_1 < S_D $, we partition the dataset as $ D = \bigcup_{i=1}^N D_i $, dividing it into chunks $D_i$ such that $|D_i| \leq S_1$, each of which fits within the capacity of a single node. Here, $N$ denotes the total number of nodes used.


The process proceeds by letting 
1. Trainig
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

This approach works for any type of data partitioning, making it well-suited for federated learning, where data privacy must be maintained at each node. In our application with ECG data, for instance, where data not necessarily can be shared between hospitals, we would train a separate model at each hospital. During the inference stage (assuming this is permitted by the hospitals), the new data would be processed at each node (hospital) to evaluate anomalies by averaging the outputs from the individual models. This ensures the privacy of the original data, as it remains within the hospital and only the aggregated inference results are returned to the master node.

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

In our setting we will let the function, "training", be a function that trains one model with one partitioned data set. <args> will contain training instructions, such as epochs to run, learning rate and other hyperparameters we might want to tune. 

*TO BE SPECIFIED* The data is partitioned in the preprocessing and each node is assigned one partition. The models are either created at this stage or initialized in the "training" function.

Running the ensemble on our own computer (laptop without GPUs `use_gpu = False`) , we see that we get best scaling by setting $N$ as the number of CPU cores. We set `local_mode = True`, otherwise the master node (i.e. the only node) will not work.

In our main training function (where each node is working) we make use `local_rank = int(os.environ["LOCAL_RANK"])`. This retrieves which node that is working and in a very simple way we can collect the correct partition and model. Similarly, the model parameters are saved based on their `local_rank` to know which node it belongs to.

## Discussion

TODO: PLENTY OF RESEARCH SUGGESTS AUTOENCODERS IS NOT SUITABLE FOR THE APPLICATION: SHOULD WE DISCUSS THIS?

Neural networks in general and autoencoders can have surprisingly good out-of-sample performance, meaning that even if an autoencoder has not seen a type of anomalous time series, it could still reconstruct the anomalous time series. This in turn results in a low reconstruction performance, meaning that we do not detect the anomaly. For medical applications, one could ask, given this, if it is reasonable to still use autoencoders for anomaly detection purposes. On the one hand, a human interpreting ECG time series can also make mistakes, but on the other, the by some perceived objectivity of deep learning based methods could mean that life-threatening diseases are never diagnosed, and the patient is sent home, because "the computer is objective and cannot lie". 


## Final notes:

### Collaboration Environment

The technical stack we ustilize to set up our collaboration environment consists of **Docker** and **Github**.

We chose Docker since we have hetogenious devices (4 Macs and 1 Windows) which suits containerization and it is widely used in the industry. We take it as a learning opportunity to increase our knowledge and extend our skillset. It further prevents dependency inconsistency and conflits.

The development environment is built on top of an official pyspark docker image. Each team member can pull it down to their machine and run it locally for coding. 

![Collaboration Environment](report_images/timeseries-dev-env-v2.png "Collaboration Environment")

This ensures the team has an uniform development environment. Afterwards, the team members push their code to the git repository.

### Choice of PySpark, PyTorch and TorchDistributor
We chose PySpark as the framework to develop the scalable and distributed machine learning pipeline.

We chose PyTorch to develop the model since we had two Macs with Apple M1 chip. Both experienced dependency compatibility issues with TensorFlow. 

We investigated both TorchDistributor and Flower (a federated learning framework) for realizing distributed machine learning. We chose TorchDistributor because it is native to Pyspark and for its simplicity and brevity.