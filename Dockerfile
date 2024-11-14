# Use the official JupyterHub Docker image as the base
FROM jupyterhub/jupyterhub:latest

# Install OpenJDK 11
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    apt-get install -y vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make a directory for Spark installation
RUN mkdir -p /usr/local/spark

# Set environment variables for Spark
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
ENV PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"
RUN echo "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(type -P java))))" > ~/.bashrc
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Install Spark
RUN curl -L https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz | tar -xz -C /opt && \
   mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}/* $SPARK_HOME

RUN cd /srv/jupyterhub/
RUN jupyterhub upgrade-db

# # Create a new user jovyan
# RUN useradd -m jovyan
# RUN useradd -m test
# RUN useradd -m peng
# RUN useradd -m semla
# RUN useradd -m mechele
# RUN useradd -m kasper
# RUN useradd -m erik

WORKDIR /home/jovyan/work
RUN mkdir -p /home/jovyan/work/data
RUN cd /home/jovyan/work

# Copy a configuration file for JupyterHub (explained below)
COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py
COPY test.py /home/jovyan/work/test.py

# Install PyTorch, PySpark, and JupyterLab (optional but recommended for more features)
# RUN pip install torch && \
#     pip install jupyterlab pyspark numpy pandas scipy && \
#     pip install traitlets dockerspawner

RUN pip install findspark jupyterlab dockerspawner

# Expose JupyterHub's default port
EXPOSE 8000

# Start JupyterHub when the container launches
CMD ["jupyterhub", "--config", "/srv/jupyterhub/jupyterhub_config.py"]
