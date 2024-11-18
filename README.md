# How to run the project with the PySpark docker image

## 1. Clone the project
Git clone this repo to a directory of your machine:

`git clone https://github.com/PengKuang/team_scalable_timeseries.git`

Then, navigate into the project folder you just cloned.
e.g., on MacOS:

`cd path/to/your/project_name`

For me, it is: 

`cd /Users/pengkuang/team_scalable_timeseries` 

Or if I'm already in my home directory '/Users/pengkuang/' where I git cloned the project: 

`cd team_scalable_timeseries`

## 2. Run the docker image
Assuming that you have already installed the Docker app on your machine. 

### Step 1
First, pull down this docker image: <https://hub.docker.com/r/jupyter/pyspark-notebook>

`docker pull jupyter/pyspark-notebook`

Then run it locally: 

`docker run -it -p 8888:8888 -v $(pwd):/work -w /work jupyter/pyspark-notebook`

This command is to run the docker image in an interactive mode, with the port number 8888 and mount your current folder to the work directory in the container. 

### Step 2
After this, open your browser and go to 'http://localhost:8888'

You should see a unusual login page asking you for a token. 

Read this article for where to find the token: <https://levelup.gitconnected.com/using-docker-and-pyspark-134cd4cab867>

Go to your terminal and find that token. Paste it in and then you can log into the container.

### Notes

Now you can code freely there. Note you still need to use 'pip install package_name' to install some libraries such as torch by yourself. 

All the changes you made in the container will be saved to your local project directory. Once you are happy with it, you can push it to the remote team repository. 

## 3. Dataset 

Download the corresponding dataset and move it to your project folder so that the Autoencoder can read it. 

GaitRec: <https://springernature.figshare.com/collections/GaitRec_A_large-scale_ground_reaction_force_dataset_of_healthy_and_impaired_gait/4788012/1>
