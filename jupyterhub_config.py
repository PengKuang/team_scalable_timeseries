from traitlets.config import get_config
import os
import glob

# Create a Config object
c = get_config()

# DockerSpawner configuration to isolate user notebooks in containers
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = 'jupyter/scipy-notebook:latest'  # Base image for user notebooks

c.DockerSpawner.network_name = 'jupyterhub'
c.DockerSpawner.remove = True

# c.Spawner.environment = {'SPARK_HOME': '/usr/local/spark'}
# c.Spawner.environment = {'PYSPARK_SUBMIT_ARGS': '--master local[2] pyspark-shell'}
# c.Spawner.environment = {'JAVA_HOME': '/usr/lib/jvm/java-11-openjdk-amd64'}

c.Spawner.env_keep.append('JAVA_HOME')
c.Spawner.env_keep.append('SPARK_HOME')
c.Spawner.env_keep.append('PYSPARK_SUBMIT_ARGS')

# # Find pyspark modules to add to PYTHONPATH, so they can be used as regular
# # libraries
# pyspark = '/usr/local/spark/python/'
# pypath = os.path.join(pyspark, 'lib', 'py4j-*.zip')
# py4j_files = glob.glob(pypath)
# # print(f"py4j_files: {py4j_files}")
# py4j = py4j_files[0]
# # print(f"py4j: {py4j}")
# pythonpath = ':'.join([pyspark, py4j])

# # Set PYTHONPATH and PYSPARK_PYTHON in the user's notebook environment
# c.YarnSpawner.environment = {
#     'PYTHONPATH': pythonpath,
#     'PYSPARK_PYTHON': '/opt/jupyterhub/miniconda/bin/python',
# }

# Configure DockerSpawner to mount persistent volumes for each user
# notebook_dir = os.environ.get('DOCKER_NOTEBOOK_DIR') or '/home/jovyan/work'
# c.DockerSpawner.notebook_dir = notebook_dir

# c.DockerSpawner.volumes = { 'jupyterhub-user-{username}': notebook_dir }
c.DockerSpawner.volumes = { 'jupyterhub-user-{username}': '/home/jovyan/work' }

# Persistence
c.JupyterHub.db_url = "sqlite:///data/jupyterhub.sqlite"

# Network settings
c.JupyterHub.hub_ip = '0.0.0.0'  # JupyterHub listens on all IPs
c.JupyterHub.port = 8000

# Authentication (simple login with usernames/passwords or OAuth)
c.JupyterHub.authenticator_class = 'jupyterhub.auth.PAMAuthenticator'

# # Admin users
c.Authenticator.admin_users = {'jovyan', 'peng', 'root'}

# Sample user configuration for local users
c.Authenticator.allowed_users = {'erik', 'kasper', 'mechele', 'selma', 'test'}
c.LocalAuthenticator.create_system_users = True