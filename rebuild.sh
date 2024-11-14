#!/bin/bash

# Stop and remove the container
docker-compose down

# Remove the image
docker rmi $(docker images -q "timeseries-dev")

# Build the image
docker-compose build --no-cache

# Build and start the container
docker-compose up

# check if the container is running
docker ps

# Enter the container
# docker run -it timeseries-dev:latest /bin/bash