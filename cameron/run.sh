#!/bin/bash

# Build the Docker image
docker build -t myapp .

# Run the Docker container
docker run -p 5000:5000 myapp