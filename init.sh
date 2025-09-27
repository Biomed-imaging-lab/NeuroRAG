#!/bin/bash

# Set the name of your environment
ENV_NAME="neurorag"

# Start the environment.yml file
echo "name: $ENV_NAME" > environment.yml
echo "channels:" >> environment.yml
echo "  - conda-forge" >> environment.yml
echo "  - defaults" >> environment.yml
echo "dependencies:" >> environment.yml

# Read each line of requirements.txt and add it to environment.yml
while IFS= read -r line
do
  # Strip any leading/trailing spaces and add the line to environment.yml
  line=$(echo $line | xargs)
  if [ -n "$line" ]; then
    echo "  - $line" >> environment.yml
  fi
done < requirements.txt
