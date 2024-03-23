#!/bin/bash

 ENV_PATH=$(poetry env info --path)
 if [ -d "$ENV_PATH" ]; then
     rm -rf $ENV_PATH
     echo "Removed the poetry environment at $ENV_PATH."
 else
     echo "No poetry environment found."
 fi

 poetry install --extras benchmark
 echo "Setup completed successfully."
 exit 0
