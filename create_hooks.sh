#!/bin/sh

# Define the source and destination directories
source_dir="hooks/*"
destination_dir=".git/hooks/"

# Copy the files
cp $source_dir $destination_dir

# Make the copied files executable
chmod +x $destination_dir*
