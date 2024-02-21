#!/bin/bash

# File to append the output
output_file="/data/personal/nus-ytj/watch_nvidia.log"

# Interval in seconds between checks
interval=5

while true; do
  # Append the current date and time to the file
  echo "Timestamp: $(date)" > "$output_file"
  # Append the nvidia-smi output to the file
  nvidia-smi >> "$output_file"
  # Wait for the specified interval
  sleep "$interval"
done
