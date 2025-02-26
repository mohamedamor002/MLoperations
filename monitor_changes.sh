#!/bin/bash

# Directory to monitor
DIRECTORY="./"

# Exclude patterns (files or directories to ignore)
EXCLUDE_PATTERNS="preprocessed_data.joblib|trained_model.joblib|scaler.joblib"

# Monitor for changes, excluding specified patterns
inotifywait -m -r -e modify,move,create,delete --exclude "${EXCLUDE_PATTERNS}" ${DIRECTORY} |
while read path action file; do
    echo "Change detected in ${file}. Running make all..."
    make all
done