#!/bin/bash

# Specify the URL of the zip file
ZIP_URL="https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip"

# Download the zip file
wget "$ZIP_URL"

# Unzip the downloaded file
unzip -d uncased_L-8_H-512_A-8 uncased_L-8_H-512_A-8.zip

# Remove the downloaded zip file if you don't need it
# rm uncased_L-8_H-512_A-8.zip

echo "Download and unzip complete."

'''Before running the script, make sure to give it executable permissions using the following command:
chmod +x download.sh
'''