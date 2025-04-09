#!/bin/bash

# Make sure the chroma_db directory exists
mkdir -p /home/site/wwwroot/chroma_db

# Copy the chroma database from the persisted storage to the app directory
cp -r /home/site/persistentstorage/chroma_db/* /home/site/wwwroot/chroma_db/

# Start the Gunicorn server
gunicorn --bind=0.0.0.0:8000 prowl:app