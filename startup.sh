#!/bin/bash
apt-get update && apt-get install -y sqlite3 libsqlite3-dev
python -m gunicorn -w 4 -b 0.0.0.0:8000 prowl:app