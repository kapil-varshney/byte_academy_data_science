#!/usr/bin/env bash

python3 scraper.py
TIMESTAMP=$(date  +%s)
rm payload.pkl
rm -rf __pycache__
clear && echo 'EXITED:' $TIMESTAMP && tree --dirsfirst
