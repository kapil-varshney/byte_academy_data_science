#!/usr/bin/env bash

python3 scraper.py
TIMESTAMP=$(date  +%s)
rm -rf __pycache__
clear && echo 'EXITED:' $TIMESTAMP && tree --dirsfirst
