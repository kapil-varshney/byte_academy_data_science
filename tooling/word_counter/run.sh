#!/usr/bin/env bash

python3 core.py >> logs/keyword_frequency.log
TIMESTAMP=$(date  +%s)
rm -rf reference_data/__pycache__
clear && echo 'LAST' $TIMESTAMP && tree --dirsfirst
