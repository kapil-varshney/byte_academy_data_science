#!/usr/bin/env bash

cp toolkit/build_track.py .
python3 build_track.py
rm build_track.py
mkdir phase_1
mkdir phase_2
mkdir phase_3
mv week_01 phase_1
mv week_02 phase_1
mv week_03 phase_1
mv week_04 phase_1
mv week_05 phase_2
mv week_06 phase_2
mv week_07 phase_2
mv week_08 phase_2
mv week_09 phase_3
mv week_10 phase_3
mv week_11 phase_3
mv week_12 phase_3