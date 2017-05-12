#!/usr/bin/env python3

import os

# Setting

weeks_in_track = 12
days_in_week = 5

# Scripting

for week in range(weeks_in_track):
    week += 1
    os.system('mkdir week_{0}'.format(week))
    os.system('touch week_{0}'.format(week))
    os.system('echo "# Week {0}" >> week_{0}/README.md'.format(week))


    for day in range(days_in_week):
        day += 1
        os.system('mkdir week_{0}/day_{1}'.format(week, day))
        os.system('touch week_{0}/day_{1}/README.md'.format(week, day))
        os.system('echo "# Day {0}" >> week_{0}/day_{1}/README.md'.format(week, day))
