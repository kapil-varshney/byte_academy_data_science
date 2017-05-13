#!/usr/bin/env python3

import os


# Setting

default_weeks_in_track = 12
default_days_in_week   = 5


# Scripting

for week in range(default_weeks_in_track):
    week += 1

    if week < 10:

        os.system('mkdir week_0{0}'.format(week))
        os.system('touch week_0{0}'.format(week))
        os.system('echo "# Week 0{0}" >> week_0{0}/README.md'.format(week))

        for day in range(default_days_in_week):
            day += 1

            os.system('mkdir week_0{0}/day_{1}'.format(week, day))
            os.system('touch week_0{0}/day_{1}/README.md'.format(week, day))
            os.system('echo "# Week 0{0}, Day {1}" >> week_0{0}/day_{1}/README.md'.format(week, day))


    else:
        os.system('mkdir week_{0}'.format(week))
        os.system('touch week_{0}'.format(week))
        os.system('echo "# Week {0}" >> week_{0}/README.md'.format(week))

        for day in range(default_days_in_week):
            day += 1

            os.system('mkdir week_{0}/day_{1}'.format(week, day))
            os.system('touch week_{0}/day_{1}/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}" >> week_{0}/day_{1}/README.md'.format(week, day))
