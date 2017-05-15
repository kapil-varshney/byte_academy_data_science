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
        os.system('echo "# Week 0{0}" >> week_0{0}/README.md'.format(week))

        for day in range(default_days_in_week):
            day += 1

            os.system('mkdir week_0{0}/day_{1}'.format(week, day))
            os.system('mkdir week_0{0}/day_{1}/assignments'.format(week, day))
            os.system('mkdir week_0{0}/day_{1}/challenges'.format(week, day))
            os.system('mkdir week_0{0}/day_{1}/drills'.format(week, day))
            os.system('mkdir week_0{0}/day_{1}/drills/warmups'.format(week, day))
            os.system('mkdir week_0{0}/day_{1}/drills/walkthroughs'.format(week, day))
            os.system('mkdir week_0{0}/day_{1}/exercises'.format(week, day))

            os.system('touch week_0{0}/day_{1}/README.md'.format(week, day))
            os.system('touch week_0{0}/day_{1}/assignments/README.md'.format(week, day))
            os.system('touch week_0{0}/day_{1}/challenges/README.md'.format(week, day))
            os.system('touch week_0{0}/day_{1}/drills/README.md'.format(week, day))
            os.system('touch week_0{0}/day_{1}/drills/warmups/README.md'.format(week, day))
            os.system('touch week_0{0}/day_{1}/drills/walkthroughs/README.md'.format(week, day))
            os.system('touch week_0{0}/day_{1}/exercises/README.md'.format(week, day))

            os.system('echo "# Week 0{0}, Day {1}: Overview" >> week_0{0}/day_{1}/README.md'.format(week, day))
            os.system('echo "# Week 0{0}, Day {1}: Assignments" >> week_0{0}/day_{1}/assignments/README.md'.format(week, day))
            os.system('echo "# Week 0{0}, Day {1}: Challenges" >> week_0{0}/day_{1}/challenges/README.md'.format(week, day))
            os.system('echo "# Week 0{0}, Day {1}: Drills" >> week_0{0}/day_{1}/drills/README.md'.format(week, day))
            os.system('echo "# Week 0{0}, Day {1}: Warmups" >> week_0{0}/day_{1}/drills/warmups/README.md'.format(week, day))
            os.system('echo "# Week 0{0}, Day {1}: Walkthroughs" >> week_0{0}/day_{1}/drills/walkthroughs/README.md'.format(week, day))            
            os.system('echo "# Week 0{0}, Day {1}: Exercises" >> week_0{0}/day_{1}/exercises/README.md'.format(week, day))

            if day == 1:
                os.system('echo "" >> week_0{0}/day_{1}/README.md'.format(week, day))
                os.system('echo "Reminder: Debates begin at 10:10 AM." >> week_0{0}/day_{1}/README.md'.format(week, day))

            if day == 5:
                os.system('echo "" >> week_0{0}/day_{1}/README.md'.format(week, day))
                os.system('echo "Reminder: Check-ins begin at 11:40 AM." >> week_0{0}/day_{1}/README.md'.format(week, day))

                os.system('echo "" >> week_0{0}/day_{1}/assignments/README.md'.format(week, day))
                os.system('echo "Debate preparation" >> week_0{0}/day_{1}/assignments/README.md'.format(week, day))


    else:
        os.system('mkdir week_{0}'.format(week))
        # os.system('mkdir week_{0}/debates'.format(week))
        os.system('echo "# Week {0}" >> week_{0}/README.md'.format(week))

        for day in range(default_days_in_week):
            day += 1

            os.system('mkdir week_{0}/day_{1}'.format(week, day))
            os.system('mkdir week_{0}/day_{1}/assignments'.format(week, day))
            os.system('mkdir week_{0}/day_{1}/challenges'.format(week, day))
            os.system('mkdir week_{0}/day_{1}/drills'.format(week, day))
            os.system('mkdir week_{0}/day_{1}/drills/warmups'.format(week, day))
            os.system('mkdir week_{0}/day_{1}/drills/walkthroughs'.format(week, day))
            os.system('mkdir week_{0}/day_{1}/exercises'.format(week, day))

            os.system('touch week_{0}/day_{1}/README.md'.format(week, day))
            os.system('touch week_{0}/day_{1}/assignments/README.md'.format(week, day))
            os.system('touch week_{0}/day_{1}/challenges/README.md'.format(week, day))
            os.system('touch week_{0}/day_{1}/drills/README.md'.format(week, day))
            os.system('touch week_{0}/day_{1}/drills/warmups/README.md'.format(week, day))
            os.system('touch week_{0}/day_{1}/drills/walkthroughs/README.md'.format(week, day))
            os.system('touch week_{0}/day_{1}/exercises/README.md'.format(week, day))

            os.system('echo "# Week {0}, Day {1}: Overview" >> week_{0}/day_{1}/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}: Assignments" >> week_{0}/day_{1}/assignments/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}: Challenges" >> week_{0}/day_{1}/challenges/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}: Drills" >> week_{0}/day_{1}/drills/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}: Warmups" >> week_{0}/day_{1}/drills/warmups/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}: Walkthroughs" >> week_{0}/day_{1}/drills/walkthroughs/README.md'.format(week, day))
            os.system('echo "# Week {0}, Day {1}: Exercises" >> week_{0}/day_{1}/exercises/README.md'.format(week, day))

            if day == 1:
                os.system('echo "" >> week_{0}/day_{1}/README.md'.format(week, day))
                os.system('echo "Reminder: Debates begin at 10:10 AM." >> week_{0}/day_{1}/README.md'.format(week, day))

            if day == 5:
                os.system('echo "" >> week_{0}/day_{1}/README.md'.format(week, day))
                os.system('echo "Reminder: Check-ins begin at 11:40 AM." >> week_{0}/day_{1}/README.md'.format(week, day))

                os.system('echo "" >> week_{0}/day_{1}/assignments/README.md'.format(week, day))
                os.system('echo "Debate preparation" >> week_{0}/day_{1}/assignments/README.md'.format(week, day))
