Features are along x-axis(columns 1 to 80)
Samples are along the y-axis(rows)
The last column(81) consists labels such that:
1 == index_finger
2 == middle_finger
3 == ring_finger
4 == little_finger
5 == thumb
6 == rest
7 == victory_gesture

There are 80 columns because there were 8 electrodes and 10 features were extracted for each electrode.

Features are in the order {standard_deviation; root_mean_square; minimum; maximum; zero_crossings; average_amplitude_change; amplitude_first_burst; mean_absolute_value; wave_form_length; willison_amplitude}

First 8 columns are standard_deviation, the next 8 columns are root_mean_square and so on according to the order described above...

Note: You may want to normalize some features because their ranges are dramatically different.