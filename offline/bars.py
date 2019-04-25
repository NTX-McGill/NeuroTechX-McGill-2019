#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:59:29 2019

@author: marley
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.figure()
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 15}

#matplotlib.rc('font', **font)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title


chan_4 = [[77.79329608938548,
          84.48637316561845,
          87.39495798319328,
          88.81856540084388,
          91.45299145299145],
         [65.1984126984127,
          69.55357142857143,
          72.70833333333333,
          73.75,
          77.05357142857143],
         [63.539307667421554,
          64.65997770345597,
          70.80909571655208,
          74.26160337552743,
          77.77777777777779],
         [58.9037284362827,
          59.956167814652474,
          63.9515455304929,
          65.11919698870766,
          67.38035264483628],
         [51.01851851851852,
          52.39583333333333,
          52.63888888888889,
          55.34722222222223,
          55.55555555555556],
         [48.40902909980962,
          48.10223446587083,
          49.080506742950554,
          48.55562384757222,
          46.28252788104089],
         [50.62091503267973,
          51.617647058823536,
          51.813725490196084,
          50.73529411764706,
          49.11764705882353]]

chan_2 = [[75.60521415270019,
          81.44654088050315,
          87.74509803921569,
          91.13924050632912,
          93.80341880341881],
         [63.948412698412696,
          67.94642857142857,
          72.17261904761905,
          76.42857142857142,
          81.07142857142857],
         [66.6774506632158,
          71.01449275362319,
          75.99153886832364,
          78.90295358649789,
          81.02564102564102],
         [57.56816917084029,
          60.206637445209765,
          63.11612364243943,
          63.6762860727729,
          65.23929471032746],
         [51.141975308641975,
          52.326388888888886,
          52.824074074074076,
          56.111111111111114,
          57.08333333333333],
         [50.02719608376394,
          48.34710743801653,
          48.97834082550061,
          49.0780577750461,
          46.84014869888476],
         [50.3921568627451,
          51.507352941176464,
          52.352941176470594,
          49.9264705882353,
          50.882352941176464]]
chan_4 = np.array(chan_4).T
chan_2 = np.array(chan_2).T
window_sizes = [1,2,4,6,8]

limit = 3
chan_4 = chan_4[:limit]
chan_2 = chan_2[:limit]
window_sizes = window_sizes[:limit]

both = True
# shape distinct_windows, num_subjects
distinct_windows, num_subjects = chan_4.shape

# set width of bar
barWidth = 1/(distinct_windows + 1)

colors = ['#fde725', '#4fc16b','#546e9f', '#239a89','#440658']

# Set position of bar on X axis
r1 = np.arange(num_subjects)
if both:
    r1 = r1 * 2

# Make the plot
"""
2 channels
"""
i = 0
two_channels = []
for window_, window_s in zip(chan_2, window_sizes):
    r = [x + barWidth * i for x in r1]
    two_channels.append(plt.bar(r, window_, width=barWidth, color=colors[int(i/2)],edgecolor='white', label=str(window_s) + ' s', alpha=0.5))
    i+= 1
    if both:
        i += 1

"""
4 channels
"""
i = 1
four_channels = []
for window_, window_s in zip(chan_4, window_sizes):
    r = [x + barWidth * i for x in r1]
    four_channels.append(plt.bar(r, window_, width=barWidth, color=colors[int(i/2)], edgecolor='white', label=str(window_s) + ' s'))
    i += 1
    if both:
        i+= 1

# Add xticks on the middle of the group bars
#plt.title('Inter-subject Accuracy by Window Size',fontweight='bold')
# plt.style.use('ggplot')
plt.ylabel('Accuracy (%)')#, fontweight='bold')
plt.xlabel('Subject')#, fontweight='bold')
plt.xticks([r + barWidth*distinct_windows for r in r1], [i for i in range(1,num_subjects + 1)])

# Create legend & Show graphic
legend1 = plt.legend(handles=four_channels, bbox_to_anchor=(1.04,0.65), title="4 channels", loc="upper left")
plt.legend(handles=two_channels, bbox_to_anchor=(1.04,1), title="2 channels", loc="upper left")
plt.gca().add_artist(legend1)
plt.subplots_adjust(right=0.8)
plt.show()
