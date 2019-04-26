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
chan_2 = [[[76.22794026903615, 3.5685147247787223],
  [81.85703185703186, 4.538309820571024],
  [87.29591836734694, 3.703346003674905]],
 [[63.55855855855856, 4.98455531995419],
  [69.01515151515152, 3.975649019212704],
  [74.56666666666666, 5.070393366287168]],
 [[66.15644631265788, 4.405181961106589],
  [72.60848522191708, 6.255148913963715],
  [76.86926569848113, 5.7705459168686914]],
 [[57.72384629527487, 1.5627098117352618],
  [60.18794818413139, 6.9625307750503165],
  [62.86666666666666, 7.3805258636687405]],
 [[51.09797297297297, 2.8826794251825953],
  [52.55681818181819, 3.6286119239937324],
  [54.775, 5.075615726195197]],
 [[49.56819427288433, 3.3876808129159284],
  [49.23754021872301, 3.959675319533922],
  [49.53172606255181, 5.972517957341434]],
 [[51.554054054054056, 3.6850660059522657],
  [51.13636363636363, 6.125364253331906],
  [52.175, 8.528884745381426]]]
 
chan_4 = [[[78.28119215105517, 2.371995233658082],
  [83.41103341103342, 2.4218754009145576],
  [88.14285714285715, 3.5341298194113624]],
 [[65.05630630630631, 5.225965406568135],
  [69.69696969696969, 4.152178668498399],
  [73.43333333333334, 3.7101961613310386]],
 [[62.63813258500653, 3.360721360827933],
  [66.10948005846106, 6.280873434432706],
  [72.92946114892048, 8.482918245977306]],
 [[59.14460378746092, 2.8140893954289794],
  [60.193731205181585, 8.169242914015365],
  [62.973737373737364, 9.057154731561441]],
 [[50.50675675675676, 2.6927604291877243],
  [53.63636363636364, 5.306546453363807],
  [54.675000000000004, 5.581834375901884]],
 [[49.54785986369437, 3.934440578681111],
  [49.11768158810407, 4.625096371438731],
  [49.496935818297345, 5.634202338946582]],
 [[51.38513513513514, 2.84927713719267],
  [51.81818181818182, 6.619799100317042],
  [51.275, 9.611288935413398]]]
#chan_4 = np.array(chan_4).T

chan_4_err = np.array(chan_4)[:,:,1].T
chan_4 = np.array(chan_4)[:,:,0].T
chan_2_err = np.array(chan_2)[:,:,1].T
chan_2 = np.array(chan_2)[:,:,0].T

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
for window_, window_s,err_s in zip(chan_2, window_sizes,chan_2_err):
    r = [x + barWidth * i for x in r1]
    two_channels.append(plt.bar(r, window_, width=barWidth, yerr=err_s, error_kw=dict(lw=1, capsize=1, capthick=1), color=colors[int(i/2)],edgecolor='white', label=str(window_s) + ' s', alpha=0.5))
    i+= 1
    if both:
        i += 1

"""
4 channels
"""
i = 1
four_channels = []
for window_, window_s,err_s in zip(chan_4, window_sizes,chan_4_err):
    r = [x + barWidth * i for x in r1]
    four_channels.append(plt.bar(r, window_, width=barWidth, yerr=err_s, error_kw=dict(lw=1, capsize=1, capthick=1), color=colors[int(i/2)], edgecolor='white', label=str(window_s) + ' s'))
    i += 1
    if both:
        i+= 1

# Add xticks on the middle of the group bars
#plt.title('Inter-subject Accuracy by Window Size',fontweight='bold')
# plt.style.use('ggplot')
plt.ylabel('Accuracy (%)')#, fontweight='bold')
#plt.xlabel('Subject')#, fontweight='bold')
plt.xticks([r + barWidth*distinct_windows for r in r1], ["S" + str(i) for i in range(1,num_subjects + 1)])

# Create legend & Show graphic
legend1 = plt.legend(handles=four_channels, bbox_to_anchor=(1.04,0.65), title="4 channels", loc="upper left")
plt.legend(handles=two_channels, bbox_to_anchor=(1.04,1), title="2 channels", loc="upper left")
plt.gca().add_artist(legend1)
plt.subplots_adjust(right=0.8)
plt.show()
