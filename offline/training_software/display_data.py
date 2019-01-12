import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    with open('batch-0.csv','r') as f:
        reader = csv.reader(f, delimiter=',')
        data = [line[0] for line in reader]
    # ax1.clear()
    print(len(data))
    # ax1.plot([1,2,3])

ani = animation.FuncAnimation(fig, animate, interval=2000)
plt.show()
