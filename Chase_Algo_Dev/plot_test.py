import numpy as np
import matplotlib.pyplot as plt
plt.rc('legend', frameon=False)
x = [1,2,3,4,5]
y = x
MAPX = [-4.0, 2.0]
MAPY = [-1.5, 4.5]
ALIAS_LIST = ["Leader", "Follower-1", "Follower-2", "Follower-3"]
COLOR_LIST = ["r", "g", "b", "k", "c", "p"]
id = 0
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
axes.set_xlim(MAPX)
axes.set_ylim(MAPY)
for id in range(4):
    axes.plot(x,y, COLOR_LIST[id], linewidth=2)
axes.grid()
axes.tick_params(axis="y", which="both", direction="in", right=True)
leg = axes.legend(ALIAS_LIST)
#leg.get_frame().set_edgecolor('k')
#leg.get_frame().set_facecolor('w')
axes.set_title("Motion of the Formation")
axes.set_xlabel("Position X (m)")
axes.set_ylabel("Position Y (m)")
fig.savefig('test.eps', format='eps', dpi=1000)
plt.show()
