### __ Plots & Animation __ ###    

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation

from IPython.display import HTML


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_lbl.set_text('')
    
    return (line1,line2,time_lbl)

def animate(i):
    earth_trail = 40
    jupiter_trail = 200
    
    time_lbl.set_text('t = ' + str(round(t[i],1)) + ' yr')
    line1.set_data(earth['r'][i:max(1,i-earth_trail):-1,0], earth['r'][i:max(1,i-earth_trail):-1,1])
    line2.set_data(jupiter['r'][i:max(1,i-jupiter_trail):-1,0], jupiter['r'][i:max(1,i-jupiter_trail):-1,1])
    
    return (line1,line2,time_lbl)


# plot framework

fig, ax = plt.subplots()
#fig, ax = plt.subplots(figsize=(13,4.5))

ax.set_xlim((-5.2, 5.2))
ax.set_ylim((-5.2, 5.2))
ax.axis('off')


# result settings

line1, = ax.plot([], [], 'o-',color = '#d2eeff',markevery=10000, markerfacecolor = '#0077BE',lw=2)   # line for Earth

line2, = ax.plot([], [], 'o-',color = '#e3dccb',markersize = 8, markerfacecolor = '#f66338',lw=2,markevery=10000)   # line for Jupiter

line3, = ax.plot(0,0,'o',markersize = 9, markerfacecolor = "#FDB813",markeredgecolor ="#FD7813" )

time_lbl = ax.text(-6, -5.5, '')

# result animation

#plt.show()
plt.close()

matplotlib.rcParams['animation.embed_limit'] = 2**128

anm = animation.FuncAnimation(fig, animate, init_func=init, frames=4000, interval=5, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
anm.save('test_1.3_result.mp4', writer=writer)