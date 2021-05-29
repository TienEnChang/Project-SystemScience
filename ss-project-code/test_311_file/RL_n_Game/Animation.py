import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

t = []
State_History = []
rank = [0,0,0]
#color_cA = ['#0055cc', '#ffcc00', '#ff3300']
color_cA = ['#555555','#555555','#555555']
color_cB = ['#ff3300','#ff9900','#338833','#A9A9A9']
color_l = '#e3dccb'

Bc = 5
Ba = 0.5
trail_len = 10        

def init():                               

    time_lbl.set_text('')

    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])

    circle0 = plt.Circle((0, 0), Bc, color=color_cB[3], fill=False)
    ax.add_patch(circle0)

    return (line1,line2,line3,time_lbl)

def animate(i):                          

    time_lbl.set_text('t = ' + str(round(t[i],2)) + ' secs')

    draw_circle = True
    if i > len(State_History)-1:
        i = len(State_History)-1
        draw_circle = False

    k = i+1 if i < trail_len else trail_len 

    data = np.zeros([3,2,k])
    for n1 in range(3):
        for n2 in range(k):
            data[n1,:,n2] = State_History[i-n2][n1]['r']
        
    line1.set_data( data[0,0], data[0,1] )
    line2.set_data( data[1,0], data[1,1] )
    line3.set_data( data[2,0], data[2,1] )

    if draw_circle:
        for n in range(3):
            if State_History[i][n]['alive/dead'] == 0 and State_History[i-1][n]['alive/dead']  == 1 :
                rank[n] = 1
                circle = plt.Circle(State_History[i][n]['r'], Ba, color=color_cB[sum(rank)-1], fill=False)
                ax.add_patch(circle)
                
    return (line1,line2,line3,time_lbl)


# plot framework

fig, ax = plt.subplots()

def plot_settings():
    k = 5.2
    ax.set_xlim((-k, k))
    ax.set_ylim((-k, k))
    ax.set_aspect('equal')
    ax.axis('off')


# result settings                         

line1, = ax.plot([], [], 'o-',color = color_l, markerfacecolor = color_cA[0], markevery=10000, markersize = 8, lw=2)   # line for Earth

line2, = ax.plot([], [], 'o-',color = color_l, markerfacecolor = color_cA[1], markevery=10000, markersize = 8, lw=2)   # line for Jupiter

line3, = ax.plot([],[],'o-',color = color_l, markerfacecolor = color_cA[2], markevery=10000, markersize = 8 ,lw=2)

time_lbl = ax.text(-8, -4.5, '')


# result animation 

matplotlib.rcParams['animation.embed_limit'] = 2**128


def ani_to_video(location, max_frames):

    f = len(State_History) if len(State_History) < max_frames else max_frames

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=f, interval=5, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(location, writer=writer)