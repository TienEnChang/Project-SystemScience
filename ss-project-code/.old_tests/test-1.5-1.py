### __ Numerical Integration __ ###

import numpy as np

        # use "np.array([])" rather than "[]"


# Constants . __ { 1 = time unit (yr) , 1 = length unit (AU) , 1 = mass unit (MoE) }

T_tr = 1/(365*24*60*60.)            # sec to yr
R_tr = 1/(1.496e11)                 # m to AU
M_tr = 1/(6e24)                     # kg to Mass of Earth
G_si = 6.673e-11                    # SI Gravitational Constant
G = G_si*(R_tr**3)/(M_tr*T_tr**2)   # translated Gravitational Constant


# Variables 1 . __ { 1 = time unit (yr) } { h = time step (yr) , h*t_I = tf-ti }

ti = 0                              # initial time = 0
tf = 120                            # final time = 120 years
t_I = 100*tf                        # 100 points per year

t = np.linspace(ti,tf,t_I)          # time array from ti to tf with N points 
h = t[2]-t[1]                 


# Variables 2 . __ { 1 = time unit (yr) , 1 = length unit (AU) , 1 = mass unit (MoE) }

Agent = []
a_N = 4                                                             # __ New Agent: 3 > 4 __ #
D = 2

M_i = np.zeros(a_N)

M_i[0] = 1                                 # adjustable (Me)
M_i[1] = 60                                # adjustable (Mj)
M_i[2] = 100                               # adjustable (Ms)
M_i[3] = 30                                                       # __ New Agent! __ #

M_i = M_i*1000                             # adjustable (mag)

r_i = np.zeros([D,a_N])

r_i[:,0] = [0,1]                           # adjustable (re)
r_i[:,1] = [1,-1]                          # adjustable (rj)
r_i[:,2] = [-1,-1]                         # adjustable (rs)
r_i[:,3] = [0,-3]                                                 # __ New Agent! __ #
    
    
Mc = 0
for n in range(a_N): Mc = Mc + M_i[n]

rc = np.zeros(D)
for n in range(a_N): rc = rc + M_i[n]*r_i[:,n]
rc = rc/Mc



# rot, l

def rot(v, theta):
    m = [ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ]
    return np.matmul(m,v)

def l(v):
    return np.linalg.norm(v)

# acc

def acc(r0, i, id):
    
    r = np.zeros([D,a_N])
    
    for n in range(a_N):
        r[:,n] = r0 - Agent[n]['r'][i,:] if id != n else np.zeros(2)
    
    a_t = np.zeros(D)
    
    for n in range(a_N):
        a = np.zeros(2) if not r[:,n].any() else -r[:,n]/l(r[:,n]) * G*Agent[n]['M']/l(r[:,n])**2 
        a_t = a_t + a

    return a_t


# Initialize 'M' , 'r'

for n in range(a_N):
    Agent.append({'M':0, 'r':np.zeros([t_I,D]), 'v':np.zeros([t_I,D])})
    Agent[n]['M'] = M_i[n]
    Agent[n]['r'][0,:] = r_i[:,n]


# Initialize 'v'

for n in range(a_N):
    Agent[n]['v'][0,:] = rot(r_i[:,n]-rc,np.pi/2)/l(r_i[:,n]-rc) * np.sqrt(l(acc(r_i[:,n],0,n))*l(r_i[:,n]-rc))



# derivitive & integration

def d(r, v, i, id):
    return { 'dr_dt':v , 'dv_dt':acc(r, i, id) } 

def RK4Solver(i, id):
    
    r = Agent[id]['r'][i,:]
    v = Agent[id]['v'][i,:]
    
    k1 = d( r , v , i ,id )
    k2 = d( r + k1['dr_dt']*h/2 , v + k1['dv_dt']*h/2 , i , id )
    k3 = d( r + k2['dr_dt']*h/2 , v + k2['dv_dt']*h/2 , i , id )
    k4 = d( r + k3['dr_dt']*h , v + k3['dv_dt']*h , i , id )

    Kr_h = h * ( k1['dr_dt'] + 2*k2['dr_dt'] + 2*k3['dr_dt'] + k4['dr_dt'] )/6
    Kv_h = h * ( k1['dv_dt'] + 2*k2['dv_dt'] + 2*k3['dv_dt'] + k4['dv_dt'] )/6

    return np.array([ r+Kr_h , v+Kv_h ])


# Main Loop . __ { i = time counter , n = agent counter }

for i in range(t_I-1):

    for n in range(a_N):

        [ Agent[n]['r'][i+1,:] , Agent[n]['v'][i+1,:] ] = RK4Solver(i, n)   

print(Agent[0])