### __ Numerical Integration __ ###

import numpy as np
        # use "np.array([])" rather than "[]"


# Variables 1 . __ { 1 = time unit (yr) } { h = time step (yr) , h*t_I = tf-ti }

ti = 0                              # initial time = 0
tf = 120                            # final time = 120 years
t_I = 100*tf                        # 100 points per year

t = np.linspace(ti,tf,t_I)          # time array from ti to tf with N points 
h = t[2]-t[1]                 


# Variables 2 . __ { 1 = time unit (yr) , 1 = length unit (AU) , 1 = mass unit (MoE) }

Agent = []
a_N = 3                                                           # __ New Agent: 3 > 4 __ #
D = 2


r_i = []
for n in range(a_N): r_i.append(np.zeros(D)) 

r_i[0] = [0,1]                           # adjustable (re)
r_i[1] = [1,-1]                          # adjustable (rj)
r_i[2] = [-1,-1]                         # adjustable (rs)
#r_i[3] = [0,-3]                                                 # __ New Agent! __ #

E_i = []
for n in range(a_N): E_i.append(np.zeros([t_I, a_N])) 

for id in range(a_N):
    for i in range(t_I):
        for n in range(a_N):
            rd_n = np.random.randint(3) - 1
            E_i[id][i,n] = rd_n if n != id else 0

S = 7



# Initialize 'r', 'E', 'S'

for n in range(a_N):
    Agent.append({ 'r':np.zeros([t_I, D]), 'v':np.zeros([t_I, D]), 'E':np.zeros([t_I, a_N]), 'S':0 })
    Agent[n]['r'][0,:] = r_i[n]
    Agent[n]['E'] = E_i[n]
    Agent[n]['S'] = S



# rot, l

def rot(v, theta):
    m = [ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ]
    return np.matmul(m,v)

def l(v):
    return np.linalg.norm(v)

# acc

def acc(r0, i, id):
    
    a_t = np.zeros(D)
    
    for n in range(a_N):
        
        r = Agent[n]['r'][i,:] - r0 if id != n else np.zeros(2)
        
        a1 = np.zeros(2) if not r.any() else r/l(r) *Agent[n]['E'][i,id] *Agent[n]['S']
        a2 = np.zeros(2) if not r.any() else r/l(r) *Agent[id]['E'][i,n] *Agent[id]['S']
        
        a_t = a_t + a1 + a2

    return a_t



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

