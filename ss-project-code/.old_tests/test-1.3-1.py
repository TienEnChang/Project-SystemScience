### __ Numerical Integration __ ###

import numpy as np

        # use "ndarray" rather than "list"

# acc

def acc(r0, i, id):
    
    r1 = r0 - sun['r'][i,:] 
    r2 = r0 - earth['r'][i,:] if id != 'e' else np.zeros(2)
    r3 = r0 - jupiter['r'][i,:] if id != 'j' else np.zeros(2)

    a1 = np.zeros(2) if not r1.any() else -r1/np.linalg.norm(r1) * G*Ms/np.linalg.norm(r1)**2 
    a2 = np.zeros(2) if not r2.any() else -r2/np.linalg.norm(r2) * G*Me/np.linalg.norm(r2)**2
    a3 = np.zeros(2) if not r3.any() else -r3/np.linalg.norm(r3) * G*Mj/np.linalg.norm(r3)**2
    
    return a1+a2+a3

# derivitive & integration

def d(r, v, i, id):
    return { 'dr_dt':v , 'dv_dt':acc(r, i, id) } 

def RK4Solver(info, i):

    id = 'n'
    if info is earth: id = 'e'
    if info is jupiter: id = 'j'
    
    r = info['r'][i,:]
    v = info['v'][i,:]
    
    k1 = d( r , v , i ,id )
    k2 = d( r + k1['dr_dt']*h/2 , v + k1['dv_dt']*h/2 , i , id )
    k3 = d( r + k2['dr_dt']*h/2 , v + k2['dv_dt']*h/2 , i , id )
    k4 = d( r + k3['dr_dt']*h , v + k3['dv_dt']*h , i , id )

    Kr_h = h * ( k1['dr_dt'] + 2*k2['dr_dt'] + 2*k3['dr_dt'] + k4['dr_dt'] )/6
    Kv_h = h * ( k1['dv_dt'] + 2*k2['dv_dt'] + 2*k3['dv_dt'] + k4['dv_dt'] )/6

    return np.array([ r+Kr_h , v+Kv_h ])


# Constants . __ { 1 = time unit (yr) , 1 = length unit (AU) , 1 = mass unit (MoE) }

T_tr = 1/(365*24*60*60.)            # sec to yr
R_tr = 1/(1.496e11)                 # m to AU
M_tr = 1/(6e24)                     # kg to Mass of Earth

G_si = 6.673e-11                    # SI Gravitational Constant
G = G_si*(R_tr**3)/(M_tr*T_tr**2)   # translated Gravitational Constant

Ms = 2e30*M_tr
Me = 6e24*M_tr
Mj = 500*1.9e27*M_tr                # Mj = Super Jupiter

# Variables 1 . __ { 1 = time unit (yr) } { h = time step (yr) , h*N = tf-ti }

ti = 0                        # initial time = 0
tf = 120                      # final time = 120 years
N = 100*tf                    # 100 points per year

t = np.linspace(ti,tf,N)      # time array from ti to tf with N points 
h = t[2]-t[1]                 

# Variables 2 . __ { 1 = time unit (yr) , 1 = length unit (AU) }

earth = { 'r':np.zeros([N,2]) , 'v':np.zeros([N,2]) }                 
jupiter = { 'r':np.zeros([N,2]) , 'v':np.zeros([N,2]) }                      
sun = { 'r':np.zeros([N,2]) , 'v':np.zeros([N,2]) }                      

re_i = [1,0]                                     # initial position of earth
rj_i = [5,0]                                     # initial position of Jupiter

ve_i = [0, np.sqrt(G*Ms/re_i[0])]                # Initial velocity of Earth (|- to r)
vj_i = [0, np.sqrt(G*Ms/rj_i[0])]                # Initial velocity of Jupiter (|- to r)

earth['r'][0,:] = re_i
earth['v'][0,:] = ve_i

jupiter['r'][0,:] = rj_i
jupiter['v'][0,:] = vj_i


# Main Loop . __ { i = time counter }

for i in range(N-1):

    [ earth['r'][i+1,:] , earth['v'][i+1,:] ] = RK4Solver(earth, i)   
    [ jupiter['r'][i+1,:] , jupiter['v'][i+1,:] ] = RK4Solver(jupiter, i)