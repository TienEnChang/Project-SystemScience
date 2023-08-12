import numpy as np

class Game():

    def __init__(self, S=10, time_N=400, B_a=0.5, B_c=5):
        self.agent_N = 3                         ##### __ New Agent! __
        self.D = 2                               #####
        
        self.E_Strength = S

        self.action_N_r = 3                                 # restricted to -1,0,1
        self.action_N_s = self.action_N_r**(self.agent_N-1) # action per single agent
        self.action_N_t = self.action_N_s**(self.agent_N)   # total possible action

        self.B_c = B_c
        self.B_a = B_a

        self.time_N = time_N
        self.t_init()                            #####

    def t_init(self):
        ti = 0                              
        tf = self.time_N/100                             
        self.t = np.linspace(ti,tf,self.time_N)           
        self.h = 0.01

    

    ## __ Numerical Integration __ ##

    def rot(self, v, theta):
        m = [ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ]
        return np.matmul(m,v)
    
    def l(self, v):
        return np.linalg.norm(v)

    def acc(self, r0, State, Effect, id):
        
        a_t = np.zeros(self.D)
        
        for n in range(self.agent_N):

            if self.is_Alive( State, n ):
                
                r = State[n]['r'] - r0 if id != n else np.zeros(self.D)
                
                a1 = np.zeros(self.D) if not r.any() else r/self.l(r) * Effect[n]['+/-'][id] * Effect[n]['s/w']
                a2 = np.zeros(self.D) if not r.any() else r/self.l(r) * Effect[id]['+/-'][n] * Effect[id]['s/w']
                
                a_t += (a1 + a2)

        return a_t

    def d(self, r, v, State, Effect, id):
        return { 'dr_dt':v , 'dv_dt':self.acc(r, State, Effect, id) } 

    def RK4Solver(self, State, Effect, id):
        
        r = State[id]['r']
        v = State[id]['v']
        
        k1 = self.d( r , v , State , Effect , id)
        k2 = self.d( r + k1['dr_dt']*self.h/2 , v + k1['dv_dt']*self.h/2 , State , Effect , id)
        k3 = self.d( r + k2['dr_dt']*self.h/2 , v + k2['dv_dt']*self.h/2 , State , Effect , id)
        k4 = self.d( r + k3['dr_dt']*self.h , v + k3['dv_dt']*self.h , State , Effect , id)

        Kr_h = self.h * ( k1['dr_dt'] + 2*k2['dr_dt'] + 2*k3['dr_dt'] + k4['dr_dt'] )/6
        Kv_h = self.h * ( k1['dv_dt'] + 2*k2['dv_dt'] + 2*k3['dv_dt'] + k4['dv_dt'] )/6

        return { 'r':r+Kr_h, 'v':v+Kv_h, 'alive/dead':1 }


    def getNextState(self, State, Effect):

        next_State = self.getNullState()

        for n in range(self.agent_N):

            if self.is_Alive( State, n ):
                next_State[n] = self.RK4Solver( State, Effect, n )
            else:
                next_State[n] = State[n]

        return self.OB_to_Dead(next_State)



    ## __ State & Effect __ ##

    def getNullState(self):

        State = []
        for n in range(self.agent_N): 
            State.append({ 'r':np.zeros(self.D), 'v':np.zeros(self.D), 'alive/dead':1 })

        return State

    def getInitState(self, r=1, th=0, init_v=(0,0)): ###

        State = self.getNullState()

        State[0]['r'] = self.rot( np.array([ r, 0 ]), th )
        State[1]['r'] = self.rot( np.array([ -r/2, r*np.sqrt(3)/2 ]), th )
        State[2]['r'] = self.rot( np.array([ -r/2 , -r*np.sqrt(3)/2 ]), th )

        for n in range(self.agent_N):
            State[n]['v'] = init_v[0]* self.rot( State[n]['r'], init_v[1] )
        
        return State


    def State_POV_Transform(self, State , pov):

        trans_State = self.getNullState()
        th0 = self.r_to_th( State[pov]['r'] )
        
        for n in range(self.agent_N): 

            trans_State[n]['r'] = self.rot( State[self.mod(pov+n)]['r'] , -th0 )
            trans_State[n]['v'] = self.rot( State[self.mod(pov+n)]['v'] , -th0 )

            trans_State[n]['alive/dead'] = State[self.mod(pov+n)]['alive/dead']
                
        return trans_State


    def getNullEffect(self):

        Effect = []
        for n in range(self.agent_N): 
            Effect.append({ '+/-':np.zeros(self.agent_N), 's/w':self.E_Strength })

        return Effect


    def Action_to_Effect(self, Action, id):
    
        Effect_s = { '+/-':np.zeros(self.agent_N), 's/w':self.E_Strength }
        
        Effect_s['+/-'][self.next(id)] = Action%3 -1
        Effect_s['+/-'][self.prev(id)] = Action//3 -1

        return Effect_s


    def Pi_to_Effect(self, pi, id):

        Effect_s = { '+/-':np.zeros(self.agent_N), 's/w':self.E_Strength }

        bestA = 0
        for n in range(len(pi)):
            if pi[n] == np.max(pi):
                bestA = n
                break

        directness = np.sum([ (pi[n]-sum(pi)/len(pi))**2 for n in range(len(pi)) ])*len(pi)/(len(pi)-1)
        boost = 1-directness

        if pi.any():
            Effect_s = self.Action_to_Effect(bestA, id)
            Effect_s['s/w'] = self.E_Strength*(1+boost*10)

        ### PRINT ###
        print('pi: ',pi)
        print('bestA/boost: ',bestA,',',boost)
            
        return Effect_s


    def mod(self,n):
        return np.mod( n, self.agent_N )

    def next(self, n):
        return np.mod( n+1, self.agent_N )

    def prev(self, n):
        return np.mod( n-1, self.agent_N )

    def r_to_th(self, r=[1,0]):
        th = np.arctan2( r[1], r[0] )
        if r[1] < 0: th -= 2*np.pi
        return th


    def round(self, State):

        r_State = self.getNullState()

        for n in range(self.agent_N):
            r_State[n]['r'] = np.array([ round(State[n]['r'][0], 6), round(State[n]['r'][1], 6) ])
            r_State[n]['v'] = np.array([ round(State[n]['v'][0], 6), round(State[n]['v'][1], 6) ])
            r_State[n]['alive/dead'] = State[n]['alive/dead']
            
        return r_State


    def getSymmetries(self, State, pi):
        
        sym_State = self.getNullState()

        sym_State[0] = State[0]
        sym_State[1] = State[2]
        sym_State[2] = State[1]

        sym_pi = np.reshape(np.reshape(pi, (3,3)).transpose(), (1,9)).flatten()

        return [ [State, pi] , [sym_State, sym_pi] ]



    ## __ OB & Game Result __ ##

    def is_Alive(self, State, id, check_OB='no'):
        
        if check_OB == 'yes':

            r_c = self.l( State[id]['r'] )
            r_n = 5 if State[self.next(id)]['alive/dead'] == 0 else self.l( State[self.next(id)]['r'] - State[id]['r'] )
            r_p = 5 if State[self.prev(id)]['alive/dead'] == 0 else self.l( State[self.prev(id)]['r'] - State[id]['r'] )

            if r_c > self.B_c or r_n < self.B_a or r_p < self.B_a:
                return False
            else:
                return True

        if State[id]['alive/dead'] == 1:
            return True
        else:
            return False


    def OB_to_Dead(self, State):

        for n in range(self.agent_N):
            if not self.is_Alive( State , n ,check_OB='yes'):
                State[n]['alive/dead'] = 0

        return State


    def getGameResult(self, State):

        r = []
        for n in range(self.agent_N):
            r.append( int(self.is_Alive( State, n )) )

        alive_N = 0
        for n in range(self.agent_N):
            if r[n] == 1: 
                alive_N += 1
            if r[n] == 0:
                r[n] = -1

        if alive_N > 1: 
            return np.zeros(self.agent_N)
        else:
            return np.array(r)


    def Death_time(self, State_History):
        Dt = np.zeros(self.agent_N)
        for n in range(self.agent_N):
            for i in range(len(State_History)):
                if State_History[i][n]['alive/dead'] == 0 and State_History[i-1][n]['alive/dead'] == 1:
                    Dt[n] = i
                    break
        return Dt


    ## __ Others __ ##
          
    def to_reversed_Base3(self, Num, digit_n):
    
        rvB3_Num=[]
        for n in range(digit_n):
            rvB3_Num.append( (Num%3**(n+1))//3**n )

        return rvB3_Num
