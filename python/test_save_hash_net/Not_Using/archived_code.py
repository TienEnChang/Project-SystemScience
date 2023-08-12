import numpy as np

def to_Base3( Num, index ):
    rvB3_Num = to_reversed_Base3(Num, index)
    B3_Num = np.flipud( rvB3_Num )

    return ''.join( [ str(int(x)) for x in B3_Num ] )
        
def to_reversed_Base3(Num, index, rvB3_Num=[], firsttime=True):
    if firsttime: rvB3_Num = np.zeros(index+1)
    
    rvB3_Num[index] = Num // 3**index
    if index != 0:
        rvB3_Num = to_reversed_Base3( Num - (rvB3_Num[index]*3**index), index-1, rvB3_Num, firsttime=False)

    return rvB3_Num

def print_list_order():
    list = []
    hash_mix_cp = hash_mix/np.sum(hash_mix)   

    for _ in range(len(hash_mix)):
        max = 0
        max_n = 0
        for i in range(len(hash_mix)):
            if hash_mix_cp[i] > max:
                max = hash_mix_cp[i]
                max_n = i
        list.append(max_n)
        hash_mix_cp[max_n] = 0

    print(list)


def Abs_to_Rel(self, As, id):

    Rs = []

    for n in range(self.agent_N): 
        Rs.append({ 'r_cen':0, 'r_agn':np.zeros(self.agent_N),
                    'v_cen':0, 'v_agn':np.zeros(self.agent_N),
                    'pov':0, 'abs/rel':'rel'})

        Rs[n]['r_cen'] = self.l( -As[self.mod(id+n)]['r'] )
        Rs[n]['r_agn'][self.next(n)] = self.l( As[self.next(id+n)]['r'] - As[self.mod(id+n)]['r'] )
        Rs[n]['r_agn'][self.prev(n)] = self.l( As[self.prev(id+n)]['r'] - As[self.mod(id+n)]['r'] )
        
        Rs[n]['v_cen'] = self.l( -As[self.mod(id+n)]['v'] )
        Rs[n]['v_agn'][self.next(n)] = self.l( As[self.next(id+n)]['v'] - As[self.mod(id+n)]['v'] )
        Rs[n]['v_agn'][self.prev(n)] = self.l( As[self.prev(id+n)]['v'] - As[self.mod(id+n)]['v'] )
        
    return Rs


def Rel_to_Pseudo_Abs(self, Rs):

    As = self.getInitState()

    for i in range(2):

        str = 'r' if i == 0 else 'v'

        k0c = Rs[0][str+'_cen']
        As[0][str] = [ k0c , 0 ]
        
        k1c = Rs[1][str+'_cen']
        k01 = Rs[0][str+'_agn'][1]
        cos_th1 = (k0c**2 + k1c**2 - k01**2)/(2*k0c*k1c)
        sin_th1 = np.sqrt( 1-cos_th1**2 )
        As[1][str] = [ k1c*cos_th1 , k1c*sin_th1 ]

        k2c = Rs[2][str+'_cen']
        k02 = Rs[0][str+'_agn'][2]
        k12 = Rs[1][str+'_agn'][2]
        cos_th2 = (k0c**2 + k2c**2 - k02**2)/(2*k0c*k2c)
        sin_th2 = np.sqrt( 1-cos_th1**2 )

        if self.l( np.array([ k2c*cos_th2, k2c*sin_th2 ]) - As[1][str] ) == k12: 
            As[2][str] = [ k2c*cos_th2, k2c*sin_th2 ]
        else: 
            As[2][str] = [ k2c*cos_th2, -k2c*sin_th2 ]

    return As

def Rel_Transform(self, Rs1, nx_id):

    Rs2 = []

    for n in range(self.agent_N): 
        Rs2.append({ 'r_cen':0, 'r_agn':np.zeros(self.agent_N),
                        'v_cen':0, 'v_agn':np.zeros(self.agent_N), 
                        'pov':0, 'abs/rel':'rel'})

        Rs2[n] = Rs1[self.mod(nx_id+n)]
        Rs2[n]['pov'] += nx_id

    return Rs2

def Rel_Restore(self, Rs):
    return Rel_Transform( Rs, -Rs[0]['pov'] )


def getRandomInitState(self,):

    State = self.getNullState()
    for n in range(self.agent_N):
        r = np.random.uniform(0,2)
        th = np.random.uniform(-np.pi,np.pi)
        State[n]['r'] = np.array([ r*np.cos(th) , r*np.sin(th) ])
        State[n]['v'] = self.rot( State[n]['r'] , np.pi/2 )

    return State


def getRandomEffect(self):

    Effect = self.getNullEffect()

    for id in range(self.agent_N):
        for n in range(self.agent_N): 
            Effect[id]['+/-'][n] = np.random.randint(3) - 1 if n != id else 0
        Effect[id]['s/w'] = 4

    return Effect


def Pi_to_Effect_old(self, pi, id):

    Effect_s = { '+/-':np.zeros(self.agent_N), 's/w':self.E_Strength }

    exp = np.sum(np.array([ (n+1)*pi[n] for n in range(len(pi)) ]))
    var = np.sum(np.array([ (((n+1)-exp)**2)*pi[n] for n in range(len(pi)) ]))
    boost = 2*np.sqrt(var)/(len(pi)-1)

    if exp != 0:
        Effect_s = self.Action_to_Effect((int(exp)-1), id)
        Effect_s['s/w'] = self.E_Strength*(1+boost*50)

        ### PRINT ###
        #print('E_strength: ',Effect_s['s/w'])
        
    return Effect_s


def list_collapse(self, list_t):
    a_s = self.game.action_N_s
    return np.sum( np.reshape( list_t, (a_s**2, a_s) ), axis=0 )

def list_combine(self, list_t):
    a_s = self.game.action_N_s

    list_01 = np.reshape( (np.tile(list_t[0], (a_s,1)).transpose()*list_t[1]).transpose(), (1,a_s**2) ).flatten()
    list_012 = np.reshape( (np.tile(list_01, (a_s,1)).transpose()*list_t[2]).transpose(), (1,a_s**3) ).flatten()

    return list_012/np.sum(list_012)

    ### / ### / ### / ### / ### / ### / ### / ### / ###   [0]
        #  ,  #  ,  #  /  #  ,  #  ,  #  /  #  ,  #  ,  #    [1]
            #        ,        #        ,        #          [2]


print(len(str_h))

