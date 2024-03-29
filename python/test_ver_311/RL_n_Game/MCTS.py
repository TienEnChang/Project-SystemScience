import numpy as np
import logging

log = logging.getLogger(__name__)

class MCTS():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        self.Qxsa = {} 
        self.Nxsa = {}
        

    def getActionProb(self, root_State, temp=1):

        # ignore dead State
        if not self.game.is_Alive( root_State, 0 ): return np.zeros(self.game.action_N_s)  
            
        # Initialize with root_State

        self.Pxsa = {} 
        self.Vxs = {}           # maybe not needed? try to use normal_value as Vxs_last
        self.Exs = {}

        self.dead_value = -100  # -1e-25                                                   
        self.normal_value = -1e-25   # v!=0                                      

        self.root_index = (0,0)                                                           
        self.State_Block_list = { (0,0):self.get_New_State_Block((0,0), root_State, []) } 

        # perform MCTS

        for i in range(self.args.numMCTSSims):      
            self.search( self.root_index )

        # N(s,a) to Pi(s,a)
        ''' temp need to be adjusted according to "probability of death" '''
        ''' search value drops below a resignation threshold > game ends '''

        if temp == 0: temp = 1/float('inf')

        count_list = np.array([ self.Nxsa[ self.root_index ][(0, a)] ** (1. / temp) if (0, a) in self.Nxsa[ self.root_index ] else 0 for a in range(self.game.action_N_s) ])
        count_sum = float(np.sum(count_list))
        pi = count_list/count_sum
        return pi


    def search(self, x):

        # Return New / Dead Vxs[x]

        check = self.check_State_Block(x)

        ### PRINT ###
        print(x, check)

        if check == 'Expanded':
            return self.Vxs[x]
        elif check == 'Dead':
            self.Vxs[x] *= 10
            return self.Vxs[x]
        elif check == 'Unreachable':
            return []


        # Select Action

        As = [0]*self.game.agent_N

        for s in range(self.game.agent_N):

            cur_best = -float('inf')
            for a in range(self.game.action_N_s):

                u = self.args.cpuct * self.Pxsa[x][s][a] 

                if x in self.Nxsa:
                    if (s, a) in self.Nxsa[x]:
                        u = self.Qxsa[x][(s, a)] + self.args.cpuct * self.Pxsa[x][s][a] / (1 + self.Nxsa[x][(s, a)]*self.args.k)
                   
                if u > cur_best:
                    cur_best = u
                    As[s] = a

            ### PRINT ###
            if x in self.Nxsa:
                if (s, As[s]) in self.Nxsa[x]:
                    print(As[s],self.game.to_reversed_Base3(As[s], 2),'/',self.Qxsa[x][(s, As[s])],self.Nxsa[x][(s, As[s])],cur_best)
                else:
                    print(As[s],self.game.to_reversed_Base3(As[s], 2),'/')
            else:
                print(As[s],self.game.to_reversed_Base3(As[s], 2),'/')
                
        next_x = self.get_next_index(x, As)


        # Simulate Action

        Vxs_last = self.search(next_x)


        # Back Propagate Vxs_last & Update Qxsa[x][s][a], Nxsa[x][s][a]


        for s in range(self.game.agent_N):

            if x not in self.Nxsa:
                self.Qxsa[x] = {}
                self.Nxsa[x] = {}
            
            if x in self.Nxsa:
                if (s, As[s]) in self.Nxsa[x]:
                    self.Qxsa[x][(s, As[s])] = (self.Qxsa[x][(s, As[s])] * self.Nxsa[x][(s, As[s])] + Vxs_last[s]) / (self.Nxsa[x][(s, As[s])] + 1)
                    self.Nxsa[x][(s, As[s])] += 1
                else:
                    self.Qxsa[x][(s, As[s])] = Vxs_last[s]
                    self.Nxsa[x][(s, As[s])] = 1

        
            ### PRINT ###
            #print(Vxs_last[s],'/',self.Qxsa[x][s][As[s]],self.Nxsa[x][s][As[s]])

        return Vxs_last



    ## __ State Block & Index Operation __ ## 

    def get_New_State_Block(self, index, State, As):
        
        new_State_Block = []

        if index != self.root_index and As != []:

            Effect = self.game.getNullEffect()
            for s in range(self.game.agent_N):
                Effect[s] = self.game.Action_to_Effect( As[s], s )
            
            State = self.game.getNextState( State, Effect )
            

        self.Vxs[index] = np.zeros(self.game.agent_N)
        self.Exs[index] = np.zeros(self.game.agent_N)

        for n in range(self.game.agent_N):

            new_State_Block.append( self.game.round( self.game.State_POV_Transform( State, n ) ) )    

            [ self.Pxsa[index][n], self.Vxs[index][n] ] = self.float_hashing( new_State_Block[n] )            
            ''' Pxsa need to be normalized! '''

            if not self.game.is_Alive( new_State_Block[n], 0 ): 
                self.Vxs[index][n] = self.dead_value
                self.Exs[index][n] = 1
                

        return new_State_Block


    def check_State_Block(self, index):

        if index in self.State_Block_list:
            if self.Exs[index][0] == 1:
                return 'Dead'
            else:
                return 'Pass'

        elif self.get_prev_index(index) in self.State_Block_list:
            
            prev_State = self.State_Block_list[ self.get_prev_index(index) ][0]
            self.State_Block_list[ index ] = self.get_New_State_Block( index, prev_State, self.get_prev_action(index) )

            return 'Expanded'

        else:
            return 'Unreachable'


    def get_next_index(self, index, As):
        At = 0
        for s in range(self.game.agent_N):
            At += As[s]*3**(2*s)

        return ( index[0]+1, At + index[1]*self.game.action_N_t )

    def get_prev_index(self, index):
        return ( index[0]-1, index[1]//self.game.action_N_t )
    
    def get_prev_action(self, index):
        At = index[1]%self.game.action_N_t
        As = np.zeros(self.game.agent_N)
        for s in range(self.game.agent_N):
            As[s] = (At%3**(2*(s+1)))//3**(2*s)
        
        return As 


    ## __ Float Hashing __ ##
    
    def float_hashing(self, State):

        hash_list = []
        hash_mix = 1

        for n in range(3):
            for opt in ['r','v']:
                for i in range(2):
                    str_h = str(hash(round(State[n][opt][i]**2, 3)+0.123))+'192837465'+'192837465'
                    h = np.array([ int(str_h[l]) for l in range(self.args.hash_shift,self.args.hash_shift+self.game.action_N_s) ])
                    hash_list.append(h)

        for n in range(len(hash_list)):
            hash_mix *= (hash_list[n]+np.ones(self.game.action_N_s,np.int64))
        

        return [ hash_mix/np.sum(hash_mix), self.normal_value ]
