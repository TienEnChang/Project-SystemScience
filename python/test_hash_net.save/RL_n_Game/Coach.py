import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler  
from random import shuffle             

import numpy as np
from tqdm import tqdm                  

#from .Arena import Arena
from .MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.State_History = []
        self.Time_Stamp = []


    def executeEpisode(self):

        Dicision = []
        omni_State = self.game.getInitState(self.args.r, self.args.th, self.args.init_v)
        Effect = self.game.getNullEffect()
        State_History = [ omni_State ]
        Result = [0,0,0]
        
        for time in range(self.game.time_N-1):
            
            for curPlayer in range(self.game.agent_N):
        
                rel_State = self.game.State_POV_Transform(omni_State, curPlayer)
                
                mcts = MCTS(self.game, self.nnet, self.args)
                pi = mcts.getActionProb(rel_State)                
                sym = self.game.getSymmetries(rel_State, pi)
                
                for s, p in sym:
                    Dicision.append([s, p, curPlayer])

                if self.game.is_Alive( rel_State, 0 ):
                    Effect[curPlayer] = self.game.Pi_to_Effect(pi, curPlayer)

                ### PRINT ###
                print('curPlayer: ',curPlayer) 
                print('...')
                

            for _ in range(self.args.effect_duration):
                    
                omni_State = self.game.getNextState(omni_State, Effect)
                State_History.append( omni_State )
                
                r = self.game.getGameResult( omni_State )
                if 1 in r: Result = r

                if all( n == -1 for n in r ):

                    Dicision_w_Result = [ ( x[0], x[1], Result[x[2]] ) for x in Dicision ]
                    return [ Dicision_w_Result , State_History ] 

        Dicision_w_Result = [ ( x[0], x[1], 1 ) for x in Dicision ]
        return [ Dicision_w_Result , State_History ] 



    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterTrainExamples = [] 
                
                for i in tqdm(range(self.args.numEps), desc="Self Play"):
                    [ trainExamples , State_History ] = self.executeEpisode()
                    iterTrainExamples.extend( trainExamples )
                    self.State_History.append( State_History )

                    ### PRINT ###
                    print( self.game.Death_time(State_History) )

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            
            #self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
