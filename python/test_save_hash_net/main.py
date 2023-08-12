from utils import *
import RL_n_Game.Animation as Ani

from RL_n_Game.Coach import Coach
from RL_n_Game.Game import Game
from RL_n_Game.NeuralNet import NeuralNet

import os
import numpy as np


args = dotdict({

    # Episode & MCTS Settings

    'numIters': 1,
    'numEps': 1,                      # Number of complete self-play games to simulate during a new iteration.

    'time':100,
    'effect_duration':1,
    'numMCTSSims': 10,        # (10)  " predict up to: 10steps (=0.1yr) "                               
    'cpuct':10,               # (100) " c*P/(1+k*N) range: 10 ~ 100 "
    'k':0,                    # (0)   " breadth/depth first: 0/1 "
    
    'hash_shift':9,           # (0)    { shuffle } " range: 0~10 "

    # Game Settings

    'r':1.3,                    # (4)    { shuffle }
    'th':0,                   # (0)
    'init_v':(7,np.pi/2),     # (0,0)  { shuffle }

    'str':150,                 # (300)  " too-strong: unable to react, 
                              #          too-weak: unable to predict "
    'Ba':0.7,
    'Bc':3,

    # Animation Settings

    'folder_n':4,
    'sim_n':6,
    'iter_n':0,
    'max_frames':700,


    # Others

    'tempThreshold': 15,        # temp < tempThreshold: 1, temp > tempThreshold: 0
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20

})


def main():

    # Init
    G = Game(args.str, args.time, args.Ba, args.Bc)
    NNet = NeuralNet(G)
    C = Coach(G, NNet, args)

    # Start Simulation
    C.learn()
    Animate(C)


def Animate(C):
    

    Ani.t = args.time
    Ani.State_History = C.State_History[args.iter_n]

    Ani.Ba = args.Ba
    Ani.Bc = args.Bc

    folder_path = '/home/erathyx/jupyter-notebook/result_'+f'{args.folder_n}'
    os.system('mkdir '+folder_path)

    Ani.plot_settings()
    Ani.ani_to_video( folder_path+'/sim_'+f'{args.sim_n}'+'.mp4', args.max_frames )


if __name__ == "__main__":
    main()