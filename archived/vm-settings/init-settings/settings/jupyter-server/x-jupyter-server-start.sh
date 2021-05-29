#!/bin/bash

screen -d -m -S jupyter bash -c 'cd ~/jupyter-notebook && jupyter notebook --ip 10.140.0.2'

#jupyter notebook --ip 10.138.0.5

#ctrl a+d

#screen -r
#screen -ls
#screen -X -S [session # you want to kill] kill

#yes \n | screen

