import numpy as np
import big2Game
import gameLogic
import enumerateOptions
from PPONetwork import PPONetwork, PPOModel
import tensorflow as tf
import joblib

tf.compat.v1.disable_eager_execution()

mainGame = big2Game.big2Game()

inDim = 412
outDim = 1695
entCoef = 0.01
valCoef = 0.5
maxGradNorm = 0.5
sess = tf.compat.v1.Session()
#networks for players
playerNetworks = {}
playerNetworks[1] = PPONetwork(sess, inDim, outDim, "p1Net")
playerNetworks[2] = PPONetwork(sess, inDim, outDim, "p2Net")
playerNetworks[3] = PPONetwork(sess, inDim, outDim, "p3Net")
playerNetworks[4] = PPONetwork(sess, inDim, outDim, "p4Net")
playerModels = {}
playerModels[1] = PPOModel(sess, playerNetworks[1], inDim, outDim, entCoef, valCoef, maxGradNorm)
playerModels[2] = PPOModel(sess, playerNetworks[2], inDim, outDim, entCoef, valCoef, maxGradNorm)
playerModels[3] = PPOModel(sess, playerNetworks[3], inDim, outDim, entCoef, valCoef, maxGradNorm)
playerModels[4] = PPOModel(sess, playerNetworks[4], inDim, outDim, entCoef, valCoef, maxGradNorm)

tf.compat.v1.global_variables_initializer().run(session=sess)

#by default load current best
params = joblib.load("modelParameters136500")
playerNetworks[1].loadParams(params)
playerNetworks[2].loadParams(params)
playerNetworks[3].loadParams(params)
playerNetworks[4].loadParams(params)

currSampledOption = -1

def sampleFromNetwork():
    global currSampledOption
    
    go, state, actions = mainGame.getCurrentState()
    (a, v, nlp) = playerNetworks[go].step(state, actions)
    currSampledOption = a[0]

def playSampledOption():
    global currSampledOption
    if currSampledOption == -1:
        return
    else:
        mainGame.step(currSampledOption)

def playout():
    wins = [0,0,0,0]
    score = [0.0, 0.0, 0.0, 0.0]
    for i in range(2000):
        if i % 50 == 0:
            print(i)
        done = False
        while not done:
            sampleFromNetwork()
        
            global currSampledOption
            if currSampledOption == -1:
                return
            else:
                reward, done, info = mainGame.step(currSampledOption)
                if done:
                    for i in range(4):
                        score[i] += reward[i]
                        if reward[i] > 0:
                            wins[i] += 1
    
    print(score)
    print(wins)

playout()