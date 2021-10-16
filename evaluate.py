import numpy as np
import big2GameEvaluation
import gameLogic
import enumerateOptions
#from PPONetworkNew import PPONetwork, PPOModel
from PPONetwork import PPONetwork, PPOModel
import tensorflow as tf
import joblib

tf.compat.v1.disable_eager_execution()

mainGame = big2GameEvaluation.big2Game()

inDim = 412
outDim = 1695
entCoef = 0.01
valCoef = 0.5
maxGradNorm = 0.5
sess = tf.compat.v1.Session()
#networks for players
playerNetworks = {}
# playerNetworks[1] = PPONetwork( inDim, outDim, "p1Net")
# playerNetworks[2] = PPONetwork(inDim, outDim, "p2Net")
# playerNetworks[3] = PPONetwork(inDim, outDim, "p3Net")
# playerNetworks[4] = PPONetwork(inDim, outDim, "p4Net")
playerNetworks[1] = PPONetwork(sess, 412, outDim, "p1Net")
playerNetworks[2] = PPONetwork(sess,412, outDim, "p2Net")
playerNetworks[3] = PPONetwork(sess,440, outDim, "p3Net")
playerNetworks[4] = PPONetwork(sess,440, outDim, "p4Net")

#tf.compat.v1.global_variables_initializer().run(session=sess)

#by default load current best
params = joblib.load("baselineParameters90000")
playerNetworks[1].loadParams(params)
playerNetworks[2].loadParams(params)
params = joblib.load("inputV2Parameters90000")
playerNetworks[3].loadParams(params)
playerNetworks[4].loadParams(params)

currSampledOption = -1

def sampleFromNetwork():
    global currSampledOption
    
    go, state, actions = mainGame.getCurrentState()
    if go == 3 or go == 4:
        go, state, actions = mainGame.getCurrentStateInputV2()
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
    for i in range(20000):       
        if (i % 100) == 0:
            print("Game %d done" % (i))
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