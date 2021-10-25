from re import A
import numpy as np
import big2GameEvaluation
import big2Game
import gameLogic
import enumerateOptions
#from PPONetworkNew import PPONetwork, PPOModel
from PPONetwork import PPONetwork, PPOModel
import tensorflow as tf
import joblib

tf.compat.v1.disable_eager_execution()

#mainGame = big2GameEvaluation.big2Game()

inDim = 412
outDim = 1695
entCoef = 0.01
valCoef = 0.5
maxGradNorm = 0.5
#sess = tf.compat.v1.Session()
#networks for players
playerNetworks = {}
# playerNetworks[1] = PPONetwork( inDim, outDim, "p1Net")
# playerNetworks[2] = PPONetwork(inDim, outDim, "p2Net")
# playerNetworks[3] = PPONetwork(inDim, outDim, "p3Net")
# playerNetworks[4] = PPONetwork(inDim, outDim, "p4Net")
# playerNetworks[1] = PPONetwork(sess, 412, outDim, "p1Net")
# playerNetworks[2] = PPONetwork(sess,412, outDim, "p2Net")
# playerNetworks[3] = PPONetwork(sess,440, outDim, "p3Net")
# playerNetworks[4] = PPONetwork(sess,440, outDim, "p4Net")

#tf.compat.v1.global_variables_initializer().run(session=sess)

#by default load current best
# params = joblib.load("baselineParameters90000")
# playerNetworks[1].loadParams(params)
# playerNetworks[2].loadParams(params)
# params = joblib.load("inputV2Parameters90000")
# playerNetworks[3].loadParams(params)
# playerNetworks[4].loadParams(params)

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

class MultiNetworkEvaluator(object):

    def __init__(self, sess, nGames):
        self.playerNetworks = {}
        self.playerNetworks[1] = PPONetwork(sess, 440, outDim, "p1Net")
        self.playerNetworks[2] = PPONetwork(sess,412, outDim, "p2Net")
        self.playerNetworks[3] = PPONetwork(sess,412, outDim, "p3Net")
        self.playerNetworks[4] = PPONetwork(sess,412, outDim, "p4Net")

        params = joblib.load("inputV2Parameters90000")
        self.playerNetworks[1].loadParams(params)        
        params = joblib.load("baselineParameters90000")
        self.playerNetworks[2].loadParams(params)
        self.playerNetworks[3].loadParams(params)
        self.playerNetworks[4].loadParams(params)

        self.nGames = nGames

        self.vectorizedGame = big2GameEvaluation.vectorizedBig2Games(nGames)

        self.wins = [0, 0, 0, 0]
        self.scores = [0., 0., 0., 0.]

    def playoutVectorized(self):
        for i in range(1000):
            if i % 10 == 0:
                print('%d games complete...' % (i * self.nGames))
            self.vectorizedGame.reset()

            completed = [False for i in range(self.nGames)]
            totalCompleted = 0
            #go, state, availAcs = vectorizedGame.getCurrStates()
            #playerNums = go

            # for now, assume turn order of [modified, modified, baseline, baseline]
            turn = 2
            
            while totalCompleted < self.nGames:
                #print("TURN %d" % (turn))
                gos, states, availAcs = self.vectorizedGame.getCurrStates() if turn != 1 else self.vectorizedGame.getCurrStatesInputV2()

                selectedAcs, v, nlp = self.playerNetworks[turn].step(np.squeeze(states), np.squeeze(availAcs))
                rewards, dones, infos = self.vectorizedGame.step(selectedAcs)

                for game in range(self.nGames):
                    if dones[game] and not completed[game]:
                        completed[game] = True
                        totalCompleted += 1
                        
                        for p in range(4):
                            self.scores[p] += rewards[game][p]
                            if rewards[game][p] > 0.:
                                self.wins[p] += 1

                turn += 1
                if turn == 5:
                    turn = 1
            
        print(self.wins)
        print(self.scores)

class SingleNetworkEvaluator(object):

    def __init__(self, sess, nGames, nRounds):
        self.playerNetworks = {}
        self.testNetwork = PPONetwork(sess, 440, outDim, "p1Net")
        self.baselineNetwork = PPONetwork(sess,412, outDim, "p2Net")

        params = joblib.load("inputV2Parameters225000")
        self.testNetwork.loadParams(params)        
        params = joblib.load("originalParameters136500")
        self.baselineNetwork.loadParams(params)

        self.nGames = nGames
        self.nRounds = nRounds

        self.vectorizedGame = big2GameEvaluation.vectorizedBig2Games(nGames)

        self.wins = 0
        self.score = 0.

    def playoutVectorized(self):
        for i in range(self.nRounds):
            if i % 10 == 0:
                print('{0}% complete...'.format('{:.2f}'.format((i * 100) / self.nRounds)))

            self.vectorizedGame.reset()

            completed = [False for i in range(self.nGames)]
            totalCompleted = 0

            evaluatedNetworkTurn = i % 4
            currTurn = 1
            
            while totalCompleted < self.nGames:
                #print("TURN %d" % (turn))
                gos, states, availAcs = self.vectorizedGame.getCurrStates() if currTurn != evaluatedNetworkTurn else self.vectorizedGame.getCurrStatesInputV3()
                # gos, states, availAcs = self.vectorizedGame.getCurrStates()

                selectedAcs, v, nlp = self.baselineNetwork.step(np.squeeze(states), np.squeeze(availAcs)) if currTurn != evaluatedNetworkTurn else self.testNetwork.step(np.squeeze(states), np.squeeze(availAcs))
                rewards, dones, infos = self.vectorizedGame.step(selectedAcs)

                for game in range(self.nGames):
                    if dones[game] and not completed[game]:
                        completed[game] = True
                        totalCompleted += 1
                        
                        evaluatedNetworkScore = rewards[game][evaluatedNetworkTurn]
                        self.score += evaluatedNetworkScore
                        if evaluatedNetworkScore > 0:
                            self.wins += 1

                currTurn += 1
                if currTurn == 4:
                    currTurn = 0

        winPercentage = '{:.2f}'.format(self.wins * 100 / (self.nGames * self.nRounds))
        print("Wins: {0} of {1} games ({2}%)".format(self.wins, self.nGames * self.nRounds, winPercentage))
        print("Score: {0} (average per game: {1})".format(int(self.score), '{:.2f}'.format(self.score / (self.nGames * self.nRounds))))

if __name__ == '__main__':

    config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        evaluator = SingleNetworkEvaluator(sess=sess, nGames=64, nRounds=1000)
        evaluator.playoutVectorized()

            