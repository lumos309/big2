import numpy as np
from sac_torch import Agent
from big2Game import vectorizedBig2Games
import numpy as np
import sys

#taken directly from baselines implementation - reshape minibatch in preparation for training.
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])



class big2PPOSimulation(object):
    
    def __init__(self, *, inpDim = 412, nGames = 4, nSteps = 100, nMiniBatches = 4, nOptEpochs = 5, lam = 0.95, gamma = 0.995, ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 0.5, minLearningRate = 0.000001, learningRate, clipRange, saveEvery = 500):
         
        available_action_space =  1695
        observation_space = [412]
        
        #environment
        self.vectorizedGame = vectorizedBig2Games(nGames)
        
        #network/model for training
        self.trainingNetwork =  Agent(input_dims=observation_space, env=self.vectorizedGame, n_actions=available_action_space)
        
        #player networks which choose decisions - allowing for later on experimenting with playing against older versions of the network (so decisions they make are not trained on).
        self.playerNetworks = {}
        
        #for now each player uses the same (up to date) network to make it's decisions.
        self.playerNetworks[1] = self.playerNetworks[2] = self.playerNetworks[3] = self.playerNetworks[4] = self.trainingNetwork
        self.trainOnPlayer = [True, True, True, True]

        #params
        self.nGames = nGames
        self.inpDim = inpDim
        self.nSteps = nSteps
        self.nMiniBatches = nMiniBatches
        self.nOptEpochs = nOptEpochs
        self.lam = lam
        self.gamma = gamma
        self.learningRate = learningRate
        self.minLearningRate = minLearningRate
        self.clipRange = clipRange
        self.saveEvery = saveEvery
        
        self.rewardNormalization = 5.0 #divide rewards by this number (so reward ranges from -1.0 to 3.0)
        
        #test networks - keep network saved periodically and run test games against current network
        self.testNetworks = {}
        
        # final 4 observations need to be carried over (for value estimation and propagating rewards back)
        self.prevObs = []
        self.prevGos = []
        self.prevAvailAcs = []
        self.prevRewards = []
        self.prevActions = []
        self.prevValues = []
        self.prevDones = []
        
        #episode/training information
        self.totTrainingSteps = 0
        self.epInfos = []
        self.gamesDone = 0
        self.losses = []
        
    def run(self):
        mb_individual_rewards = []
        #run vectorized games for nSteps and generate mini batch to train on.
        mb_obs, mb_pGos, mb_actions,  mb_neglogpacs, mb_rewards, mb_dones, mb_availAcs, mb_next_obs = [], [], [], [], [], [], [], []
        for i in range(len(self.prevObs)):
            mb_obs.append(self.prevObs[i])
            mb_pGos.append(self.prevGos[i])
            mb_actions.append(self.prevActions[i])
            # mb_values.append(self.prevValues[i])
            mb_neglogpacs.append(self.prevNeglogpacs[i])
            mb_rewards.append(self.prevRewards[i])
            mb_dones.append(self.prevDones[i])
            mb_availAcs.append(self.prevAvailAcs[i])
        if len(self.prevObs) == 4:
            endLength = self.nSteps
        else:
            endLength = self.nSteps-4
        for _ in range(self.nSteps):
            currGos, currStates, currAvailAcs = self.vectorizedGame.getCurrStates()
            currStates = np.squeeze(currStates)
            currAvailAcs = np.squeeze(currAvailAcs)
            currGos = np.squeeze(currGos)
            
            # generate observatios from curstates and curavailacs
            neglogpacs = self.trainingNetwork.choose_action(currStates)
            
            # add probabilities to the available actions, to rule out impossible actions
            possible_actions_probablities = np.add(neglogpacs,  currAvailAcs)
            actions = np.argmax(possible_actions_probablities, axis=1)
            print('actions taken: ' , actions)
            
            # step in the environment
            rewards, dones, infos = self.vectorizedGame.step(actions)
            # print('rewards', rewards)
            
            # add to the previous observations
            
            
            # if (dones[0] == True):
            #     print('results', rewards[0], dones[0], infos[0])
            mb_obs.append(currStates.copy())
            mb_pGos.append(currGos)
            mb_availAcs.append(currAvailAcs.copy())
            mb_actions.append(actions)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(list(dones))
            mb_rewards.append(np.array(rewards))
            currGos, currStates, currAvailAcs = self.vectorizedGame.getCurrStates()
            # print('currGos', currGos[0])
            mb_next_obs.append(currStates.copy())
            
            #now back assign rewards if state is terminal
            toAppendRewards = np.zeros((self.nGames,))
            mb_rewards.append(toAppendRewards)
            for i in range(self.nGames):
                if dones[i] == True:
                    print('finished game' , dones[i], rewards[i])
                    reward = rewards[i]
                    mb_rewards[-1][i] = reward[mb_pGos[-1][i]-1] / self.rewardNormalization
                    mb_rewards[-2][i] = reward[mb_pGos[-2][i]-1] / self.rewardNormalization
                    mb_rewards[-3][i] = reward[mb_pGos[-3][i]-1] / self.rewardNormalization
                    mb_rewards[-4][i] = reward[mb_pGos[-4][i]-1] / self.rewardNormalization
                    mb_dones[-2][i] = True
                    mb_dones[-3][i] = True
                    mb_dones[-4][i] = True
                    self.epInfos.append(infos[i])
                    self.gamesDone += 1
                    # print("Game %d finished.    1Lasted %d turns" % (self.gamesDone, infos[i]['numTurns']))
            
            
        self.prevObs = mb_obs[endLength:]
        self.prevGos = mb_pGos[endLength:]
        self.prevRewards = mb_rewards[endLength:]
        self.prevActions = mb_actions[endLength:]
        # self.prevValues = mb_values[endLength:]
        self.prevDones = mb_dones[endLength:]
        self.prevNeglogpacs = mb_neglogpacs[endLength:]
        self.prevAvailAcs = mb_availAcs[endLength:]
            # print('rewards shape',  np.asarray(mb_rewards, dtype=np.float32))
        mb_individual_rewards = np.zeros((self.nSteps, self.nGames))    
        print('mb_rewards', mb_rewards)
        print('mb_rewards', np.asarray(mb_rewards, dtype=np.float32).shape)
        for i in range(self.nGames):
            for j in range(self.nSteps):
                # mb_rewards[j][i] = mb_rewards[0][i][currGos[i]-1]
                # print('mb_rewards', mb_rewards[0][i])
                # print('asdasd',  mb_rewards[0][i][currGos[i]-1])
                # print('currgos', currGos[i])
                mb_individual_rewards[j][i] =  np.max(mb_rewards[j][i][currGos[i]-1])
                # mb_individual_rewards[j][i] =  mb_rewards[j][i][currGos[i]-1]
                
        mb_obs =  np.asarray(mb_obs, dtype=np.float32)[:endLength]
        mb_availAcs =  np.asarray(mb_availAcs, dtype=np.float32)[:endLength]
        # mb_rewards =  np.asarray(mb_rewards, dtype=np.float32)
        mb_actions =   np.asarray(mb_actions, dtype=np.float32)[:endLength]
        mb_next_obs =  np.asarray(mb_next_obs, dtype=np.float32)
        mb_dones =  np.asarray(mb_dones, dtype=np.float32)[:endLength]
        mb_individual_rewards =  np.asarray(mb_individual_rewards, dtype=np.float32)[:endLength]

        # print('mb_rewards', mb_rewards)
        return map(sf01, (mb_obs, mb_availAcs, mb_individual_rewards, mb_actions, mb_next_obs, mb_dones))
        
    def train(self, nTotalSteps):   

        # available_action_space =  1695
        # observation_space = [412]
        # nGames = 8 
        # gamesDone = 0
        # n_step = 250
        # filename = 'big2.png'
        # figure_file = 'plots/' + filename
        
        # best_score = 0
        # score_history = []
        # load_checkpoint = False


        # env = vectorizedBig2Games(nGames)
        
        
        # # self play
        # oppponents = {}
        # agent = Agent(input_dims=observation_space, env=env,
        #         n_actions=available_action_space)
        # oppponents[1] = oppponents[2] = oppponents[3] = agent


        # if load_checkpoint:
        #     agent.load_models()
        #     env.render(mode='human')

        # for i in range(n_step):
        #     observation = env.reset()
        #     done = False
        #     score = 0
        #     while not done:
        #         action = agent.choose_action(observation)
        #         observation_, reward, done, info = env.step(action)
                
        #         score += reward
        #         agent.remember(observation, action, reward, observation_, done)
                
        #         if not load_checkpoint:
        #             agent.learn()
                    
        #         observation = observation_
                
                
                
        #     score_history.append(score)
        #     avg_score = np.mean(score_history[-100:])

        #     if avg_score > best_score:
        #         best_score = avg_score
        #         if not load_checkpoint:
        #             agent.save_models()

        #     print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        # if not load_checkpoint:
        #     x = [i+1 for i in range(n_step)]
        #     plot_learning_curve(x, score_history, figure_file)
        
        # --------------------------------------------------
        
        
        nUpdates = nTotalSteps // (self.nGames * self.nSteps)
        best_score = 0
        score_history = []
        load_checkpoint = False
        for update in range(nUpdates):
            
            # get minibatch  
            mb_obs, mb_availAcs, mb_rewards, mb_actions, mb_next_obs, mb_dones = self.run()
            print('minibatch', mb_obs.shape, mb_availAcs.shape, mb_rewards.shape, mb_actions.shape, mb_next_obs.shape, mb_dones.shape)
            # print('mb_returns', np.max(mb_rewards))
            for i in range(mb_obs.shape[0]):
                self.trainingNetwork.remember(mb_obs[i], mb_actions[i], mb_rewards[i], mb_next_obs[i], mb_dones[i])
            
            if not load_checkpoint:
                self.trainingNetwork.learn()
            # batchSize = states.shape[0]
            # self.totTrainingSteps += batchSize
            
            # nTrainingBatch = batchSize // self.nMiniBatches
            
            # score_history.append(score)
            # avg_score = np.mean(score_history[-100:])

            # if avg_score > best_score:
            #     best_score = avg_score
            #     if not load_checkpoint:
            #         self.trainingNetwork.save_models()

            # print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
            # do the learning here
            
            # currParams = self.trainingNetwork.getParams()
            
            # mb_lossvals = []
            # inds = np.arange(batchSize)
            # for _ in range(self.nOptEpochs):
            #     np.random.shuffle(inds)
            #     for start in range(0, batchSize, nTrainingBatch):
            #         end = start + nTrainingBatch
            #         mb_inds = inds[start:end]
            #         mb_lossvals.append(self.trainingModel.train(lrnow, cliprangenow, states[mb_inds], availAcs[mb_inds], returns[mb_inds], actions[mb_inds], values[mb_inds], neglogpacs[mb_inds]))
            # lossvals = np.mean(mb_lossvals, axis=0)
            # self.losses.append(lossvals)
            
            # newParams = self.trainingNetwork.getParams()
            # needToReset = 0
            # for param in newParams:
            #     if np.sum(np.isnan(param)) > 0:
            #         needToReset = 1
                    
            # if needToReset == 1:
            #     self.trainingNetwork.loadParams(currParams)
            # print(update)
            # if update % self.saveEvery == 0:
            #     name = "modelParameters" + str(update)
            #     self.trainingNetwork.saveParams(name)
            #     joblib.dump(self.losses,"losses.pkl")
            #     joblib.dump(self.epInfos, "epInfos.pkl")

    
if __name__ == "__main__":
    import time
    old_stdout = sys.stdout

    log_file = open("SAC.log","w")
    sys.stdout = log_file




    # with tf.compat.v1.Session() as sess:
    mainSim = big2PPOSimulation( nGames=64, nSteps=20, learningRate = 0.00025, clipRange = 0.2)
    start = time.time()
    mainSim.train(1000000)
    end = time.time()
    print("Time Taken: %f" % (end-start))
    sys.stdout = old_stdout

    log_file.close()
        
        
