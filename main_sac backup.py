
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
from big2Game import vectorizedBig2Games

available_action_space =  1695
observation_space = [412]

if __name__ == '__main__':
    nGames = 1
    env = vectorizedBig2Games(nGames)

    agent = Agent(input_dims=observation_space, env=env,
            n_actions=available_action_space)
    n_games = 150
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    filename = 'big2.png'

    figure_file = 'plots/' + filename

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')
    for i in range(n_games):
        print('Game: {}'.format(i))
        observation = env.reset()
        observation = observation[0][1][0]
        done = False
        score = 0
        while not done: 
            action = agent.choose_action(observation)
            # print('action chosen', np.argmax(action))
            
            ##### to do -> add multiple SAC agent to the game
            
            current_turn, current_state, current_available_actions = env.getCurrStates()
            print('current_available_actions', current_available_actions)
            print('action chosen', np.argmax(action))
            
            reward, done, observation_ = env.step([np.argmax(action)])
            # print([np.argmax(action)], np.array(done)[0])

            # print('1 reward done observation',reward, done, observation_)
            done = np.array(done)[0]
            reward = np.array(reward)[0]
            player_turn = np.array(observation_)[0][0]
            observation_ = np.array(observation_)[0][1]
            
            print('reward: ', reward, ', done: ', done, ', player turn: ', player_turn, ', observation: ', observation_)
            
            # print('2 reward done observation', reward, done, player_turn,  observation_.shape)

            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

