import json
#matplotlib for rendering
import matplotlib.pyplot as plt
#numpy for handeling matrix operations
import numpy as np
#time, to, well... keep track of time
import time
#Python image libarary for rendering
from PIL import Image
#iPython display for making sure we can render the frames
from IPython import display
#seaborn for rendering
import seaborn
import keras
import sys
import gym
import pylab
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


#Setup matplotlib so that it runs nicely in iPython
get_ipython().run_line_magic('matplotlib', 'inline')
#setting up seaborn
seaborn.set()

class Catch(object):
    """
    Class catch is the actual game.
    """
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas
        
    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]

        
"""
Here we define some variables used for the game and rendering later
"""
#last frame time keeps track of which frame we are at
last_frame_time = 0
#translate the actions to human readable words
translate_action = ["Left","Stay","Right","Create Ball","End Test"]
#size of the game field
grid_size = 10

def display_screen(action,points,input_t):
    #Function used to render the game screen
    #Get the last rendered frame
    global last_frame_time
    print("Action %s, Points: %d" % (translate_action[action],points))
    #Only display the game screen if the game is not over
    if("End" not in translate_action[action]):
        #Render the game with matplotlib
        plt.imshow(input_t.reshape((grid_size,)*2),
               interpolation='none', cmap='gray')
        #Clear whatever we rendered before
        display.clear_output(wait=True)
        #And display the rendering
        display.display(plt.gcf())
    #Update the last frame time
    last_frame_time = set_max_fps(last_frame_time)
    
    
def set_max_fps(last_frame_time,FPS = 16):
    current_milli_time = lambda: int(round(time.time() * 1000))
    sleep_time = 1./FPS - (current_milli_time() - last_frame_time)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return current_milli_time()




EPISODES = 1000000


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = True
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.9
        self.learning_rate = 1e-3
#         self.decay = 0.999
#         self.learning_rate = 0.00035
        self.hidden1, self.hidden2 = 100, 20

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        
    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a). 
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=8).flatten()
#         print(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
    
    def clear_episode(self):
        self.states.clear()
        self.rewards.clear()
        self.actions.clear()

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
#             print(self.actions[i])
            advantages[i][self.actions[i]] = discounted_rewards[i]-1.434

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.clear_episode()
#         self.states, self.actions, self.rewards = [], [], []

if __name__ == "__main__":
    state_size = 100
    action_size = 3
    
    #setup environment
    plt.ion()
    # Define environment, game
    env = Catch(grid_size)
    maxs = 0;
    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes, average_reward = [], [], []
    score = 0
    for e in range(EPISODES):
        done = False
        env.reset()
        state = env.observe()
        state = np.reshape(state, [1, state_size])
        while not done:
#             if agent.render:
#                 display_screen(action,score,state)
            
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
#             print(action)   
#             next_state, reward, done, info = env.step(action)
            next_state, reward, done = env.act(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = float(reward)

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)
        
            score += reward
            state = next_state
            if done:
                # every episode, agent learns from sample returns
                agent.train_model()
                average_reward.append(score)
                score = 0
                if len(average_reward)>300: average_reward.pop(0)
                ar = np.mean(average_reward)
                if maxs<ar and e > 300:
                    maxs = ar
                    agent.model.save("./catch_reinforce.h5")
                    print("saved model maxs:", maxs, "caught fruits (300): ",(300*maxs+300)/2)

                print("episode:", e,"reward:",reward,"average reward:",ar)



print(maxs)


from keras.models import load_model
state_size = 100
action_size = 3
env = Catch(grid_size)
# make REINFORCE agent
agent = load_model('catch_reinforce.h5')
score = 0
for e in range(500):
    done = False
    env.reset()
    state = env.observe()
    state = np.reshape(state, [1, state_size])
    print(str(e)+" "+str(score))
    while not done:
        policy = agent.predict(state)
        action = np.argmax(policy)
        next_state, reward, done = env.act(action)
        next_state = np.reshape(next_state, [1, state_size])
        score = score + reward
#         display_screen(action,score,next_state)
        state = next_state
        

