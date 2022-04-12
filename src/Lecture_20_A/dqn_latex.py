import numpy as np
import random
import gym
import tensorflow as tf
import keras

# Example Python code for training DQN corresponding to lines of Algorithm 8
def TrainDQN():
    # Line 1: Initialize replay buffer D to capacity N
    D = collections.deque(maxlen = N)
    # Line 2: Initialize action value function Q with random weights  
    Q = DQN()
    dqn.random_init()
    # Line 3: Initialize target action value function Q_hat with same weights
    Q_hat = DQN()
    Q_hat.weights = Q.weights
    # Line 4: for each episode...
    for i in range(M):
        # Line 5: Initialize sequence s1 and preprocessed sequence phi
        x = env.reset()
        s = [x]
        phi = preprocess(x)
        # Line 6: for each time step...
        for t in range(T):
            # Line 7: select random action with prob epsilon, or the action that maximizes Q
            q_values = dqn.predict(phi)
            if (random.random() < epsilon):
                a = np.random.choice(np.arange(q_values.shape[0]))
            else:
                a = np.argmax(q_values) 
            # Line 8: observe r and image x from action a
            next_x, r, done = env.step(a)
            # Line 9: set s_{t+1} and preprocess phi_{t+1}
            s.extend([a, next_x])
            next_phi = preprocess(next_s)
            # Line 10: store transition in replay memory
            D.append((phi, a, r, next_phi, done))
            # Line 11: Randomly sample minibatch of transitions
            transitions = random.sample(D, batch_size) 
            batch_phi = np.array([t[0] for t in transitions])
            batch_a = np.array([t[1] for t in transitions])
            batch_r = np.array([t[2] for t in transitions])
            batch_next_phi = np.array([t[3] for t in transitions])
            batch_done = [t[4] for t in transitions]
            # Line 12: set y_i
            batch_y = batch_r
            for i,done in enumerate(batch_done):
                if not done:
                    batch_y[i] += gamma*np.max(Q_hat.predict(batch_phi[i]))
            # Line 13: gradient descent on squared error with respect to theta
            DQN.backprop((batch_y - Q.predict(batch_phi)[batch_a])**2)
            # Line 14: copy weights every C steps
            if (t % C == 0): 
                Q_hat.weights = Q.weights
    # Line 17: return 
    return Q
