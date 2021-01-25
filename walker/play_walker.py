import numpy as np
import gym
import time
import multiprocessing as mp
from deterministic_bipedal_walker import BipedalWalker
import sys

num_processes = 4
gen_size = 20
is_continuous = [True, 1]

def choose_eta(size):
    choices = np.arange(size[0])
    basis_ind = np.random.choice(choices, size[1])
    eta = np.zeros(size)
    # print(eta.shape)
    eta[basis_ind, np.arange(size[1])] = 1
    return eta

def params_reshape(params,shapes):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p

def build_net(num_features, num_actions):
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32)*0.1
        b = np.random.randn(n_out).astype(np.float32)*0.1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(num_features, 10)
    s1, p1 = linear(10, 5)
    s2, p2 = linear(5, num_actions)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))

def get_action(params,state,shape):
    # params = params_reshape(weights,shape)
    # x = state[np.newaxis, :].astype(np.float64)
    x = state.T
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    if not is_continuous[0]:
        return np.argmax(x, axis=1)[0]
    else:
        return is_continuous[1]*np.tanh(x)

def play(weights,env,ep_max_step=1000):
    s = env.reset()
    # print(weights.shape)
    params = params_reshape(weights,shape)
    reward = 0.
    for i in range(ep_max_step):
        print(i)
        env.render()
        # s = np.concatenate((s['observation'], s['desired_goal']))
        s = s.reshape(24,1)
        a = get_action(params,s,shape)
        a = a.reshape(4,1)
        s,r,done, _ = env.step(a)
        reward += r
        if done:
            print("Done")
            break
    return reward


if __name__ == "__main__":
    pool = mp.Pool(processes = num_processes)
    env = BipedalWalker()
    # print(env.reset())
    num_features = 24
    num_actions = 4
    iterations = 100
    # play(weights[:,0],env,ep_max_step = 10000)
    # print(shape,weights[:,0].shape)
    # train(weights,shape,env,pool, iterations)
    # train_random(weights, shape, env, pool, iterations)
    filename = sys.argv[1]
    shape,weights = build_net(num_features,num_actions)
    # play(weights,env,ep_max_step = 10000)
    weights = np.loadtxt(filename)
    print(play(weights,env,ep_max_step = 1000))