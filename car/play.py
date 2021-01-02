import sys
import numpy as np
import gym
import time
import multiprocessing as mp
import highway_env


is_continuous = [True, 2]

def play(weights,shape, env,ep_max_step=1000):
    while True:
        s = env.reset()
        # print(weights.shape)
        params = params_reshape(weights,shape)
        reward = 0.
        for i in range(ep_max_step):
            env.render()
            s = np.concatenate((s['observation'], s['desired_goal']))
            a = get_action(params,s,shape).T
            # a = a.reshape(2,1)
            s,r,done, _ = env.step(a)
            reward += r
            if done:
                print("Done")
                break

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
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(num_features, 10)
    s1, p1 = linear(10, 5)
    s2, p2 = linear(5, num_actions)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))

def get_action(params,state,shape):
    # params = params_reshape(weights,shape)
    x = state[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    if not is_continuous[0]:
        return np.argmax(x, axis=1)[0]
    else:
        return is_continuous[1]*np.tanh(x)


if __name__ == "__main__":
    file_name = str(sys.argv[1])
    weights = np.loadtxt(file_name)
    env = gym.make("parking-v0")
    num_features = 12
    num_actions = 2

    shape,_ = build_net(num_features, num_actions)
    play(weights, shape, env, 100000000000)