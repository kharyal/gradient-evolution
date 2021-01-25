import numpy as np
import gym
import time
import multiprocessing as mp

num_processes = 4
gen_size = 10
is_continuous = [False, 2]

def choose_eta(size):
    choices = np.arange(size[0])
    basis_ind = np.random.choice(choices, size[1])
    eta = np.zeros(size)
    # print(eta.shape)
    eta[basis_ind, np.arange(size[1])] = 1
    return eta

def get_reward(weights,env,shape,ep_max_step=700):
    rewards = []
    N = 10
    for i in range(N):
        s = env.reset()
        # print(s.shape)
        params = params_reshape(weights,shape)
        # reward = 100.
        reward = 0.
        # rt = 0.
        # ss = []
        for i in range(ep_max_step):
            # s = np.concatenate((s['observation'], s['desired_goal']))
            s = s.reshape(8,1)
            # print(s.shape)
            a = get_action(params,s,shape)
            # a = a.reshape(2,1)
            s,r,done, _ = env.step(a)
            reward += r
            # rt += r
            # reward = max(rt[0], reward)
            # print(r)
            # ss.append()
            if done:
                break
        # return np.max(ss)
        rewards.append(reward)
    return np.average(rewards)

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

def mutate_random(parent_weights):
    return parent_weights + np.random.uniform(-1,1, parent_weights.shape)


def mutate(parent_weights,prev_grad,parent_reward,env,shape,pool,num_eta = 30,t = 0.00001,lr = 0.1, gamma = 0.3):
    eta = choose_eta((parent_weights.shape[0],num_eta))
    eta_weights = parent_weights[:,None] + t*eta
    jobs = [pool.apply_async(get_reward,(eta_weights[:,k], env, shape)) for k in range(num_eta)]
    rewards = np.array([j.get() for j in jobs])
    b = (rewards - parent_reward)/t
    b = b.reshape(max(b.shape),1)
    # print(eta.shape, b.shape)
    grad,_,_,_ = np.linalg.lstsq(eta.T,b)
    grad = grad[:,None]; print(lr, np.linalg.norm(grad), end = " ")
    if np.linalg.norm(grad)/grad.shape[0] < 0.1:
        print("random")
        grad = np.random.uniform(-10/lr,10/lr, grad.shape)
    else:
        print("")
    child_weights = parent_weights + lr*(grad.T + gamma* prev_grad)
    return child_weights.reshape(parent_weights[:,None].shape), (grad.T + prev_grad).reshape(parent_weights[:,None].shape)

def play(weights,env,ep_max_step=1000):
    s = env.reset()
    # print(weights.shape)
    params = params_reshape(weights,shape)
    reward = 0.
    for i in range(ep_max_step):
        env.render()
        # s = np.concatenate((s['observation'], s['desired_goal']))
        s = s.reshape(8,1)
        a = get_action(params,s,shape)
        # a = a.reshape(4,1)
        s,r,done, _ = env.step(a)
        reward += r
        if done:
            print("Done")
            break
    return reward


def train(weights, shape, env, pool, max_iteration = 50):
    # max_iteration = 50
    avg_top5 = 0
    num_eta = 6
    lr = 1
    print(weights.shape)
    grads = np.zeros(weights.shape)
    print(grads.shape)
    for i in range(max_iteration):
        num_eta = min(num_eta+i//2, 30)
        jobs = [pool.apply_async(get_reward,(weight.T,env,shape)) for weight in weights.T]
        rewards = np.array([j.get() for j in jobs])
        prev_avg_top5 = avg_top5
        err = np.array(sorted(zip(rewards,np.arange(gen_size)),reverse = True))
        weights = weights[:,err[:gen_size//2,1].astype(int)]
        grads = grads[:,err[:gen_size//2,1].astype(int)]
        avg_top5 = np.mean(err[:5,0])
        # jobs = [pool.apply_async(mutate,(weights[:,j],err[j,0],env,shape,pool)) for j in range(weights.shape[1])]
        # new_weights = np.array([j.get() for j in jobs])
        ret = [mutate(weights[:,j],grads[:,j],err[j,0],env,shape,pool, num_eta=num_eta, lr = lr) for j in range(gen_size//2)]
        # new_weights = np.array(ret[:,1])
        
        for j in range(gen_size//2):
            ret = mutate(weights[:,j],grads[:,j],err[j,0],env,shape,pool, num_eta=num_eta, lr = lr)
            weights = np.hstack((weights,ret[0]))
            grads = np.hstack((grads,np.zeros(ret[1].shape)))

        # new_weights= np.reshape(new_weights,(weights.shape[0],gen_size//2))
        # print(new_weights.shape)
        # weights = np.hstack((weights,new_weights))
        # grads = np.hstack((grads,np.zeros_like(new_weights.shape)))
        
        # print(weights.shape)
        # exit()
        #if abs(avg_top5 - prev_avg_top5)<0.5:
        #    lr = min(lr*1.2, 2)
        #else:
        #    lr = 0.1
        print(err)
        print("Iter: ", i , err[0,0])
        if i%10 == 0:
            np.savetxt(str(i)+".csv", weights[:,0])
    
    play(weights[:,0], env, ep_max_step = 10000000)

def train_random(weights, shape, env, pool, max_iteration = 20):
    for i in range(max_iteration):
        jobs = [pool.apply_async(get_reward,(weight.T,env,shape)) for weight in weights.T]
        rewards = np.array([j.get() for j in jobs])
        err = np.array(sorted(zip(rewards,np.arange(gen_size)),reverse = True))
        weights = weights[:,err[:gen_size//2,1].astype(int)]
        # jobs = [pool.apply_async(mutate,(weights[:,j],err[j,0],env,shape,pool)) for j in range(weights.shape[1])]
        # new_weights = np.array([j.get() for j in jobs])
        new_weights = np.array([mutate_random(weights[:,j]) for j in range(weights.shape[1])])
        new_weights = np.reshape(new_weights,(weights.shape[0],gen_size//2))
        # print(new_weights.shape)
        
        # print(weights.shape)
        # exit()
        print("Iter: ", i , err[0,0])
    #play(weights[:,0], env, ep_max_step = 10000000)
        if i%10 == 0:
            np.savetxt(str(i)+"random.csv", weights[:,0])
        weights = np.hstack((weights,new_weights))
if __name__ == "__main__":
    pool = mp.Pool(processes = num_processes)
    env = gym.make("LunarLander-v2")
    # print(env.reset())
    num_features = 8
    num_actions = 4
    ep_max_step = 700
    iterations = 1001
    weights = np.array([build_net(num_features,num_actions)[1] for j in range(gen_size)]).T
    shape,_ = build_net(num_features,num_actions)
    # play(weights[:,0],env,ep_max_step = 10000)
    print(shape,weights[:,0].shape)
    train(weights,shape,env,pool, iterations)
    #train_random(weights, shape, env, pool, iterations)
