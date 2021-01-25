import numpy as np
import gym
import time
import multiprocessing as mp
import matplotlib.pyplot as plt

num_processes = 4
gen_size = 20
is_continuous = [True, 2]
errs_ours = []
errs_random = []
def choose_eta(size):
    choices = np.arange(size[0])
    basis_ind = np.random.choice(choices, size[1])
    eta = np.zeros(size)
    # print(eta.shape)
    eta[basis_ind, np.arange(size[1])] = 1
    return eta

def get_reward(weights):
    reward = 10*weights.shape[0] + np.sum(weights**2 - 10*np.cos(2*np.pi*weights), axis = 0)
    return reward

def mutate_random(parent_weights):
    return parent_weights + np.random.uniform(-1,1, parent_weights.shape)

def mutate(parent_weights,parent_reward,pool,num_eta = 30,t = 0.00001,lr = 0.1):
    eta = choose_eta((parent_weights.shape[0],num_eta))
    eta_weights = parent_weights[:,None] + t*eta
    # jobs = [pool.apply_async(get_reward,(eta_weights[:,k])) for k in range(num_eta)]
    # rewards = np.array([j.get() for j in jobs])
    rewards = get_reward(eta_weights)
    b = (rewards - parent_reward)/t
    b = b.reshape(max(b.shape),1)
    # print(eta.shape, b.shape)
    grad,_,_,_ = np.linalg.lstsq(eta.T,b)
    grad = grad[:,None]; 
    # print(lr, np.linalg.norm(grad), end = " ")
    if np.linalg.norm(grad)/grad.shape[0] < 0.1:
        # print("random")
        grad = np.random.uniform(-10/(lr+1e-9),10/(lr+1e-9), grad.shape)
    # else:
    #     print("")
    child_weights = parent_weights - lr*grad.T
    return child_weights

def train(weights, pool, max_iteration = 50):
    avg_top5 = 0
    num_eta = 6
    lr_init = 0.01
    lr = lr_init
    for i in range(max_iteration):
        # lr = lr_init/(i+1)
        num_eta = min(num_eta+i//2, 30)
        rewards = get_reward(weights)
        prev_avg_top5 = avg_top5
        # get_reward(weights)
        err = np.array(sorted(zip(rewards,np.arange(gen_size)),reverse = False))
        weights = weights[:,err[:gen_size//2,1].astype(int)]
        avg_top5 = np.mean(err[:5,0])
        # jobs = [pool.apply_async(mutate,(weights[:,j],err[j,0],env,shape,pool)) for j in range(weights.shape[1])]
        # new_weights = np.array([j.get() for j in jobs])
        new_weights = np.array([mutate(weights[:,j],err[j,0],pool, num_eta=num_eta, lr = lr) for j in range(gen_size//2)])
        new_weights = np.reshape(new_weights,(weights.shape[0],gen_size//2))
        # print(new_weights.shape)
        weights = np.hstack((weights,new_weights))
        # if abs(avg_top5 - prev_avg_top5)<0.5:
            # lr = min(lr*1.2, 5)
        # else:
            # lr = lr_init/(i+1)
        # print(err)
        errs_ours.append(err[0,0])
        print("Iter: ", i , err[0,0], lr)
    

def train_random(weights, pool, max_iteration = 20):
    for i in range(max_iteration):
        rewards = get_reward(weights)
        err = np.array(sorted(zip(rewards,np.arange(gen_size)),reverse = False))
        weights = weights[:,err[:gen_size//2,1].astype(int)]
        # jobs = [pool.apply_async(mutate,(weights[:,j],err[j,0],env,shape,pool)) for j in range(weights.shape[1])]
        # new_weights = np.array([j.get() for j in jobs])
        new_weights = np.array([mutate_random(weights[:,j]) for j in range(weights.shape[1])])
        new_weights = np.reshape(new_weights,(weights.shape[0],gen_size//2))
        # print(new_weights.shape)
        weights = np.hstack((weights,new_weights))
        print("Iter: ", i , err[0,0])
        errs_random.append(err[0,0])
    # play(weights[:,0], env, ep_max_step = 10000000)

if __name__ == "__main__":
    pool = mp.Pool(processes = num_processes)
    iterations = 1000
    weights = np.array([np.random.uniform(-20,20, (800,1)) for j in range(gen_size)]).T
    weights = weights.reshape(800,20)
    # print(weights.shape)
    train(weights, pool, iterations)
    train_random(weights, pool, iterations)
    plt.plot(errs_ours)
    plt.plot(errs_random)
    plt.legend(["ours","evolutionary"])
    plt.grid(True)
    plt.show()
