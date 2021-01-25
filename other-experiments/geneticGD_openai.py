import numpy as np
import gym
import time
import multiprocessing as mp

num_processes = 4
gen_size = 30

def choose_eta(size):
	choices = np.arange(size[0])
	basis_ind = np.random.choice(choices, size[1])
	eta = np.zeros(size)
	# print(eta.shape)
	eta[basis_ind, np.arange(size[1])] = 1
	return eta

def get_reward(weights,env,shape,ep_max_step=700):
	s = env.reset()
	# print(weights.shape)
	params = params_reshape(weights,shape)
	# reward = 100.
	reward = 0.
	# ss = []
	for i in range(ep_max_step):
		a = get_action(params,s,shape)
		s,r,done, _ = env.step(a)
		reward += r
		# ss.append(s)
		if done:
			break
	# return np.max(ss)
	return reward

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
	s0, p0 = linear(num_features, 30)
	s1, p1 = linear(30, 20)
	s2, p2 = linear(20, num_actions)
	return [s0, s1, s2], np.concatenate((p0, p1, p2))

def get_action(params,state,shape):
	# params = params_reshape(weights,shape)
	x = state[np.newaxis, :]
	x = np.tanh(x.dot(params[0]) + params[1])
	x = np.tanh(x.dot(params[2]) + params[3])
	x = x.dot(params[4]) + params[5]
	return np.argmax(x, axis=1)[0]

def mutate(parent_weights,parent_reward,env,shape,pool,num_eta = 10,t = 0.001,lr = 0.5):
	eta = choose_eta((parent_weights.shape[0],num_eta))
	eta_weights = parent_weights[:,None] + t*eta
	jobs = [pool.apply_async(get_reward,(eta_weights[:,k], env, shape)) for k in range(num_eta)]
	rewards = np.array([j.get() for j in jobs])
	b = (rewards - parent_reward)/t
	grad,_,_,_ = np.linalg.lstsq(eta.T,b)
	grad = grad[:,None]
	child_weights = parent_weights + lr*grad.T
	return child_weights

def play(weights,env,ep_max_step=1000):
	s = env.reset()
	# print(weights.shape)
	params = params_reshape(weights,shape)
	reward = 0.
	for i in range(ep_max_step):
		env.render()
		a = get_action(params,s,shape)
		s,r,done, _ = env.step(a)
		reward += r
		if done:
			print("Done")
			break
	return reward


def train(weights, shape, env, pool):
	max_iteration = 50
	for i in range(max_iteration):
		jobs = [pool.apply_async(get_reward,(weight.T,env,shape)) for weight in weights.T]
		rewards = np.array([j.get() for j in jobs])
		err = np.array(sorted(zip(rewards,np.arange(gen_size)),reverse = True))
		weights = weights[:,err[:gen_size//2,1].astype(int)]
		# jobs = [pool.apply_async(mutate,(weights[:,j],err[j,0],env,shape,pool)) for j in range(weights.shape[1])]
		# new_weights = np.array([j.get() for j in jobs])
		new_weights = np.array([mutate(weights[:,j],err[j,0],env,shape,pool) for j in range(weights.shape[1])])
		new_weights = np.reshape(new_weights,(weights.shape[0],gen_size//2))
		# print(new_weights.shape)
		weights = np.hstack((weights,new_weights))
		# print(weights.shape)
		# exit()
		print(err)
		print("Iter: ", i , err[0,0])
	
	play(weights[:,0], env, ep_max_step = 10000000)


if __name__ == "__main__":
	pool = mp.Pool(processes = num_processes)
	env = gym.make('CartPole-v0')
	num_features = 4
	num_actions = 2
	ep_max_step = 700
	weights = np.array([build_net(num_features,num_actions)[1] for j in range(gen_size)]).T
	shape,_ = build_net(num_features,num_actions)
	# play(weights[:,0],env,ep_max_step = 10000)
	# print(shape,weights.shape)
	train(weights,shape,env,pool)