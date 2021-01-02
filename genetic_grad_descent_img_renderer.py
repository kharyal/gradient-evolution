import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from copy import deepcopy

im_sz = (32,32)
im1 = cv2.resize(cv2.imread("1.jpg",0), im_sz)[:,:,None]/255
im1 = np.reshape(im1, (im_sz[0]*im_sz[1],1))

def choose_eta(size):
    choices = np.arange(size[0])
    basis_ind = np.random.choice(choices, size[1])
    eta = np.zeros(size)
    # print(eta.shape)
    eta[basis_ind, np.arange(size[1])] = 1
    return eta

def genetic_grad(im1, size_generation=30, size_eta=15, iterations=10000, lr = 0.5):
    #generation
    sz = size_generation
    gen_img = np.random.uniform(0,1, (im_sz[0]*im_sz[1], sz))
    
    # grad_actual = 0
    # grad_pred = 0
    with tqdm(iter(range(iterations))) as titer:
        for i in titer:

            pix_err = (gen_img-im1)
            err = np.sum((pix_err)**2, axis=0)
            err = np.array(sorted(zip(err, np.arange(sz))))
            gen_img = gen_img[:,err[:sz//2,1].astype(int)]
            gen_img_copy = deepcopy(gen_img)
            # print(err)
            lr = min(lr, 400/(i+0.1))
            t = 0.001
            for j in range(gen_img_copy.shape[1]):
                eta = choose_eta((im_sz[0]*im_sz[1], size_eta))
                img_eta = gen_img[:,j][:,None] + t*eta
                err_eta = np.sum((img_eta - im1)**2, axis = 0)
                b = (err_eta - err[j,0])/t
                grad,_,_,_ = np.linalg.lstsq(eta.T, b)
                grad = grad[:,None]
                gen_img = np.hstack((gen_img, (gen_img[:,j] - lr*grad.T).T))
                if j == 0:
                    grad_pred = grad
                    grad_actual = 2*(gen_img[:,j] - im1.T)
            # print(grad_pred.shape)
            grad_angle = np.arccos(grad_actual.dot(grad_pred)/(np.linalg.norm(grad_actual)*np.linalg.norm(grad_pred)))
            titer.set_postfix(ERROR = err[0,0], grad_angle = grad_angle)
            # titer.set_postfix(ERROR = err[0,0])
            cv2.imshow('reconstruction', np.reshape(gen_img[:,0], (im_sz[0], im_sz[1])))
            cv2.waitKey(1)
genetic_grad(im1)
