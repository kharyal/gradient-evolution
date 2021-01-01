import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from copy import deepcopy

im_sz = (128,128)
im1 = cv2.resize(cv2.imread("1.png",0), im_sz)[:,:,None]/255
# plt.imshow(np.reshape(im1,im_sz), 'gray')
# plt.show()
im1 = np.reshape(im1, (im_sz[0]*im_sz[1],1))


# sz = 50
# gen_img = np.random.uniform(0,1, (im_sz[0],im_sz[1],3, sz))


# iterations = 1001
# with tqdm(iter(range(iterations))) as titer:
#     for i in titer:

#         pix_err = np.abs((gen_img-im1))
#         err = np.sum((pix_err)**2, axis=(0,1,2))
#         err = np.array(sorted(zip(err, np.arange(sz))))
#         titer.set_postfix(ERROR = err[0,0])
#         gen_img = gen_img[:,:,:,err[:sz//2,1].astype(int)]
#         pix_err = pix_err[:,:,:,err[:sz//2,1].astype(int)]
#         mutated = (gen_img + np.random.normal(0,0.5, (im_sz[0], im_sz[1],3, sz//2)))
#         mutated = np.clip(mutated, 0, 1)
#         gen_img = np.concatenate((gen_img, mutated), axis=3)
#         if i%1 == 0:
#             # plt.imshow(gen_img[:,:,:,0])
#             # plt.show()
#             cv2.imshow('reconstruction', gen_img[:,:,:,0])
#             cv2.waitKey(1)
            
#             # time.sleep(1)


# cv2.destroyAllWindows()

# def genetic_grad(im1, size_generation=30, size_eta=50, iterations=10000, lr = 0.5):
#     #generation
#     sz = size_generation
#     gen_img = np.random.uniform(0,1, (im_sz[0]*im_sz[1], sz))
    
#     # grad_actual = 0
#     # grad_pred = 0
#     with tqdm(iter(range(iterations))) as titer:
#         for i in titer:

#             pix_err = (gen_img-im1)
#             err = np.sum((pix_err)**2, axis=0)
#             err = np.array(sorted(zip(err, np.arange(sz))))
#             gen_img = gen_img[:,err[:sz//2,1].astype(int)]
#             gen_img_copy = deepcopy(gen_img)
#             # print(err)
#             lr = min(lr, 150/(i+0.1))
#             for j in range(gen_img_copy.shape[1]):
#                 eta = np.random.uniform(-0.0000005, 0.0000005, (im_sz[0]*im_sz[1], size_eta))
#                 img_eta = gen_img[:,j][:,None] + eta
#                 err_eta = np.sum(np.abs((img_eta - im1))**2, axis = 0)
#                 b = err_eta - err[j,0]
#                 # grad,_,_,_ = np.linalg.lstsq(eta.T, b)
#                 # grad = grad[:,None]
#                 grad = np.linalg.pinv(eta.dot(eta.T))@eta@b
#                 # grad = 2*(gen_img[:,j] - im1.T)
#                 # print(grad.shape)
#                 # print("AAAAA",(gen_img[:,j] - lr*grad).shape)
#                 # print("AAAAA",(gen_img[:,j] - lr*grad.T).T.shape)
#                 gen_img = np.hstack((gen_img, (gen_img[:,j] - lr*grad.T).T))
#                 # gen_img = np.hstack((gen_img, (gen_img[:,j] - lr*grad).T))
#                 # print(gen_img.shape)
#                 # exit()
#                 # print(gen_img.shape)
#                 if j == 0:
#                     grad_pred = grad
#                     grad_actual = 2*(gen_img[:,j] - im1.T)
#             # print(grad_pred.shape)
#             grad_angle = np.arccos(grad_actual.dot(grad_pred)/(np.linalg.norm(grad_actual)*np.linalg.norm(grad_pred)))
#             titer.set_postfix(ERROR = err[0,0], grad_angle = grad_angle)
#             # titer.set_postfix(ERROR = err[0,0])
#             cv2.imshow('reconstruction', np.reshape(gen_img[:,0], (im_sz[0], im_sz[1])))
#             cv2.waitKey(1)
# genetic_grad(im1)



def choose_eta(size):
    choices = np.arange(size[0])
    basis_ind = np.random.choice(choices, size[1])
    eta = np.zeros(size)
    # print(eta.shape)
    eta[basis_ind, np.arange(size[1])] = 1
    return eta

def genetic_grad(im1, size_generation=30, size_eta=30, iterations=10000, lr = 0.5):
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

                # eta = np.random.uniform(-1, 1, (im_sz[0]*im_sz[1], size_eta))
                eta = choose_eta((im_sz[0]*im_sz[1], size_eta))
                # eta = eta[:, :size_eta]
                img_eta = gen_img[:,j][:,None] + t*eta
                err_eta = np.sum((img_eta - im1)**2, axis = 0)
                b = (err_eta - err[j,0])/t
                grad,_,_,_ = np.linalg.lstsq(eta.T, b)
                grad = grad[:,None]
                # print(grad.shape)
                # print("AAAAA",(gen_img[:,j] - lr*grad).shape)
                # print("AAAAA",(gen_img[:,j] - lr*grad.T).T.shape)
                # print(gen_img.shape)
                gen_img = np.hstack((gen_img, (gen_img[:,j] - lr*grad.T).T))
                # gen_img = np.hstack((gen_img, (gen_img[:,j] - lr*grad).T))
                # print(gen_img.shape)
                # exit()
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
