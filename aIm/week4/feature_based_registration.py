#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:20:32 2022

@author: annarife
"""

import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.ndimage
import cv2

#%% FEATURED-BASED REGISTRATION

# 4.1 ROTATION, TRANSLATION AND SCALE

#Generation of random 2D points = P set
P= np.matrix(np.random.uniform(1,10,(2,20)))


#Translation, rotation and scale variables. 
tx,ty = 0.2, 2
alpha= 30
t=np.matrix([tx,ty])
alpha=20
R=np.matrix([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
s=0.6

#Transform P to obtain Q set
Q=s*np.matmul(R, P)+t.T

#Plot points set
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('x coordinates')
ax.set_ylabel('y coordinates')
ax.plot(P[0,:],P[1,:],'go') #original set P
ax.plot(Q[0,:],Q[1,:],'ro') #transformed set Q
ax.legend(['Original Points','Transformed Points'])
plt.show()



#Function that computes parameters t', rotation R' and scale s' from P and Q. 
def transformation(P,Q):
    up= np.mean(P,axis=1)
    uq= np.mean(Q,axis=1)
    
    s_q= np.sum(np.sqrt(np.array(np.abs(Q-uq)[0])**2 + np.array(np.abs(Q-uq)[1])**2))
    s_p= np.sum(np.sqrt(np.array(np.abs(P-up)[0])**2 + np.array(np.abs(P-up)[1])**2))
    scale= s_q/s_p

    C=np.matmul((Q-uq),(P-up).T)
    u,s,vh= np.linalg.svd(C)
    R=np.matmul(u,vh)
    
    t=uq.T-scale*((R*up).T)
    
    return scale,R,t

s_new,R_new,t_new= transformation(P,Q)



#%% 4.2 COMPUTE AND MATCH SIFT

im1=skimage.io.imread('quiz_image_1.png') #query image
im2=skimage.io.imread('quiz_image_2.png') #trained image
fig,ax=plt.subplots(1,2)
ax[0].imshow(im1,cmap='gray')
ax[1].imshow(im2,cmap='gray')


#Compute SIFT features in the two images, class for extracting keypoints and computing descriptors using SIFT
sift= cv2.SIFT_create() #Initialize SIFT creator

#Find keypoints and descriptors with SIFT (number of keypoints x 128(8binsx16blocks))
kp1,des1=sift.detectAndCompute(im1,None)
kp2,des2=sift.detectAndCompute(im2,None)

#Extract coordianates of the keypoints. 
coord_img1=[]
coord_img2=[]
for a in range(len(kp1)):
    coord_img1.append((kp1[a].pt))
for a in range(len(kp2)):
    coord_img2.append((kp2[a].pt))


#Match the SIFT features, BFMAtcher with default parameters
bf=cv2.BFMatcher()
matches= bf.knnMatch(des1,des2,k=2)

#Apply ratio rest
good=[]
for m,n in matches: 
    #Apply the Lowe crieterion best should be closer than second best 
    if m.distance/(n.distance + 10e-10) <0.6:
        good.append([m])


# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

#%% Match SIFT - function to extract the coordinates of matching points

# Parameters that can be extracted from keypoints
# keypoint.pt[0],
# keypoint.pt[1],
# keypoint.size,
# keypoint.angle,
# keypoint.response,
# keypoint.octave,
# keypoint.class_id,


def match_SIFT(im1, im2, thres = 0.6):
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good_matches = []
    
    for m,n in matches:
        if m.distance/(n.distance+10e-10) < thres:
            good_matches.append([m])
    
    # Get the coordinates of the matching keypoints for each of the images
    pts_im1 = [kp1[m[0].queryIdx].pt for m in good_matches]
    pts_im1 = np.array(pts_im1, dtype=np.float32).T
    pts_im2 = [kp2[m[0].trainIdx].pt for m in good_matches]
    pts_im2 = np.array(pts_im2, dtype=np.float32).T
    return pts_im1, pts_im2


#Matching keypoints of the two images
pts_im1,pts_im2= match_SIFT(im1,im2,0.6)



#Function to plot the keypoints
def plot_matching_keypoints(im1, im2, pts_im1, pts_im2):
    r1,c1 = im1.shape
    r2,c2 = im2.shape
    n_row = np.maximum(r1, r2)
    n_col = c1 + c2
    im_comp = np.zeros((n_row,n_col))
    im_comp[:r1,:c1] = im1
    im_comp[:r2,c1:(c1+c2)] = im2
    
    fig,ax = plt.subplots(1)
    ax.imshow(im_comp, cmap='gray')
    ax.plot(pts_im1[0],pts_im1[1],'.r')
    ax.plot(pts_im2[0]+c1,pts_im2[1],'.b')
    ax.plot(np.c_[pts_im1[0],pts_im2[0]+c1].T,np.c_[pts_im1[1],pts_im2[1]].T,'w',linewidth = 0.5)


plot_matching_keypoints(im1,im2,pts_im1, pts_im2)


#%% Compute the transformaions (roataion, translation, scale) before transforming keypoints of im1 to the ones found in im2. 

#QUIZ QUESTIONS WEEK 4
s,R,t = transformation(np.matrix(pts_im2),np.matrix(pts_im1))
print('The magnification scale is:',s)
angle= (np.arccos(np.array(R))[0][0] *180)/np.pi
print('The absolute angle of rotation in degrees is:',angle)
len_trans= np.sqrt(np.array(t)[0][0]**2 +np.array(t)[0][1]**2)
print('The translation vector is:',len_trans)


#We transform the set of points found image 1 to the set of keypoints found in image 2. 
pts_im1_trans=s*np.matmul(R, pts_im2)+t.T

#To confirm that the points matched. 
fig,ax = plt.subplots(1)
ax.imshow(im1)
ax.plot(pts_im1[0],pts_im1[1],'.r')
ax.plot(pts_im1_trans[0],pts_im1_trans[1],'.b')



