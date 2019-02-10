#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 20:36:25 2019

@author: anubhav
"""

import cv2
import numpy as np

import skimage.data

image = skimage.data.astronaut()
image.shape
image[0,0]

import matplotlib.pyplot as plt
plt.imshow(image[0:10,0:10])

plt.imshow(cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY), cmap='gray')

image.dtype

plt.imshow(image.astype(np.float32))

plt.imshow(image.astype(np.float32) / 255.)

plt.subplot(1,3,1),plt.imshow(cv2.flip(image.copy(), 1))
plt.subplot(1,3,2),plt.imshow(cv2.flip(image.copy(), -1))
plt.subplot(1,3,3),plt.imshow(cv2.flip(image.copy(), 0));

plt.imshow(cv2.GaussianBlur(image.copy(), ksize=(11,11), sigmaX=-1))


#------------------------------------------------------------------------------
#Image blur
#1) Load the astronaut image to a variable, convert to grayscale and convert to a [0,1] floating point.
#
#2) Blur it with a 11x11 Box filter (cv2.boxFilter), and a 11x11 Gaussian filter
#
#3) Subtract the blurred images from the original
#
#4) Show the results to the notebook (plt.imshow(...)) side-by-side (plt.subplot(1,2,1)), with a colorbar (plt.colorbar())
#------------------------------------------------------------------------------


grayImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
grayImage = grayImage.astype(np.float32) / 255.

boxFilter = cv2.boxFilter(grayImage , ddepth = -1 , ksize=(11 , 11))
gaussFilter = cv2.GaussianBlur(grayImage , ksize=(11,11) , sigmaX=-1)

boxRes = grayImage - boxFilter
gaussRes = grayImage - gaussFilter

plt.subplot(1,2,1),plt.imshow(boxRes, cmap='gray'),plt.colorbar()
plt.subplot(1,2,2),plt.imshow(gaussRes, cmap='gray'),plt.colorbar()


#------------------------------------------------------------------------------
#Colorspaces
#1) Load the astronaut image (RGB)
#
#2) Convert to HSV (cv2.cvtColor)
#
#3) Display the H, S and V components, side-by-side
#------------------------------------------------------------------------------


hsvImage = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
#hsvImage = image.copy()
hChannel = hsvImage.copy()
hChannel[:,:,1] = 0
hChannel[:,:,2] = 0

sChannel = hsvImage.copy()
sChannel[:,:,0] = 0
sChannel[:,:,2] = 0

vChannel = hsvImage.copy()
vChannel[:,:,0] = 0
vChannel[:,:,1] = 0

plt.subplot(1,3,1),plt.imshow(hChannel)
plt.subplot(1,3,2),plt.imshow(sChannel)
plt.subplot(1,3,3),plt.imshow(vChannel)

pts3d = np.hstack([np.random.uniform(-5,5,(1000,2)),np.random.uniform(5,100,(1000,1))])[:,np.newaxis,:]
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts3d[:,0,0],pts3d[:,0,1],pts3d[:,0,2],c=-pts3d[:,0,2],cmap='gray');
ax.set_aspect('equal')

K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float32)


#------------------------------------------------------------------------------
#Calculate the 2D projection of the 3D points on the image plane, and plot them (`plt.scatter`).
#
#Try doing the calculation yourself, using matrix multiplication for the entire group, and also using `cv2.projectPoints`.
#
#Try changing the `K` matrix focal length parameters to see how it affects the projection.
#------------------------------------------------------------------------------


pts2d = cv2.projectPoints(pts3d , rvec = (0,0,0) , tvec = (0,0,0) , cameraMatrix = K , distCoeffs = None)
myPts2d = []
for pt in pts3d:
    newPt = pt.transpose()
    mult = np.matmul(K , newPt)
    myPts2d.append([mult[0][0]/mult[2][0] , mult[1][0]/mult[2][0]])
    
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(121 , projection = '3d')
ax1.set_title("Using Matrix Multiplication")
ax1.scatter(myPts2d[:] , myPts2d[:] , cmap='gray')
ax1.set_aspect('equal')

ax2 = fig.add_subplot(122 , projection = '3d')
ax2.set_title("Using `cv2.projectPoints`")
ax2.scatter(pts2d[0][:] , pts2d[0][:] , cmap='gray')
ax2.set_aspect('equal')

# cube veritces
Z = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
Z[:,2] += 10 # translate on Z
Z[:,0] += 2  # translate on X

# list of faces
faces = [[Z[0],Z[1],Z[2],Z[3]],
         [Z[4],Z[5],Z[6],Z[7]], 
         [Z[0],Z[1],Z[5],Z[4]], 
         [Z[2],Z[3],Z[7],Z[6]], 
         [Z[1],Z[2],Z[6],Z[5]],
         [Z[4],Z[7],Z[3],Z[0]]]

# Plot the cube in 3D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.75))
ax.scatter(Z[:,0], Z[:,1], Z[:,2])
ax.set_xlim(0,11), ax.set_ylim(0,11), ax.set_zlim(0,11)
ax.set_aspect('equal') # uniform scale axes

#Use the same method from before to project the 3D points to 2D
pts2d = None 
pts2d = []
K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float32)
for pt in Z:
    mult = np.matmul(K , pt)
    pts2d.append([mult[0]/mult[2] , mult[1]/mult[2]])