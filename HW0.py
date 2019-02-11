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
plt.imshow(image[0:200,0:200])

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


# TODO: your code here
grayImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
grayImage = grayImage.astype(np.float32) / 255.

boxFilter = cv2.boxFilter(grayImage , ddepth = -1 , ksize=(11 , 11))
gaussFilter = cv2.GaussianBlur(grayImage , ksize=(11 , 11) , sigmaX=-1)

boxRes = grayImage - boxFilter
gaussRes = grayImage - gaussFilter

f = plt.figure(figsize=(20, 7.5))
plt.subplot(1,2,1) , plt.title('Using Box Filter 11X11') , plt.imshow(boxRes, cmap='gray') , plt.colorbar()
plt.subplot(1,2,2) , plt.title('Using Gauss Filter 11X11') , plt.imshow(gaussRes, cmap='gray') , plt.colorbar()

# Different kernel size
grayImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
grayImage = grayImage.astype(np.float32) / 255.

boxFilter = cv2.boxFilter(grayImage , ddepth = -1 , ksize=(31 , 31))
gaussFilter = cv2.GaussianBlur(grayImage , ksize=(31 , 31) , sigmaX=-1)

boxRes = grayImage - boxFilter
gaussRes = grayImage - gaussFilter

f = plt.figure(figsize=(20, 7.5))
plt.subplot(1,2,1) , plt.title('Using Box Filter 31X31') , plt.imshow(boxRes, cmap='gray') , plt.colorbar()
plt.subplot(1,2,2) , plt.title('Using Gauss Filter 31X31') , plt.imshow(gaussRes, cmap='gray') , plt.colorbar()


#------------------------------------------------------------------------------
#Colorspaces
#1) Load the astronaut image (RGB)
#
#2) Convert to HSV (cv2.cvtColor)
#
#3) Display the H, S and V components, side-by-side
#------------------------------------------------------------------------------


# TODO: your code here
hsvImage = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
hChannel , sChannel , vChannel = cv2.split(hsvImage);
f = plt.figure(figsize=(20, 10))
plt.subplot(1,3,1) , plt.title('H Component') , plt.imshow(hChannel)
plt.subplot(1,3,2) , plt.title('S Component') , plt.imshow(sChannel)
plt.subplot(1,3,3) , plt.title('V Component') , plt.imshow(vChannel)

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


# TODO: your coded here
pts2d = cv2.projectPoints(pts3d , rvec = (0,0,0) , tvec = (0,0,0) , cameraMatrix = K , distCoeffs = None)    
newPts3d = np.resize(pts3d , (1000,3))
pts2dMatMul = np.matmul(K , np.transpose(newPts3d))
for i in range(1000):
    pts2dMatMul[:,i] = pts2dMatMul[:,i]/pts2dMatMul[2,i]
f = plt.figure(figsize = (20,5))

K = np.array([[500,0,120],[0,500,40],[0,0,1]], dtype=np.float32)
pts2dDiffK = cv2.projectPoints(pts3d , rvec = (0,0,0) , tvec = (0,0,0) , cameraMatrix = K , distCoeffs = None)    

plt.subplot(131) , plt.title("Using cv2.projectPoints") , plt.scatter(pts2d[0][:,:,0] , pts2d[0][:,:,1] , color='red')
plt.subplot(132) , plt.title("Using np.matmul") , plt.scatter(pts2dMatMul[0,:] , pts2dMatMul[1,:] , color='green')
plt.subplot(133) , plt.title("After changeing K") , plt.scatter(pts2dDiffK[0][:,:,0] , pts2dDiffK[0][:,:,1] , color='blue')

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
# TODO: your code here
K = np.array([[800,0,320],[0,800,240],[0,0,1]], dtype=np.float32)
pts2d = cv2.projectPoints(np.float32(Z) , rvec = (0,0,0) , tvec = (0,0,0) , cameraMatrix = K , distCoeffs = None)[0]

faces2d = [[pts2d[0],pts2d[1],pts2d[2],pts2d[3]],
           [pts2d[4],pts2d[5],pts2d[6],pts2d[7]], 
           [pts2d[0],pts2d[1],pts2d[5],pts2d[4]], 
           [pts2d[2],pts2d[3],pts2d[7],pts2d[6]], 
           [pts2d[1],pts2d[2],pts2d[6],pts2d[5]],
           [pts2d[4],pts2d[7],pts2d[3],pts2d[0]]]

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
patches = []
for p in np.array(faces2d):
    patches += [Polygon(np.squeeze(p), True)]

fig=plt.figure()
ax = fig.add_subplot(111)
ax.add_collection(PatchCollection(patches, alpha=0.1, linewidths=1, edgecolors='r'))
ax.scatter(pts2d[:,0,0],pts2d[:,0,1],c=-Z[:,2])
ax.set_xlim(0,640)
ax.set_ylim(0,480)

#Try to change the translation of the 3D points, as well as the K matrix, and see how it affects the 2D projection.

# cube veritces
Z = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
Z[:,0] += 5 # translate on X
Z[:,1] += 5 # translate on Y
Z[:,2] += 10 # translate on Z

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

# New Value of K
K = np.array([[500,0,0],[0,300,0],[0,0,1]], dtype=np.float32)
pts2d = cv2.projectPoints(np.float32(Z) , rvec = (0,0,0) , tvec = (0,0,0) , cameraMatrix = K , distCoeffs = None)[0]

# list of 2D faces
faces2d = [[pts2d[0],pts2d[1],pts2d[2],pts2d[3]],
           [pts2d[4],pts2d[5],pts2d[6],pts2d[7]], 
           [pts2d[0],pts2d[1],pts2d[5],pts2d[4]], 
           [pts2d[2],pts2d[3],pts2d[7],pts2d[6]], 
           [pts2d[1],pts2d[2],pts2d[6],pts2d[5]],
           [pts2d[4],pts2d[7],pts2d[3],pts2d[0]]]

# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
patches = []
for p in np.array(faces2d):
    patches += [Polygon(np.squeeze(p), True)]

fig=plt.figure()
ax = fig.add_subplot(111)
ax.add_collection(PatchCollection(patches, alpha=0.1, linewidths=1, edgecolors='r'))
ax.scatter(pts2d[:,0,0],pts2d[:,0,1],c=-Z[:,2])
ax.set_xlim(0,640)
ax.set_ylim(0,480)