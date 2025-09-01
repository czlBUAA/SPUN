'''
MIT License

Copyright (c) 2021 SLAB Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import numpy as np

def _to_numpy_image(image):
    return image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()

def imshow(image, savefn=None):
    ''' Show image
    Arguments:
        image: (3,H,W) torch.tensor image
    '''
    # image to numpy array
    image = _to_numpy_image(image)

    # plot
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight', pad_inches=0)

def plot_2D_bbox(image, bbox, xywh=True):
    ''' Show image with a bounding box
    Arguments:
        image: (3,H,W) torch.tensor image
        bbox:  (4,) numpy.ndarray bounding box
        xywh:  If True,  bounding box is in [xcenter, ycenter, width, height]
               If False, bounding box is in [xmin, xmax, ymin, ymax]
    '''
    # Processing
    data = _to_numpy_image(image)

    if xywh:
        x, y, w, h = bbox
        xmin, xmax, ymin, ymax = x-w/2, x+w/2, y-h/2, y+h/2
    else:
        xmin, xmax, ymin, ymax = bbox

    # figure
    fig = plt.figure()
    plt.imshow(data)
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
                color='g', linewidth=1.5)
    plt.show()

def scatter_keypoints(image, x_pr, y_pr, normalized=False):
    ''' Show image with keypoints
    Arguments:
        image: (3,H,W) torch.tensor image
        x_pr:  (11,) numpy.ndarray
        y_pr:  (11,) numpy.ndarray
        normalized: True if keypoints are normalized w.r.t. image size
                    False if keypoints are in pixels
    '''
    _, h, w = image.shape
    data  = _to_numpy_image(image)

    if normalized:
        x_pr = x_pr * w
        y_pr = y_pr * h

    # figure
    fig = plt.figure()
    plt.imshow(data)
    plt.scatter(x_pr , y_pr, c='lime', marker='+')
    plt.show()
def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. 
    Arguments:
        q: (4,) numpy.ndarray - unit quaternion (scalar-first)
    Returns:
        dcm: (3,3) numpy.ndarray - corresponding DCM
    """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm
def quat2SO3(q):
    """ Convert left-handed quaternion (JPL convention) to SO3 according to
        Trawny and Roumeliotis "Indirect kalman filter for 3d attitude estimation" (valid)"""

    R = np.matrix([[1 - 2*q[1]**2 - 2*q[2]**2, 2*(q[0]*q[1] + q[2]*q[3]), 2*(q[0]*q[2] - q[1]*q[3])],
                   [2*(q[0]*q[1] - q[2]*q[3]), 1 - 2*q[0]**2 - 2*q[2]**2, 2*(q[1]*q[2] + q[0]*q[3])],
                   [2*(q[0]*q[2] + q[1]*q[3]), 2*(q[1]*q[2] - q[0]*q[3]), 1 - 2*q[0]**2 - 2*q[1]**2]])
def visualize_axes(q_vbs2tango, r_Vo2To_vbs, cameraMatrix, scale, image):
    ''' Project keypoints.
    Arguments:
        q_vbs2tango:  (4,) numpy.ndarray - unit quaternion from VBS to Tango frame
        r_Vo2To_vbs:  (3,) numpy.ndarray - position vector from VBS to Tango in VBS frame (m)
        cameraMatrix: (3,3) numpy.ndarray - camera intrinsics matrix
        distCoeffs:   (5,) numpy.ndarray - camera distortion coefficients in OpenCV convention
        keypoints:    (3,N) or (N,3) numpy.ndarray - 3D keypoint locations (m)
    Returns:
        points2D: (2,N) numpy.ndarray - projected points (pix)
    '''
    keypoints = np.matrix([[1,0,0],
                   [0,1,0],
                   [0,0,1]])
    orign = np.matrix([0,0,0])
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)
    if orign.shape[0] != 3:
        orign = np.transpose(orign)
    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))
    orign = np.vstack((orign, np.ones((1, orign.shape[1]))))
    
    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q_vbs2tango)),
                          np.expand_dims(r_Vo2To_vbs, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]
    orign_xyz =np.dot(pose_mat, orign)
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:]# [1 x N] each
    xorign, yorign   = orign_xyz[0,:] / orign_xyz[2,:], orign_xyz[1,:] / orign_xyz[2,:]
    
    # apply camera matrix
    p = np.vstack((cameraMatrix[0,0]*x0 + cameraMatrix[0,2],
                          cameraMatrix[1,1]*y0 + cameraMatrix[1,2]))
    c = np.vstack((cameraMatrix[0,0]*xorign + cameraMatrix[0,2],
                          cameraMatrix[1,1]*yorign + cameraMatrix[1,2]))
    
    v = p-c
    #unit vector
    #er fanshu
    v = scale*v/np.linalg.norm(v)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color='r')
    ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color='g')
    ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color='b')
def visualize_axes_urso(q, C, K, scale, image):
    ''' Project keypoints.
    Arguments:
        q_vbs2tango:  (4,) numpy.ndarray - unit quaternion from VBS to Tango frame
        r_Vo2To_vbs:  (3,) numpy.ndarray - position vector from VBS to Tango in VBS frame (m)
        cameraMatrix: (3,3) numpy.ndarray - camera intrinsics matrix
        distCoeffs:   (5,) numpy.ndarray - camera distortion coefficients in OpenCV convention
        keypoints:    (3,N) or (N,3) numpy.ndarray - 3D keypoint locations (m)
    Returns:
        points2D: (2,N) numpy.ndarray - projected points (pix)
    '''
    # Create arrow points
    P = np.matrix([[1,0,0],
                   [0,1,0],
                   [0,0,1]])

    # Rotate arrow points
    R = quat2SO3(q)
    #R = se3lib.euler2SO3(-102.6949968, -33.6145586, 36.8984239)
    #q = se3lib.SO32quat(R)
    #print("q is", q)
    #q = se3lib.euler2quatt(-225, -45, 255)
    #print("q is", q)
    #q = np.array(q).transpose()
    #print("R is", R)
    #print("q is", q)
    #q = np.asarray(se3lib.euler2quat(90, 0, 0))
    #q = np.matrix(q).transpose()
    #print("q is", q)
    #p, y, r = se3lib.quat2euler(q)
    #print("pyr is", p, y, r)
    #v1, the = se3lib.quat2angleaxis(q)
    #print("v1the is", v1, the)
    #R = np.matrix(R).transpose()
    P_r = R*P
    # Translate to loc_gt
    P_t = np.asarray(P_r)+np.transpose([C])
    # Project to image
    p = P_t/P_t[-1,:]
    c = C/C[-1]
    p = K*p
    c = K*np.matrix(c).transpose()
    
    v = p-c
    #unit vector
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    #er fanshu
    v = scale*v/np.linalg.norm(v)
    
    ax.arrow(c[0, 0], c[1, 0], v[0, 0], v[1, 0], head_width=10, color='r')
    ax.arrow(c[0, 0], c[1, 0], v[0, 1], v[1, 1], head_width=10, color='g')
    ax.arrow(c[0, 0], c[1, 0], v[0, 2], v[1, 2], head_width=10, color='b')
