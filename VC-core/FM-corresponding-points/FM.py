'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0 )
parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
parser.add_argument("--debug", type=int, default=0)
args = parser.parse_args()

print(args)

#append extra dimension to points and set value as 1 for the new dimension
def get_appended_points(pts):
    N_pts, N_axes = pts.shape
    appended_pts = np.zeros((N_pts,N_axes+1))
    appended_pts[:,:N_axes] = pts
    appended_pts[:,N_axes] = 1
    return appended_pts

#function to normalize input points based on process outlined in question
def normalize_points(pts):
    #compute centroid
    N_pts, N_axes = pts.shape
    centroid = np.zeros(2)
    (centroid[0], centroid[1]) = (np.mean(pts[:,0]), np.mean(pts[:,1]))
    #compute mean distance from centroid
    mean_d = 0
    for counter in np.arange(0, N_pts):
        dist = np.linalg.norm(centroid - pts[counter,:])
        mean_d = mean_d + dist
    mean_d = mean_d/N_pts
    #construct matrix
    const1 = np.sqrt(2)/mean_d
    M = np.array([[const1, 0, -1*centroid[0]*const1],[0, const1, -1*centroid[1]*const1],[0, 0, 1]])     
    #transform points
    norm_pts = get_appended_points(pts)
    #M is 3x3. norms_pts is N_ptsx3. So take transpose of norm_pts for multiplication.
    norm_pts = M.dot(norm_pts.T)
    return (M, norm_pts)

#function to compute SVD of input matrix
def computeSVD(A):
    #compute A.At and At.A
    A_At = A.dot(A.T)
    At_A = (A.T).dot(A)
    #get respective eigen values and eigenvectors
    w1, v1 = np.linalg.eig(A_At)
    w2, v2 = np.linalg.eig(At_A)
    #arrange eigen values and eigen vectors in descending order
    w1_sort_idx = (np.argsort(w1))[::-1]
    w2_sort_idx = (np.argsort(w2))[::-1]
    w1 = w1[w1_sort_idx]
    w2 = w2[w2_sort_idx]
    v1 = v1[:,w1_sort_idx]
    v2 = v2[:,w2_sort_idx]
    v2 = v2.T
    #construct S
    S = np.zeros(A.shape)
    w1_len=len(w1)
    w2_len=len(w2)
    w_len = w1_len
    if w2_len < w1_len: w_len = w2_len
    for index in np.arange(w_len):
        S[index, index] = math.sqrt(abs(w1[index]))  
    #return results
    return (v1, S, v2)    

#function to get sample indices for RANSAC
def get_sample_indices(pts1):
    N_pts = pts1.shape[0]
    sample_size = 8
    sample_indices = np.random.choice(N_pts, size=sample_size)
    return sample_indices

#function for 8 point algorithm to determine fundamental matrix
def FM_by_normalized_8_point(pts1,  pts2):
    #check input points and initialise F
    N_pts, N_axes = pts1.shape
    assert N_pts > 0
    assert N_axes == 2
    assert pts1.shape == pts2.shape
    F = np.zeros((3,3))
    #normalise points. M will be 3x3 and norm_pts will 3xN_PTS
    (M_1, norm_pts1) = normalize_points(pts1)
    (M_2, norm_pts2) = normalize_points(pts2)
    #construct coefficient matrix
    Y = np.zeros((9,N_pts))
    for counter in np.arange(0,N_pts):
        pt1 = norm_pts1[:,counter]
        pt2 = norm_pts2[:,counter]
        Y[:3,counter] = pt2[0]*pt1
        Y[3:6,counter] = pt2[1]*pt1
        Y[6:,counter] = pt2[2]*pt1   
    #solve using SVD. Procedure outlined in wiki article
    (U, S, Vt) = computeSVD(Y)
    lsv = U[:,8]
    E_est = lsv.reshape((3,3))
    (U, S, Vt) = computeSVD(E_est)
    S_2 = np.zeros((3,3))
    S_2[0,0] = S[0,0]
    S_2[1,1] = S[1,1]
    f_bar = (U.dot(S_2)).dot(Vt)
    F = ((M_2.T).dot(f_bar)).dot(M_1)
    F = (1.0/F[2,2])*F
    # F:  fundamental matrix
    return  F

#function for RANSAC approach to determine fundamental matrix
def FM_by_RANSAC(pts1,  pts2):	
    #check input points. Initialise F and mask
    N_pts, N_axes = pts1.shape
    assert N_pts > 0
    assert N_axes == 2
    assert pts1.shape == pts2.shape
    F = np.zeros((3,3))
    mask = np.zeros((N_pts,1))
    #loop a fixed number of times and perform sampling, assess points against threshold
    N_iterations = 500
    threshold = 0.035
    best_inlier_count = 0
    for i in np.arange(N_iterations):
        #get a set of 8 random matching points
        sample_indices = get_sample_indices(pts1)
        pts1_sample = pts1[sample_indices,:]
        pts2_sample = pts2[sample_indices,:]
        #call 8 pt function to get FM for the sample
        FM = FM_by_normalized_8_point(pts1_sample,pts2_sample)
        #append 1 to the points. Needed to compute distance values
        pts1_appended = get_appended_points(pts1)
        pts2_appended = get_appended_points(pts2)
        #compute distance values for all points, determine the inliers, set the mask
        current_mask = np.zeros((N_pts,1))
        inlier_cnt = 0
        for j in np.arange(N_pts):
            pt1_a = pts1_appended[j,:]
            pt2_a = pts2_appended[j,:]
            dist_val = abs((pt2_a.dot(FM)).dot(pt1_a.T))
            if dist_val < threshold : 
                current_mask[j,0] = 1
                inlier_cnt = inlier_cnt + 1       
        #check if this FM is better than the best one until now
        if inlier_cnt > best_inlier_count:
            best_inlier_count = inlier_cnt
            idx = current_mask.reshape(pts1.shape[0])
            #recompute F with inlier points
            F = FM
            #set mask for current iteration as mask for RANSAC
            mask = current_mask

    # F: fundamental matrix
    # mask: whether the points are inliers
    return  F, mask

	
img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)  

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
debug_impl_msg = "FM by implementation\n"
debug_inbuilt_msg = "FM by inbuilt function\n"
if args.UseRANSAC:
    F,  mask = FM_by_RANSAC(pts1,  pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    if args.debug:
        print(debug_impl_msg, F)
        F_inbuilt, mask_inbuilt = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )
        print(debug_inbuilt_msg, F_inbuilt)	
else:
    F = FM_by_normalized_8_point(pts1,  pts2)
    if args.debug:
        print(debug_impl_msg, F)
        F_inbuilt, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
        print(debug_inbuilt_msg, F_inbuilt)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	
	
# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)
plt.show()
