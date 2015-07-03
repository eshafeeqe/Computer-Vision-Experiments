#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys , os , random
import cv, cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
 
SURF_THRESHOLD = 1000
SURF_MATCH_THRESHOLD = 0.3
 
INLIER_DISTANCE_THRESHOLD = 30
NUMBER_OF_INLIERS_THRESHOLD = 0.8
 
 
 
def SURFdetector ( image ):
	global SURF_THRESHOLD
	detector = cv2.SURF(SURF_THRESHOLD, 10, 10)
	keypoints, descriptors = detector.detectAndCompute(image, None)
	return keypoints, descriptors
 
def SURFmatcher(keypoints_set , descriptors_set):
	keypoint_1 = keypoints_set[0]
	keypoint_2 = keypoints_set[1]
	descriptor_1 = descriptors_set[0]
	descriptor_2 = descriptors_set[1]
 
	diff = descriptor_2 - descriptor_1[:,None]
	squre_of_diff = diff ** 2
	sum_square_diff = squre_of_diff.sum(axis=-1)
	score = np.sqrt(sum_square_diff.min(axis=-1))
	matches = np.argmin(sum_square_diff,axis=-1)
	invalid_matches = score > SURF_MATCH_THRESHOLD 
	
	score[invalid_matches] = -1
	matches[invalid_matches] = -1
 
	return matches , score
 
def  correspondace_map(features_1, features_2, matches):
	
	def feature_to_point(keypoint):
		x,y = keypoint.pt
		return x , y
 
	valid_matches_bool = matches != -1
	valid_matches = matches[valid_matches_bool]
	feature_to_array = np.vectorize(feature_to_point)
 
	features1 = (np.asarray(feature_to_array(features_1))).T
	features2 = (np.asarray(feature_to_array(features_2))).T
 
	valid_features1 = features1[valid_matches_bool]
	valid_features2 = features2[valid_matches,:]
 
	correspondace = np.hstack((valid_features1, valid_features2))
	return correspondace
 
def HomograpgyCalculation(correspondace, for_ransac=True):
	
	def Making_A(point):
		x,y,x1,y1 = point.item(0), point.item(1), point.item(2), point.item(3)
		return x,y,1,0,0,0,-(x*x1),-(x1*y),0,0,0,x,y,1,-(x*y1),-(y*y1)
	
	desired_correspondaces = 0
	if for_ransac == True:
		random_indices = np.array(random.sample(range(correspondace.shape[0]), 8))
		desired_correspondaces = correspondace[random_indices,:]
	else :
		desired_correspondaces = correspondace
 
	#print desired_correspondaces.shape
 
	B = desired_correspondaces[:,2:4].flatten()
	A = np.apply_along_axis(Making_A,1,desired_correspondaces).reshape(-1,8)
 
 
	#Computing over determined solution using psudo inverse of A
	A_psudo_inverse = np.dot ( np.linalg.inv( np.dot ( A.T , A ) ) , A.T )
	x = np.dot ( A_psudo_inverse , B )
 
	H = (np.append(x,1)).reshape(3,3)
	
	return H
 
def RANSAC ( features_set, matches, H ):
		
	features1 = features_set[0]
	features2 = features_set[1]
	
	correspondace = correspondace_map(features1 , features2 , matches)
	number_of_correspondences = correspondace.shape[0]
 
	inlier_set = 0
	outliers = 0
	best_inliers = 0
	
	trials = 0
	N = 1e3
	max_inliers = 10
	min_variance = 1e10
 
	while trials < N :
 
		temp_H = HomograpgyCalculation(correspondace)
 
		correspondace_img1 = correspondace[:,0:2]
		correspondace_img2 = correspondace[:,2:4]
 
		#padding one at the end
		correspondace_img = np.ones((correspondace_img1.shape[0],correspondace_img1.shape[1]+1))
		correspondace_img[:,:-1] = correspondace_img1
		#multiplying with temp 
		correspondace_mul_H = np.dot(temp_H,correspondace_img.T).T
		correspondace_H = (np.divide(correspondace_mul_H[:,0:2].T,correspondace_mul_H[:,2])).T
 
		#print np.dot(H,correspondace_img.T),T
		diff = correspondace_img2 - correspondace_H
		#print diff
		error = (((diff[:,0]**2) + (diff[:,1]**2))**0.5)
 
		inlier_indices = error < INLIER_DISTANCE_THRESHOLD
		inlier_set = correspondace[inlier_indices,:]
		number_of_inliers = inlier_set.shape[0] 
 
		#print number_of_inliers
 
		if number_of_inliers > max_inliers :
			error_mean = (error.sum(axis=-1))/number_of_inliers
			variance = ((error**2).sum(axis=-1)) - error_mean
			if variance < min_variance :
				max_inliers = number_of_inliers
				min_variance = variance
				H = temp_H
				best_inliers = inlier_set
				outliers_indices = np.logical_not(inlier_indices)
				outliers = correspondace[outliers_indices,:]
 
		#Update N and no of trials
		trials +=1 
 
		if number_of_inliers > 0 :
			e = 1.0 - float ( number_of_inliers )/ float ( number_of_correspondences )
			e_1 = 1.0 - e
			if e_1 == 1:
				break
			if np . log (1.0 - e_1 * e_1 * e_1 * e_1 * e_1 * e_1 * e_1 * e_1 ) !=0:
				N = int (np. log (1.0-0.99) /np . log (1.0- e_1 * e_1 * e_1 * e_1 * e_1 * e_1 * e_1 * e_1 ) )		
 
		if float ( number_of_inliers ) / float ( number_of_correspondences ) < NUMBER_OF_INLIERS_THRESHOLD \
			and trials > N:
			trials = 1
 
 
	#print H
	#print correspondace
	#print best_inliers
 
	Homograpgy_best_inliers =  HomograpgyCalculation(best_inliers,for_ransac = False)
 
	return best_inliers, outliers, Homograpgy_best_inliers
 
 
def main ( ) :
 
	image =[]
	keypoints_set = []
	descriptors_set = []
	H = 0
 
	#loading images
	for i in range (len ( sys . argv ) - 1 ):
		filename = sys . argv [ i + 1 ]
		image.append(cv2.imread (filename))
 
	width = image[0].shape[1]
	height = image[0].shape[0]
 
	surf_features = np.zeros((height , len(image)*width  ,3) ,np.uint8)
	surf_matches = np.zeros((height , len(image)*width  ,3) ,np.uint8)
	
	for i in range(len(image)):
 
		keypoints, descriptors = SURFdetector( image[i])
		
		keypoints_set.append(keypoints)
		descriptors_set.append(descriptors)
 
		surf_features[0:height , i*width : (i + 1)*width , :] = image[i]
	
		for j in range ( len ( keypoints ) ) :
			x , y = keypoints[j].pt
			cv2.circle ( surf_features , ( (i*width) + int (x) , int (y) ) , 0 , (255 , 0 , 0) , 4)
	
	cv2.imwrite('surf_features.png', surf_features)
	
	surf_best_inliers = surf_features
	outliers_image = surf_features
 
	matches, score = SURFmatcher(keypoints_set,descriptors_set)
	best_inliers, outliers, Homograpgy_best_inliers = RANSAC(keypoints_set, matches, H)
	
 
 
	
	for i in range ( len ( best_inliers ) ) :
		cv2.line (surf_best_inliers , ((int(best_inliers[i][0])) , (int(best_inliers[i][1]))), \
				((int(best_inliers[i][2]))+width , (int(best_inliers[i][3]))), \
				(255*( i%4) ,255*(( i+1)%4) , 255*(( i+2)%4) ) , 1 , cv2.CV_AA, 0)
 
	cv2.imwrite('surf_best_inliers.png', surf_best_inliers)
 
	for i in range ( len ( outliers ) ) :
		cv2.line (outliers_image , ((int(outliers[i][0])) , (int(best_inliers[i][1]))), \
				((int(outliers[i][2]))+width , (int(outliers[i][3]))), \
				(255 ,255 , 255 ) , 1 , cv2.CV_AA, 0)
	
	cv2.imwrite('surf_outliers.png', outliers_image)
 
 
main()
