#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys , os , random
import cv, cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
 
SURF_THRESHOLD = 500
SURF_MATCH_THRESHOLD = 0.3
SURF_RATIO_THRESHOLD = 0.9
 
INLIER_DISTANCE_THRESHOLD = 20
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
	score = np.sqrt(sum_square_diff)
 
	matching_score = np.sort(score,axis=-1)[:,:2]
	ratio = np.true_divide(matching_score[:,0],matching_score[:,1])
 
	matches = np.argmin(sum_square_diff,axis=-1)
	
	invalid_matches = np.logical_or(matching_score[:,0]>SURF_MATCH_THRESHOLD ,ratio>SURF_RATIO_THRESHOLD )
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
 
 
	B = desired_correspondaces[:,2:4].flatten()
	A = np.apply_along_axis(Making_A,1,desired_correspondaces).reshape(-1,8)
 
 
	#Computing over determined solution using psudo inverse of A
	A_psudo_inverse = np.dot ( np.linalg.inv( np.dot ( A.T , A ) ) , A.T )
	x = np.dot ( A_psudo_inverse , B )
 
	H = (np.append(x,1)).reshape(3,3)
	
	return H
 
def RANSAC ( features_set, matches, H):
	
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
 
		diff = correspondace_img2 - correspondace_H
		error = (((diff[:,0]**2) + (diff[:,1]**2))**0.5)
 
		inlier_indices = error < INLIER_DISTANCE_THRESHOLD
		inlier_set = correspondace[inlier_indices,:]
		number_of_inliers = inlier_set.shape[0] 
 
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
 
	Homograpgy_best_inliers =  HomograpgyCalculation(best_inliers, for_ransac = False)
 
	return best_inliers, outliers, Homograpgy_best_inliers
 
def homography_transformation(Homograpgy_best_inliers_set):
	L = len(Homograpgy_best_inliers_set)
	i = int ((L+1)/2) - 2
	while i >= 0:
		Homograpgy_best_inliers_set[ i ] = np.dot(Homograpgy_best_inliers_set[ i] ,Homograpgy_best_inliers_set[ i+1 ] )
		i-=1
	i = int ((L+1)/2) + 2
	while i < (L+1):
		Homograpgy_best_inliers_set[ i ] = np.dot(Homograpgy_best_inliers_set[ i] , Homograpgy_best_inliers_set[ i-1 ] )
		i+=1
 
	return 0
 
def boundary_calculation(Homograpgy_best_inliers_set, width, height):
	
	homography_transformation(Homograpgy_best_inliers_set)
	L = len(Homograpgy_best_inliers_set)
 
	S = np.asarray( Homograpgy_best_inliers_set )
 
	H = np.array([[0,0,width-1,width-1],[0,height-1,0,height-1],[1,1,1,1]])
	
	K = np.dot(S,H)
 
	U = (np.hstack((K[:,2,:],K[:,2,:]))).reshape(3,2,4)
	boundary_image = (np.divide(K[:,0:2,:],U))
 
	image_boundaries_min = boundary_image.min(axis=-1)
	image_boundaries_max = boundary_image.max(axis=-1)
	boundaries = np.hstack((image_boundaries_min,image_boundaries_max))
	image_boundaries = np.insert(boundaries,(L/2)+1,np.array([0,0,width-1,height-1]),0)
 
	cor_min = image_boundaries_min.min(axis=0)
	cor_max = image_boundaries_max.max(axis=0)
	final_image_boundaries = np.hstack((cor_min,cor_max))
 
	return final_image_boundaries, image_boundaries
 
def main ( ) :
	image =[]
	keypoints_set = []
	descriptors_set = []
	H = 0
	best_inliers_set = []
	Homograpgy_best_inliers_set = []
 
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
 
	number_of_images = len(image)
 
	for i in range(number_of_images-1):
		keypoints = keypoints_set[i:i+2]
		descriptors = descriptors_set[i:i+2]
		matches, score = 0,0
		best_inliers, outliers, Homograpgy_best_inliers = 0, 0, 0
 
		if i > ((number_of_images/2)-1):
			keypoints[1] , keypoints[0] = keypoints[0] , keypoints[1]
			descriptors[1] , descriptors[0] = descriptors[0] , descriptors[1]
			matches, score = SURFmatcher(keypoints,descriptors)
			best_inliers, outliers, Homograpgy_best_inliers = RANSAC(keypoints, matches, H)
		else:
			matches, score = SURFmatcher(keypoints,descriptors)
			best_inliers, outliers, Homograpgy_best_inliers = RANSAC(keypoints, matches, H)
 
		best_inliers_set.append(best_inliers)
		Homograpgy_best_inliers_set.append(Homograpgy_best_inliers)
 
	final_image_boundaries, image_boundaries = boundary_calculation(Homograpgy_best_inliers_set, width, height)
	fx_min = final_image_boundaries.item(0)
	fx_max = final_image_boundaries.item(2)
	fy_min = final_image_boundaries.item(1)
	fy_max = final_image_boundaries.item(3)
 
	mosaic_image = np.zeros((int(fy_max -fy_min ), int(fx_max - fx_min) ,3) ,np.uint8)
	mosaic_image2 = np.zeros((int(fy_max -fy_min + 20), int(fx_max - fx_min + 20) ,3) ,np.uint8)
 
	
	H_inverse = []
	count = 0
	for i in range(number_of_images):
		if i == number_of_images/2:
			H_inv = np.eye(3, dtype=float)
			H_inverse.append(H_inv)
		else : 
			H_inv = np.linalg.inv(Homograpgy_best_inliers_set[count])
			H_inverse.append(H_inv)
			count += 1
 
	mosaic = np.indices((int(fy_max - fy_min),int(fx_max - fx_min))).swapaxes(0,2).swapaxes(0,1)[:,:,::-1]
 
	mosaic += [fx_min, fy_min]
	ones_to_add = np.ones((int(fy_max - fy_min),int(fx_max - fx_min)))
 
 
	new_mosaic_set = []
	for i in range(len(image)):
 
		bool_mosaic = np.logical_and ((mosaic < [image_boundaries.item(i,2),image_boundaries.item(i,3)]).all(axis =-1) , \
			(mosaic > [image_boundaries.item(i,0),image_boundaries.item(i,1)]).all(axis=-1))
		bool_mosaic_neg = np.logical_not(bool_mosaic)
		new_mosaic = np.dstack((mosaic , ones_to_add))
	
		
		desired = new_mosaic[bool_mosaic] 
		new_mosaic[bool_mosaic_neg] = [0,0,0]
		image_points = np.dot(H_inverse[i], desired.T).T
 
		image_points_transformed =  (image_points[:,:2].swapaxes(-2,-1)/image_points[:,2]).swapaxes(-1,-2)
	
		condition = np.logical_and((image_points_transformed < [width-1, height-1]).all(axis =-1) , \
			(image_points_transformed > [0,0]).all(axis=-1))
 	
 		condition_neg = np.logical_not(condition)
 
 		image_points[condition_neg] =[0,0,0]
 
 		X_Y = image_points_transformed[condition]
 
 		X_Y_floor = np.floor(X_Y)
 
 		diff = X_Y - X_Y_floor
 
	 	diff_X = diff[:,0]
	 	diff_Y = diff[:,1]
 
	 	indeces_0 = (X_Y_floor[:,::-1]).astype(int)
 
	 	indeces_1 = np.array([0 , 1]) + indeces_0 
	 	indeces_2 = indeces_0 + np.array([1 , 0])
	 	indeces_3 = indeces_0 + np.array([1 , 1])
 
	 	image_points_0 = image[i][indeces_0[:,0],indeces_0[:,1]]
		image_points_1 = image[i][indeces_1[:,0],indeces_1[:,1]]
		image_points_2 = image[i][indeces_2[:,0],indeces_2[:,1]]
		image_points_3 = image[i][indeces_3[:,0],indeces_3[:,1]]
 
		temp_0 = ((1-diff_X)*(1-diff_Y)*(image_points_1.T)).T
		temp_1 = ((diff_X)*(1-diff_Y)*(image_points_1.T)).T
		temp_2 = ((1-diff_X)*(diff_Y)*(image_points_2.T)).T
		temp_3 = ((diff_X * diff_Y)*(image_points_3.T)).T
 
		temp = temp_0 + temp_1 + temp_2 + temp_3
		image_points[condition] = temp
		new_mosaic[bool_mosaic] = image_points
 
		new_mosaic_set.append(new_mosaic)
		cv2.imwrite('bool_mosaic'+str(i)+'.png', new_mosaic)	
 
	
	#for i in range(len(new_mosaic_set)):
 
	#mosaic = new_mosaic_set[0] + 0.8*new_mosaic_set[1] + 0.8*new_mosaic_set[2] + 0.8*new_mosaic_set[3]
 
	condition1 = np.logical_and((new_mosaic_set[0]>np.array([0,0,0])).all(axis=-1),(new_mosaic_set[1]>np.array([0,0,0])).all(axis=-1))
	condition2 = np.logical_and((new_mosaic_set[1]>np.array([0,0,0])).all(axis=-1),(new_mosaic_set[2]>np.array([0,0,0])).all(axis=-1))
	condition3 = np.logical_and((new_mosaic_set[2]>np.array([0,0,0])).all(axis=-1),(new_mosaic_set[3]>np.array([0,0,0])).all(axis=-1))
 
	new_mosaic_set[0][condition1] = 1*new_mosaic_set[0][condition1]
	new_mosaic_set[1][condition1] = 0*new_mosaic_set[1][condition1]
	new_mosaic_set[1][condition2] = 1*new_mosaic_set[1][condition2]
	new_mosaic_set[2][condition2] = 0*new_mosaic_set[2][condition2]
	new_mosaic_set[2][condition3] = 1*new_mosaic_set[2][condition3]
	new_mosaic_set[3][condition3] = 0*new_mosaic_set[3][condition3]
		
	mosaic = new_mosaic_set[0] + new_mosaic_set[1] + new_mosaic_set[2] + new_mosaic_set[3]
 
	cv2.imwrite('mosaiced_image.png', mosaic)	
 
 
		
main()
