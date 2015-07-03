#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys , os , random
import cv, cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
 
SURF_THRESHOLD = 1000
SURF_MATCH_THRESHOLD = 0.4
 
INLIER_DISTANCE_THRESHOLD = 50
NUMBER_OF_INLIERS_THRESHOLD = 0.4
 
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
 
def main ( ) :
 
	image =[]
	keypoints_set = []
	descriptors_set = []
 
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
	surf_matches = surf_features
 
	matches, score = SURFmatcher(keypoints_set,descriptors_set)
	for i in range ( len ( matches ) ) :
		match = matches.item(i)
		if match != -1:
 
			cv2.line (surf_matches , (int(((keypoints_set[0][i]).pt)[0]) , int(((keypoints_set[0][i]).pt)[1])), \
				(int(((keypoints_set[1][match]).pt)[0])+width , int(((keypoints_set[1][match]).pt)[1])), \
				(255*( i%4) ,255*(( i+1)%4) , 255*(( i+2)%4) ) , 1 , cv2.CV_AA, 0)
 
	cv2.imwrite('surf_matches.png', surf_matches)
 
 
 
main()
