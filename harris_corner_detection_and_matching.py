#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys , os , random
import cv, cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
 
SSD_THRESHOLD = 80
SSD_RATIO_THRESHOLD = 0.95
NCC_RATIO_THRESHOLD = 0.95
NCC_THRESHOLD = 0.75
 
HARRIS_CORNER_THRESHOLD = 4e8
 
WINDOW_SIZE = 5
TOTAL_FEATURES = 200
size = 40
 
 
def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)
 
def harrisfeature ( image ) :
	global TOTAL_FEATURES
	global WINDOW_SIZE #size of mask
	global HARRIS_CORNER_THRESHOLD
 
	width = image.shape[1]
	height = image.shape[0]
 
	R_final = np.zeros( ( height,width ), float)
	R_supressed_final = np.zeros( ( height,width ), float)
 
	#order of derivative in x
	order_x = 1
	#order of derivative in y
	order_y = 1
	#using 3x3 sobel operator
	aperturesize = 3 
	
	#grey_scale_image
	gray_scale_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
	image_der_x = cv2.Sobel(gray_scale_image1,cv2.CV_64F,order_x,0,aperturesize)
	image_der_y = cv2.Sobel(gray_scale_image1,cv2.CV_64F,0,order_y,aperturesize)
 
	windows_x = sliding_window(image_der_x, WINDOW_SIZE)
	windows_y = sliding_window(image_der_y, WINDOW_SIZE)
 
	ix = (windows_x * windows_x)/WINDOW_SIZE
	iy = (windows_y * windows_y)/WINDOW_SIZE
	ixy = (windows_x * windows_y)/WINDOW_SIZE
 
	Ix = ix.sum(axis=-1).sum(axis=-1)
	Iy = iy.sum(axis=-1).sum(axis=-1)
	Ixy = ixy.sum(axis=-1).sum(axis=-1)
 
	C = np.vstack(([Ix.T], [Ixy.T], [Ixy.T], [Iy.T])).T
	C_reshaped =  C.reshape(1,-1,2,2)
	U, s, V = np.linalg.svd(C_reshaped, full_matrices=True)
 
	#Sum and product of eigen values
	sum_ =  s.sum(axis=-1)
	prod =  s.prod(axis=-1)
 
	Response = (prod - 0.04*np.power((sum_) , 2)).reshape(-1) 
	Reshaped_Response = Response.reshape(height - WINDOW_SIZE + 1,width - WINDOW_SIZE + 1)  
	R_final[WINDOW_SIZE/2:height - (WINDOW_SIZE/2) , WINDOW_SIZE/2:width - (WINDOW_SIZE/2)] \
	= Reshaped_Response
 
	R_feature = sliding_window(R_final, WINDOW_SIZE)
	R_max = R_feature.max(axis=-1).max(axis=-1)
	
	Reshaped_Response[R_max>Reshaped_Response] = 0
	R_supressed_final[WINDOW_SIZE/2:height - (WINDOW_SIZE/2) , WINDOW_SIZE/2:width - (WINDOW_SIZE/2)]\
	= Reshaped_Response
	features = np.column_stack(np.where(R_supressed_final>HARRIS_CORNER_THRESHOLD))
 
	return features
 
def getNeighbours(image,features):
	global size
	gray_scale_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	windowed_image = sliding_window( gray_scale_image1 , size )
	
	features_x = features[:,0]
	features_y = features[:,1]
 
	desired_features_x = (features_x<(gray_scale_image1.shape[0]-size)) & (features_x > size)
	desired_features_y = (features_y<(gray_scale_image1.shape[1]-size)) & (features_y > size )
	desired_features = np.logical_and(desired_features_x,desired_features_y) 
	
	desired_features = features[desired_features]
	
	neighbors =  windowed_image[desired_features[:,0],desired_features[:,1]]
	return desired_features, neighbors
 
def ssd_(neighbors_set,desired_features_set):
	global size
 
	neighbors_1 = neighbors_set[0]
	neighbors_2 = neighbors_set[1]
 
	subs = (neighbors_2 - neighbors_1[:,None])
	squre_of_subs = subs**2
	average_of_squares = ((squre_of_subs.sum(axis = -1)).sum(axis = -1))/(size*size)
	matches = np.argmin(average_of_squares, axis=-1)
	
	matching_score = np.sort(average_of_squares,axis=-1)[:,:2]
	ratio = np.true_divide(matching_score[:,0],matching_score[:,1])
 
	invalid_matches = np.logical_or(matching_score[:,0]>SSD_THRESHOLD,ratio>SSD_RATIO_THRESHOLD)
 
	matching_score[invalid_matches] = -1
	matches[invalid_matches] = -1
 
	return matches, matching_score[0]
 
def ncc_(neighbors_set,desired_features_set):
	global size
 
	neighbors_1 = neighbors_set[0]
	neighbors_2 = neighbors_set[1]
 
	mean_1 = ((neighbors_1.mean(axis=-1)).mean(axis=-1)).reshape(-1,1,1)
	mean_2 = ((neighbors_2.mean(axis=-1)).mean(axis=-1)).reshape(-1,1,1)
 
	mean_1_sub = neighbors_1 - mean_1
	mean_2_sub = neighbors_2 - mean_2
 
	mul = mean_2_sub * mean_1_sub[:,None]
	numerator = mul.sum(axis=-1).sum(axis=-1)
 
	squre_mean_1_sub = mean_1_sub**2
	squre_mean_2_sub = mean_2_sub**2
 
	sum_1 = squre_mean_1_sub.sum(axis=-1).sum(axis=-1)
	sum_2 = squre_mean_2_sub.sum(axis=-1).sum(axis=-1)
 
	demominator = np.sqrt( sum_2 * sum_1[:,None] )
 
	ncc = numerator/demominator
 
	matching_score = np.sort(ncc,axis=-1)[:,-2:]
	matches = np.argmax(ncc, axis=-1)
	ratio = np.true_divide(matching_score[:,0],matching_score[:,1])
 
	invalid_matches = np.logical_or( matching_score[:,-1] < NCC_THRESHOLD, ratio > NCC_RATIO_THRESHOLD )
 
	matching_score[invalid_matches] = -1
	matches[invalid_matches] =-1
 
	return matches, matching_score[-1]
 
 
def main ( ) :
 
	image =[]
	features_set = []
	desired_features_set =[]
	neighbors_set = []
 
	for i in range (len ( sys . argv ) - 1 ):
		filename = sys . argv [ i + 1 ]
		image.append(cv2.imread (filename))
 
	#Assuming two image widths and heights are same
	width = image[0].shape[1]
	height = image[0].shape[0]
 
	harris_corner = np.zeros((height , len(image)*width  ,3) ,np.uint8)
	harris_corner_desired = np.zeros((height , len(image)*width  ,3) ,np.uint8)
	full_image = np.zeros((height , len(image)*width  ,3) ,np.uint8)
 
	for i in range(len(image)):
		features = harrisfeature ( image[i] )
 
		desired_features, neighbors = getNeighbours( image[i] , features )
		
		features_set.append(features)
		desired_features_set.append(desired_features)
		neighbors_set.append(neighbors)
 
		full_image[0:height , i*width : (i + 1)*width , :] = image[i]
 
		harris_corner[0:height , i*width : (i + 1)*width , :] = image[i]
		for j in range ( features.shape[0]  ) :
			cv2.circle ( harris_corner , ( (i*width)+features.item(j,1) , features.item(j,0)) , 0 , (255 , 0 , 0) , 4)
 
		harris_corner_desired[0:height , i*width : (i + 1)*width , :] = image[i]
		for j in range ( desired_features.shape[0]  ) :
			cv2.circle ( harris_corner_desired , ( (i*width)+desired_features.item(j,1) , desired_features.item(j,0)) , 0 , (255 , 0 , 0) , 4)
 
	cv2.imwrite('harris_corner.png', harris_corner)
	cv2.imwrite('harris_corner_desired.png', harris_corner_desired)
 
	ssd_image = full_image
 
	matches_1, matching_score_1 = ssd_(neighbors_set,desired_features_set)
 
	for i in range ( len ( matches_1 ) ) :
		match = matches_1.item(i)
		if match != -1:
			cv2.line(ssd_image , ((desired_features_set[0]).item(i,1) , (desired_features_set[0]).item(i,0)), \
				((desired_features_set[1]).item(match,1)+width , (desired_features_set[1]).item(match,0)), \
				(255*( i%4) ,255*(( i+1)%4) , 255*(( i+2)%4) ) , 1 , cv2.CV_AA, 0)
	cv2.imwrite('ssd_image.png', ssd_image)
 
	ncc_image = full_image
	matches_2, matching_score_2 = ncc_(neighbors_set,desired_features_set)
 
	for i in range ( len ( matches_2 ) ) :
		match = matches_2.item(i)
		if match != -1:
			cv2.line(ncc_image , ((desired_features_set[0]).item(i,1) , (desired_features_set[0]).item(i,0)), \
				((desired_features_set[1]).item(match,1)+width , (desired_features_set[1]).item(match,0)), \
				(255*( i%4) ,255*(( i+1)%4) , 255*(( i+2)%4) ) , 1 , cv2.CV_AA, 0)
 
 
	cv2.imwrite('ncc_image.png', ncc_image)
 
 
 
main()
