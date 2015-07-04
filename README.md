# Computer-Vision-Experiments

Objective:
The objective of these works are understand some classical computer vision problems like feature point extraction detection and compute homographies ect. I am starting with basic poblems and eventually I will upload more complicated problems in CV. 

Harris Corner Detection and Matching

This code will detect corner points of image by using Harris corner detection method, and find 
the matches between the two images by comparing neighbor window of each corner points by sum of 
squared differences(SSD) and normalized cross correlation method(NCC). This developed by using opencv 
functions (basic functions for reading and writing of images and sobel edge detector also used).Numpy 
array operations are backed by high speed C and FORTRAN more over we can process each step in parallel 
way in numpy. This implementation will take very much less time for executing rather than is sequential 
implementation.

Usage: python harris_corner_detection_and_matching.py

Algo Explanation: http://cannibal-eshafeeqe.blogspot.in/2014/03/harris-corner-detection-and-matching.html

SURF Features Matching

The features detected by SURF(Speeded Up Robust Features) detector (directly used opencv library). 
And the matching of these features found by measuring sum of squared differences of the discriptor 
values associated with each features this will be called as match score between two features. 
Matching will be done between the features having minimum score. And if we found more than one close 
match then such matches will be ignored for finding reliable matched points.

Algo Explanation: http://cannibal-eshafeeqe.blogspot.in/2014/03/surf-feature-detection-and-matching.html

SURF Reliable Matching using RANSAC algorithm 

Here We removing outlier matches and finding reliable matches using a RANSAC algorithm by calculating homogrphy matrix between the 
correspondance points

Image Panorama(Computer Vision apporach)

By using reliable matched points we are calculating a overdetermined homography matrix. These kind of matrixes 
can be used for image mosacing.

Algo Explanation: http://cannibal-eshafeeqe.blogspot.in/2014/03/normal-0-false-false-false-en-in-x-none.html
