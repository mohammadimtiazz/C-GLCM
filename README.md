# C-GLCM
C++ source code of Grey Level Co-occurrence Matrix

This file contains source code for the texture measurment algorithm of "gray level co-occurence matrix (GLCM)" 
This code can produce result as exactly same as the "python scikit lib of GLCM"

This algorithm also reflects the implementation of the following paper 
"Textural Features for Image Classification"

The author "James Darrell McCauley" of this paper upload a Texture measurement code in this following link 
https://github.com/wnd-charm

I used this code and modify functions to create GLCM function for i own use.
The file "C_GLCM.cpp" contains all the required functions for measuring texture from different angle and distance.
For expamle a main function has been written to demonstrate how different texture features can be measured using GLCM for distance 1 and angle 0


Possible linking error can occure if these following libs are not included in Properties-> Linker -> additional dependencies

cv.lib
cxcore.lib
highgui.lib

A sample image is also provided for illustrates the GLCM tutorial of this following link 
http://www.fp.ucalgary.ca/mhallbey/tutorial.htm

the samle image is also creted based on the example of the above link
z.bmp image has 8-bit data of 
0  0  1  1
0  0  1  1  
0  2  2  2
2  2  3  3 
