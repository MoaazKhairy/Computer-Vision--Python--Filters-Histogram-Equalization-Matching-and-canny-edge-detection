# sbe404-cvtoolbox-ver1-group_5
sbe404-cvtoolbox-ver1-group_5 created by GitHub Classroom
<h1><center>Task 2 CV</center></h1> <br>
<h3><center> Group 5 </center></h3> <br>
# Names: <br>
## Abdelrahman Ahmed Ramzy  <br>
## Ahmed Fawzy <br>                                   
## Moaz Khairy  <br>                                     
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
## Implementation: 
### - We use PyQT graph for GUI <br>
### - User browses image file, in Code we do "OpenedFile" function which takes the directory of selected <br> image when press load image button <br>
### - Filter tab: 
![filter tab in GUI](https://i.imgur.com/n1uv3dF.png "picture shows filter tab shape in GUI") <br>
### - Spatial filters:
> - ### Edge detection filters: <br> 
>> - first we convert image to gray image and show it in input image graph. <br>
>> - we show histogram beside each image in graphs (input and output images). <br>
>> - we use "convolve2d" function to apply convolution on original image with kernels of each filter. <br>
>> - we made menu to allow user to choose from it type of filter. <br>
>> - perwitt filter, we convolve image with x_perwitt kernel and y_perwitt kernel, then get magnitude and direction, then show magnitude in output image and this is result. <br>
![perwitt filter](https://i.imgur.com/IkATbFP.png "picture shows perwitt filter")
<br>
>> - sobel filter, we convolve image with x_sobel kernel and y_sobel kernel, then get magnitude and direction, then show magnitude in output image and this is result.
![sobel filter](https://i.imgur.com/SCO8aC3.png "picture shows sobel filter")
<br>
>> - laplacian filter, we convolve image with laplacian kernel and this is result.
![laplacian filter](https://i.imgur.com/iJQi5I3.png "picture shows laplacian filter")
<br>
>> - LoG filter, we convolve image with gaussian 3x3 filter then convolve result with laplacian filter and this is result.
![LoG filter](https://i.imgur.com/Btonpyf.png "picture shows LoG filter")
<br>
>> - DoG filter, we convolve image with gaussian 3x3 filter and convolve original image with gaussian 5x5 kernel then subtract two results and this is result.
![DoG filter](https://i.imgur.com/dnhqLHo.png "picture shows DoG filter")
<br>

> - ### Noise reduction filters: <br> 
>> - gaussian filter, we give 2 options for user to choose which optimum method to apply gaussian filter:
>>> 1) choose from menu gaussian 3x3 or 5x5 kernels. <br>
>>> 2) user enters manually value of standerd diviation (sigma) and prefered kernel size.
![gaussian3x3 filter](https://i.imgur.com/y24M8Hz.png "picture shows gaussian3x3 filter")
![gaussian5x5 filter](https://i.imgur.com/NZ3Vs2M.png "picture shows gaussian5x5 filter")
<br>
>> - box filter, we convolve image with box kernel and this is result
![box filter](https://i.imgur.com/jwtP4EP.png "picture shows box filter")
<br>
>> - median filter, we made empty zero kernel 3x3 array called "Median_Variable" , and made for loop with i and j indices to cover all image which called " Median_image" variable then passes this kernel on image within for loop, each 3x3 pixels in image we save it in "Median_variable", sort 9 elements, then put middle element (number 4) after sorting instead of old middle element in kernel and repeat it on all image. this is result:
![median filter](https://i.imgur.com/sdhKdXO.png "picture shows median filter")
<br>

> - ### Sharpening filter: <br> 
>> - Sharpening filter, we convolve image with Sharpening kernel and this is result
![Sharpening filter](https://i.imgur.com/2pUvz2K.png "picture shows Sharpening filter")
<br>

### - Frequency domain transform:
> - first we implementes FFT by coding without using built-in function as follows:
![FFT hand made](https://i.imgur.com/BjQghOo.jpg "picture shows FFT hand made") 
<br>
- and we get same results from built-in FFT function as follows:
![result of FFT hand made](https://i.imgur.com/2egCOo5.jpg "picture shows result of FFT hand made")
<br>
- but this our implemented FFT takes more time for processing at running which is inEfficient due to 4 for loops inside each other so, we used built-in FFT in our code.
<br>

### - Frequency domain filters:
> - we implemented high, low and band pass filters.
> - first we convert image to gray level image.
> - we apply fft transform and fftshift to image.
> - main idea depends on suppressing part of image we want.
> - then convert it again back to spatial domain by ifftshift then ifft.

> ### - High-Pass filter:
> we suppress some of pixels which at middle of image that zeros on it, and this is result:
![High pass filter](https://i.imgur.com/ZF6htjs.png "picture shows High pass filter")

> ### - Low-Pass filter:
> we suppress some of pixels which at edges of image that zeros on it, and this is result:
![Low pass filter](https://i.imgur.com/BWzUgmD.png "picture shows Low pass filter")
> ### - Band-Pass filter:
> we suppress both some of pixels which at edges and middle of image that zeros on it, and this is result:
![Band pass filter](https://i.imgur.com/F7PPXbE.png "picture shows Band pass filter")

### - Histogram:
> This is histogram tab in GUI: 
![Histogram tab](https://i.imgur.com/gE93bgJ.png "picture shows Histogram tab  ")

> In Histogram we create an array to count the number of pixels corresponding to each gray level from 0 to 255
![Histogram](https://i.imgur.com/ZTMymhm.png "picture shows Histogram  ")

### - Histogram Equalization:
> After getting the histogram array we calculate the CDF and normalization function then we looped on the image changing every pixel value by the new one. 
![Histogram Equalization ](https://i.imgur.com/jNeUHtR.png "picture shows Histogram Equalization ")

### - Histogram Matching:
> We get the histogram equalization for both input and source image the we equaled CDF function of the input with the CDF fun of the source at the end we looped on the image changing every pixel value by the new one.
![Histogram Matching ](https://i.imgur.com/zK4BvbQ.png "picture shows Histogram Matching ")

## -Hough transform (lines and circles)
> ### - Canny Edge Detection:
>> - Noise Reduction using Gaussian_Kernel k= 5, Segma = 4
>> - Gradient Calculation using Sobel_Kernal 3*3
>> - Non-Maximum Suppression
>> - Double threshold with:
>>> 1) highThreshold = 120 <br>
>>> 2) lowThreshold  = 30
>> - Edge Tracking by Hysteresis:
>>> 1) we check if the weak pixel is near to any strong one using 5*5 neighbors <br>
>>> 2) we check if the zero pixel has 2 strong neighbors at a specific direction <br>
>>> 3) Repeat step (1&2) three times <br>
>>> 4) we check if the weak or the strong pixel has a zero neighbors from all direction we put it equal zero too <br>
![Canny](https://i.imgur.com/85sqAwb.png "picture shows Canny ")
<br>
> ### - Hough Line:
> - after canny. <br>
> - convert image from spatial domain to rho & theta <br>
> - take intersection point from rho & theta domain of hough graph that indecates line in spatial domain and show it again in original image <br>
> - we detect thresthold of number of lines to avoid long time of processing. <br>
![Hough line](https://i.imgur.com/DKiegwW.jpg "picture shows Hough line ")
![Hough line](https://i.imgur.com/7BDiuKm.jpg "picture shows Hough line ")
> ### - Hough Circle:
>> - This is tab of Circle hough in our GUI with results:
![Hough Circle](https://i.imgur.com/oWfghlm.png "picture shows Hough Circle ")
>> - Using The result from Canny edge detection and giving some parameters to reduce the processing time like maximum & minimum radius and the total number of circiles into the image. <br>
>> - Hough circle detect the shape and drow a circle

## Notes and Issues we faced:
> - we used PyQT for all GUI except for show color output image on Hough circle we used Qlabel, and we do not use Qlabel for all project becauese we must save image in laptop then show it and this is inefficient.
> - in Hough circle we must detect radius, threshold and number of circles to avoid long time in processing and same in line hough we must detect thresthold.
> - not good accuracy


## References:
> [1] https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.convolve2d.html <br>
> [2] https://en.wikipedia.org/wiki/Kernel_(image_processing) <br>
> [3] https://subscription.packtpub.com/book/application_development/9781785283932/2/ch02lvl1sec22/sharpening <br>
> [4] https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering <br>
> [5] https://python-reference.readthedocs.io/en/latest/docs/functions/complex.html <br>
> [6] https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.fft.html <br>
> [7] http://www.pyqtgraph.org/documentation/graphicsItems/imageitem.html#pyqtgraph.ImageItem.setLookupTable <br>
> [8] https://www.afternerd.com/blog/python-lambdas/ <br>
> [9] https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123 <br>
> [10] http://me.umn.edu/courses/me5286/vision/Notes/2015/ME5286-Lecture9.pdf <br>
> [11] https://github.com/PavanGJ/Circle-Hough-Transform/blob/master/main.py <br>
