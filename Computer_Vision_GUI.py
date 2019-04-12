# Code for the 1st Computer Vision Task, on 14/4/2019
# Made By Abdelrahman Ahmed Ramzy, Ahmed Fawzi Hosni, Moaz Khairy Hussien



import sys
import numpy as np
import pandas as pd
import os
import argparse
import time
# PyQt5
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QLabel, QMessageBox, QMainWindow, QFileDialog, QComboBox, \
    QRadioButton, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph as pg
# Image Processing
import cv2
from skimage import color
from skimage.transform import resize
# Scipy
from scipy import signal
from scipy import misc
import scipy.fftpack as fp
# Math
import math
from math import sqrt, atan2, pi, cos, sin
from collections import defaultdict

# References
# [1] https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.convolve2d.html
# [2] https://en.wikipedia.org/wiki/Kernel_(image_processing)
# [3] https://subscription.packtpub.com/book/application_development/9781785283932/2/ch02lvl1sec22/sharpening
# [4] https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering
# [5] https://python-reference.readthedocs.io/en/latest/docs/functions/complex.html
# [6] https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.fft.html
# [7] http://www.pyqtgraph.org/documentation/graphicsItems/imageitem.html#pyqtgraph.ImageItem.setLookupTable
# [8] https://www.afternerd.com/blog/python-lambdas/
# [9] https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# [10] http://me.umn.edu/courses/me5286/vision/Notes/2015/ME5286-Lecture9.pdf
# [11] https://github.com/PavanGJ/Circle-Hough-Transform/blob/master/main.
# [12] Sections and Toaa Tarek for Hough Line transform


def OpenedFile(fileName):
    i = len(fileName) - 1
    j = -1
    x = 1

    while x == 1:
        if fileName[i] != '/':
            j += 1
            i -= 1
        else:
            x = 0
    File_Names = np.zeros(j + 1)

    # Convert from Float to a list of strings
    File_Name = ["%.2f" % number for number in File_Names]
    for k in range(0, j + 1):
        File_Name[k] = fileName[len(fileName) - 1 + k - j]  # List of Strings
    # Convert list of strings to a string
    FileName = ''.join(File_Name)  # String
    return FileName


def gaussian_kernel(kernel, std):
    """Returns a 2D Gaussian kernel array."""
    Gaussian_Kernel_1 = signal.gaussian(kernel, std=std).reshape(kernel, 1)
    Gaussian_Kernel_2 = np.outer(Gaussian_Kernel_1, Gaussian_Kernel_1)
    return Gaussian_Kernel_2


def Histogram_Equalization(Gray_image):
    Histogram = np.zeros((4, 256), dtype=int)
    new_Gray_image = np.array(Gray_image.copy(), dtype=int)
    U_max = np.max(Gray_image)
    U_min = np.min(Gray_image)

    Pixel_Count = np.size(Gray_image)

    size = np.shape(Gray_image)
    # First Row
    for i in range(256):
        Histogram[0, i] = i

    # Linear Scaling & Histogram Calculating
    a = -1 * U_min
    b = 255 / (U_max - U_min)

    for i in range(size[0]):
        for j in range(size[1]):
            intensity = math.floor(b * (Gray_image[i, j] + a))
            new_Gray_image[i, j] = intensity
            Histogram[1, intensity] = Histogram[1, intensity] + 1

    # Histogram Equalization
    # apply CDF
    Histogram[2, 0] = Histogram[1, 0]
    for i in range(1, 256):
        Histogram[2, i] = Histogram[2, i - 1] + Histogram[1, i];
    for i in range(256):
        Histogram[2, i] = math.floor(Histogram[2, i] * 255);
    # apply Normalization
    for i in range(256):
        Histogram[3, i] = Histogram[2, i] / Pixel_Count;
    return (Histogram, new_Gray_image)


# Reference [9]
def Canny_Edge_Detection(ImageGRAY):
    # Noise Reduction
    GaussianK = 5
    GaussianSTD = 4
    Gaussian_Kernel_1 = signal.gaussian(GaussianK, std=GaussianSTD).reshape(GaussianK, 1)
    Gaussian_Kernel_2 = np.outer(Gaussian_Kernel_1, Gaussian_Kernel_1)
    Image_Noise_Reduced = signal.convolve2d(ImageGRAY, Gaussian_Kernel_2, boundary='symm', mode='same')


    # Gradient Calculation
    # Edge detection by Sobel then calculate the magnitude and direction
    X_Sobel_Kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Y_Sobel_Kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    X_Sobel_Image = signal.convolve2d(Image_Noise_Reduced, X_Sobel_Kernel, boundary='symm', mode='same')
    Y_Sobel_Image = signal.convolve2d(Image_Noise_Reduced, Y_Sobel_Kernel, boundary='symm', mode='same')
    # numpy.hypot: Equivalent to sqrt(x1**2 + x2**2)
    Image_Gradient = np.hypot(X_Sobel_Image, Y_Sobel_Image)
    Image_Gradient = (Image_Gradient / Image_Gradient.max()) * 255
    theta = np.arctan2(Y_Sobel_Image, X_Sobel_Image) * (180.0 / np.pi)

    # Non-Maximum Suppression
    # The final image should have thin edges.
    # Thus, we must perform non-maximum suppression to thin out the edges
    (M, N) = Image_Gradient.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta.copy()
    angle[angle < 0] += 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = Image_Gradient[i, j + 1]
                    r = Image_Gradient[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = Image_Gradient[i + 1, j + 1]
                    r = Image_Gradient[i - 1, j - 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = Image_Gradient[i + 1, j]
                    r = Image_Gradient[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = Image_Gradient[i + 1, j - 1]
                    r = Image_Gradient[i - 1, j + 1]

                if (Image_Gradient[i, j] >= q) and (Image_Gradient[i, j] >= r):
                    Z[i, j] = Image_Gradient[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    Image_NonMax_Supression = Z.copy()

    # Double threshold
    # The double threshold step aims at identifying 3 kinds of pixels: strong, weak, and non-relevant

    highThreshold = 120
    lowThreshold = 30

    (M, N) = Image_NonMax_Supression.shape
    Image_Double_Threshold = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(50)
    strong = np.int32(255)
    strong_i, strong_j = np.where(Image_NonMax_Supression >= highThreshold)
    zeros_i, zeros_j = np.where(Image_NonMax_Supression <= lowThreshold)
    weak_i, weak_j = np.where((Image_NonMax_Supression < highThreshold) & (Image_NonMax_Supression > lowThreshold))

    Image_Double_Threshold[strong_i, strong_j] = strong
    Image_Double_Threshold[weak_i, weak_j] = weak
    Image_Double_Threshold[zeros_i, zeros_j] = 0

    # Edge Tracking by Hysteresis
    # Hysteresis consists of transforming weak pixels into strong ones,
    # iff at least one of the pixels around the one being processed is a strong one
    (M, N) = Image_Double_Threshold.shape
    Image_Hysteresis = Image_Double_Threshold.copy()
    for sure in range(1, 3):
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (Image_Hysteresis[i, j] == weak):
                    try:
                        if ((Image_Hysteresis[i + 2, j - 2] == strong)
                                or (Image_Hysteresis[i + 2, j - 1] == strong)
                                or (Image_Hysteresis[i + 2, j] == strong)
                                or (Image_Hysteresis[i + 2, j + 1] == strong)
                                or (Image_Hysteresis[i + 2, j + 2] == strong)

                                or (Image_Hysteresis[i + 1, j - 2] == strong)
                                or (Image_Hysteresis[i + 1, j - 1] == strong)
                                or (Image_Hysteresis[i + 1, j] == strong)
                                or (Image_Hysteresis[i + 1, j + 1] == strong)
                                or (Image_Hysteresis[i + 1, j + 2] == strong)

                                or (Image_Hysteresis[i, j - 2] == strong)
                                or (Image_Hysteresis[i, j - 1] == strong)

                                or (Image_Hysteresis[i, j + 1] == strong)
                                or (Image_Hysteresis[i, j + 2] == strong)

                                or (Image_Hysteresis[i - 1, j - 2] == strong)
                                or (Image_Hysteresis[i - 1, j - 1] == strong)
                                or (Image_Hysteresis[i - 1, j] == strong)
                                or (Image_Hysteresis[i - 1, j + 1] == strong)
                                or (Image_Hysteresis[i - 1, j + 2] == strong)

                                or (Image_Hysteresis[i - 2, j - 2] == strong)
                                or (Image_Hysteresis[i - 2, j - 1] == strong)
                                or (Image_Hysteresis[i - 2, j] == strong)
                                or (Image_Hysteresis[i - 2, j + 1] == strong)
                                or (Image_Hysteresis[i - 2, j + 2] == strong)):

                            Image_Hysteresis[i, j] = strong
                        else:
                            Image_Hysteresis[i, j] = 0
                    except IndexError as e:
                        pass
                if (Image_Hysteresis[i, j] == 0):
                    try:
                        if (((Image_Hysteresis[i + 1, j - 1] == strong) and (Image_Hysteresis[i - 1, j + 1] == strong))
                                or ((Image_Hysteresis[i + 1, j] == strong) and (Image_Hysteresis[i - 1, j] == strong))
                                or ((Image_Hysteresis[i + 1, j + 1] == strong) and (
                                        Image_Hysteresis[i - 1, j - 1] == strong))
                                or ((Image_Hysteresis[i, j + 1] == strong) and (Image_Hysteresis[i, j - 1] == strong))):
                            Image_Hysteresis[i, j] = strong
                    except IndexError as e:
                        pass
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (Image_Hysteresis[i, j] == strong) or (Image_Hysteresis[i, j] == weak):
                try:
                    if ((Image_Hysteresis[i + 1, j - 1] == 0)
                            and (Image_Hysteresis[i + 1, j] == 0)
                            and (Image_Hysteresis[i + 1, j + 1] == 0)
                            and (Image_Hysteresis[i, j - 1] == 0)
                            and (Image_Hysteresis[i, j + 1] == 0)
                            and (Image_Hysteresis[i - 1, j - 1] == 0)
                            and (Image_Hysteresis[i - 1, j] == 0)
                            and (Image_Hysteresis[i - 1, j + 1] == 0)):
                        Image_Hysteresis[i, j] = 0

                except IndexError as e:
                    pass
    return Image_Hysteresis


def detectCircles(img, threshold, region, radius=None):
    (M, N) = img.shape  # Get the maximum rows and columns
    if radius == None:
        R_max = np.max((M, N))
        R_min = 3
    else:  # Determine the maximum and minimum radiuses available in the image
        [R_max, R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max, M+2*R_max, N+2*R_max))
    B = np.zeros((R_max, M+2*R_max, N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0, 360)*np.pi/180
    edges = np.argwhere(img[:, :])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1), 2*(r+1)))
        (m, n) = (r+1, r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x, n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X=[x-m+R_max, x+m+R_max]                                           #Computing the extreme X values
            Y=[y-n+R_max, y+n+R_max]                                            #Computing the extreme Y values
            A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
        A[r][A[r] < threshold*constant/r] = 0

    for r, x, y in np.argwhere(A):
        temp = A[r-region:r+region, x-region:x+region, y-region:y+region]
        try:
            p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        B[r+(p-region), x+(a-region), y+(b-region)] = 1

    return B[:, R_max:-R_max, R_max:-R_max]

def houghLine(image):
    ''' Basic Hough line transform that builds the accumulator array
    Input : image : edge image (canny)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    '''
    #Get image dimensions
    # y for rows and x for columns
    Ny = image.shape[0]
    Nx = image.shape[1]

    #Max diatance is diagonal one
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90))
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    accumulator = np.zeros((2 * Maxdist, len(thetas)))
    for y in range(Ny):
        for x in range(Nx):
            # Check if it is an edge pixel
            #  NB: y -> rows , x -> columns
            if image[y,x] > 0:
                # Map edge pixel to hough space
                for k in range(len(thetas)):
                    # Calculate space parameter
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    # Update the accumulator
                    # N.B: r has value -max to max
                    # map r to its idx 0 : 2*max
                    accumulator[int(r) + Maxdist,k] += 1
    return accumulator, thetas, rs

def plotLines(image, rho, theta):
    width=image.shape[0]
    height=image.shape[1]
    fig = plt.figure()
    plt.imshow(image)
    x = np.linspace(0, width)
    cosine=np.cos(theta)
    sine=np.sin(theta)
    cotan=cosine/sine
    ratio=rho/sine
    for i in range(len(rho)):
        # if thete is not 0
        if (theta[i]):
            plt.plot(x, (-x * cotan[i]) + ratio[i])
        # if theta is 0
        else:
            # Draw a vertical line at the corresponding rho value
            plt.axvline(rho[i])
    plt.xlim(0, width)
    plt.ylim(height, 0)
    fig.savefig('Lines.png')
    plt.show()

def detectLines(image, accumulator, thetas, rhos, threshold):
    # Determine the number of lines in the image
    # By creating a threashold and determining it by multiplicating
    # The threshold level and the maximum value of the accumulator
    detectedLines = np.where(accumulator >= (threshold * accumulator.max()))
    rho = rhos[detectedLines[0]]  # Get the indices of the rohs corresponding to the detected lines
    theta = thetas[detectedLines[1]]  # Get the indices of the thetas corresponding to the detected lines
    plotLines(image, rho, theta)



global imageGRAY, imageHSV, imageRGB, Max, Min, imageSize, height, width, imageSource, Clicked, GaussianK, GaussianSTD

global imageGRAYH, imageRGBH, MaxH, MinH, imageSizeH, heightH, widthH, imageSourceH, ClickedH

global imageGRAYHM, imageRGBHM, MaxHM, MinHM, imageSizeHM, heightHM, widthHM, imageSourceHM, histogramHM, ClickedHM

#global imageGRAYF, imageRGBF, MaxF, MinF, imageSizeF, heightF, widthF, imageSourceF, ClickedF

# Clicked is used to make sure that an image is loaded before choosing a filter
Clicked = 0
ClickedH = 0
ClickedHM = 0
GaussianK = 1
GaussianSTD = 1

class CV(QMainWindow):
    def __init__(self):
        super(CV, self).__init__()
        loadUi('mainwindow.ui', self)
        self.pushButton_filters_load.clicked.connect(self.load_image)
        self.pushButton_histograms_load.clicked.connect(self.load_histogram)
        self.pushButton_Apply_Gaussian.clicked.connect(self.gaussian_kernel)
        self.pushButton_circles_load.clicked.connect(self.circle_detection)
        self.pushButton_histograms_load_target.clicked.connect(self.load_histogram_matching)
        self.pushButton_lines_load.clicked.connect(self.line_detection)
        self.comboBox.activated.connect(self.filter_selection)
        self.radioButton.clicked.connect(self.histogram_equalization)
        self.radioButton_2.clicked.connect(self.histogram_matching)

    def load_image(self):
        global imageGRAY, imageHSV, imageRGB, Max, Min, imageSize, height, width, imageSource, Clicked
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        imageSource = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        # To make sure the application doesn't crash if no image is loaded
        if imageSource:
            imageRGB = cv2.imread(imageSource)
            imageGRAY = color.rgb2gray(imageRGB)  # np.dot(image[..., :3], [0.299, 0.587, 0.114])
            imageHSV = color.rgb2hsv(imageRGB)
            Max = np.max(imageGRAY)
            Min = np.min(imageGRAY)
            imageSize = np.shape(imageGRAY)
            height = np.size(imageGRAY, 0)
            width = np.size(imageGRAY, 1)
            name = '(' + str(imageSize[0]) + 'X' + str(imageSize[1]) + ')'
            self.label_13.setText(OpenedFile(imageSource))
            self.label_14.setText(name)
            self.label_15.setText(OpenedFile(imageSource))
            self.label_16.setText(name)
            self.graphicsView_2.setImage(imageGRAY.T)
            self.graphicsView_4.setImage(imageGRAY.T)
            Clicked = 1

    def filter_selection(self):
        global imageGRAY, imageHSV, imageRGB, Max, Min, imageSize, height, width, imageSource, Clicked
        if Clicked == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Laplacian_Kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        selection = self.comboBox.currentText()
        if selection == "Prewitt Filter":
            X_Prewitt_Kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Y_Prewitt_Kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            X_Prewitt_Image = signal.convolve2d(imageGRAY, X_Prewitt_Kernel, boundary='symm', mode='same')
            Y_Prewitt_Image = signal.convolve2d(imageGRAY, Y_Prewitt_Kernel, boundary='symm', mode='same')
            Prewitt_Magnitude = np.sqrt(np.square(X_Prewitt_Image) + np.square(Y_Prewitt_Image))
            Prewitt_Direction = np.arctan(np.divide(Y_Prewitt_Image, X_Prewitt_Image))
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Prewitt_Magnitude.T)

        elif selection == "Sobel Filter":
            X_Sobel_Kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Y_Sobel_Kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            X_Sobel_Image = signal.convolve2d(imageGRAY, X_Sobel_Kernel, boundary='symm', mode='same')
            Y_Sobel_Image = signal.convolve2d(imageGRAY, Y_Sobel_Kernel, boundary='symm', mode='same')
            Sobel_Magnitude = np.sqrt(np.square(X_Sobel_Image) + np.square(Y_Sobel_Image))
            Sobel_Direction = np.arctan(np.divide(Y_Sobel_Image, X_Sobel_Image))
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Sobel_Magnitude.T)

        elif selection == "Laplacian Filter":
            Laplacian_Image = signal.convolve2d(imageGRAY, Laplacian_Kernel, boundary='symm', mode='same')
            Laplacian_Magnitude = np.abs(Laplacian_Image)  # ax_mag.imshow(np.absolute(grad), cmap='gray')
            Laplacian_Direction = np.angle(Laplacian_Image)  # ax_ang.imshow(np.angle(grad), cmap='hsv')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Laplacian_Image.T)

        elif selection == "Box Filter":
            Box_Kernel = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
            Box_Image = signal.convolve2d(imageGRAY, Box_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Box_Image.T)

        elif selection == "Gaussian Filter (3x3)":
            Gaussian_Kernel_3 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            Gaussian_Image_3 = signal.convolve2d(imageGRAY, Gaussian_Kernel_3, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Gaussian_Image_3.T)

        elif selection == "Gaussian Filter (5x5)":
            Gaussian_Kernel_5 = np.array([[1/273, 4/273, 7/273, 4/273, 1/273],
                                        [4/273, 16/273, 26/273, 16/273, 4/273],
                                        [7/273, 26/273, 41/273, 26/273, 7/273],
                                        [4/273, 16/273, 26/273, 16/273, 4/273],
                                        [1/273, 4/273, 7/273, 4/273, 1/273]])
            Gaussian_Image_5 = signal.convolve2d(imageGRAY, Gaussian_Kernel_5, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Gaussian_Image_5.T)

        elif selection == "Sharpening Filter":
            Sharpening_Kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            Sharpening_Image = signal.convolve2d(imageGRAY, Sharpening_Kernel, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(np.abs(Sharpening_Image).T)

        elif selection == "LoG":
            Gaussian_Kernel_3 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            Gaussian_Image_3 = signal.convolve2d(imageGRAY, Gaussian_Kernel_3, boundary='symm', mode='same')
            LoG_Image = signal.convolve2d(Gaussian_Image_3, Laplacian_Kernel, boundary='symm', mode='same')
            LoG_Magnitude = np.abs(LoG_Image)
            LoG_Direction = np.angle(LoG_Image)
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(LoG_Image.T)

        elif selection == "DoG":
            Gaussian_Kernel_3 = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
            Gaussian_Image_3 = signal.convolve2d(imageGRAY, Gaussian_Kernel_3, boundary='symm', mode='same')
            Gaussian_Kernel_5 = np.array([[1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273],
                                          [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                                          [7 / 273, 26 / 273, 41 / 273, 26 / 273, 7 / 273],
                                          [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                                          [1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273]])
            Gaussian_Image_5 = signal.convolve2d(imageGRAY, Gaussian_Kernel_5, boundary='symm', mode='same')
            DoG_Image = Gaussian_Image_3 - Gaussian_Image_5
            DoG_Magnitude = np.abs(DoG_Image)
            DoG_Direction = np.angle(DoG_Image)
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(DoG_Image.T)

        elif selection == "Median Filter":
            Median_Variable = [(0, 0)] * 9
            Median_Image = imageGRAY.copy()
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    Median_Variable[0] = Median_Image[i - 1, j - 1]
                    Median_Variable[1] = Median_Image[i - 1, j]
                    Median_Variable[2] = Median_Image[i - 1, j + 1]
                    Median_Variable[3] = Median_Image[i, j - 1]
                    Median_Variable[4] = Median_Image[i, j]
                    Median_Variable[5] = Median_Image[i, j + 1]
                    Median_Variable[6] = Median_Image[i + 1, j - 1]
                    Median_Variable[7] = Median_Image[i + 1, j]
                    Median_Variable[8] = Median_Image[i + 1, j + 1]
                    Median_Variable.sort()
                    Median_Image[i, j] = Median_Variable[4]
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Median_Image.T)

        elif selection == "High-pass Filter":
            Gray_image = imageGRAY.copy()
            (h, w) = imageSize
            # Resize the image to a minimum size 255*255
            if h <= 255 or w <= 255:
                Gray_image = resize(imageGRAY, (255, 255), anti_aliasing=True)
                (h, w) = Gray_image.shape
                print(h, w)
            half_h, half_w = int(h / 2), int(w / 2)
            F1 = fp.fft2((Gray_image).astype(float))
            F2 = fp.fftshift(F1)
            n = 10
            F2_High = F2.copy()
            # select all but the first 50x50 (low) frequencies
            F2_High[half_h - n:half_h + n + 1, half_w - n:half_w + n + 1] = 0
            HighPass_Image_Frequency = 20*np.log10(F2_High).astype(int)
            HighPass_Image = fp.ifft2(fp.ifftshift(F2_High)).real
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(HighPass_Image.T)

        elif selection == "Low-pass Filter":
            Gray_image = imageGRAY.copy()
            (h, w) = imageSize
            # Resize the image to a minimum size 255*255
            if h <= 255 or w <= 255:
                Gray_image = resize(imageGRAY, (255, 255), anti_aliasing=True)
                (h, w) = Gray_image.shape
                print(h, w)
            F1 = fp.fft2((Gray_image).astype(float))
            F2 = fp.fftshift(F1)
            half_h, half_w = int(h / 2), int(w / 2)
            n = 100  # Window Size
            F2_Low = F2.copy()
            F2_Low[0:n, 0:w] = 0
            F2_Low[h - n:h, 0:w] = 0
            F2_Low[0:h, 0:n] = 0
            F2_Low[0:h, w - n:w] = 0
            LowPass_Image_Frequency = 20 * np.log10(F2_Low).astype(int)
            LowPass_Image = fp.ifft2(fp.ifftshift(F2_Low)).real
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(LowPass_Image.T)

        elif selection == "Band-pass Filter":
            Gray_image = imageGRAY.copy()
            (h, w) = imageSize
            # Resize the image to a minimum size 255*255
            if h <= 255 or w <= 255:
                Gray_image = resize(imageGRAY, (255, 255), anti_aliasing=True)
                (h, w) = Gray_image.shape
                print(h, w)
            F1 = fp.fft2((Gray_image).astype(float))
            F2 = fp.fftshift(F1)
            half_h, half_w = int(h / 2), int(w / 2)
            n = 20  # Window Size
            F2_Band = F2.copy()
            F2_Band[half_h - n:half_h + n + 1, half_w - n:half_w + n + 1] = 0
            n = 100  # Window Size
            F2_Band[0:n, 0:w] = 0
            F2_Band[h - n:h, 0:w] = 0
            F2_Band[0:h, 0:n] = 0
            F2_Band[0:h, w - n:w] = 0
            BandPass_Image_Frequency = 20*np.log10(F2_Band).astype(int)
            BandPass_Image = fp.ifft2(fp.ifftshift(F2_Band)).real
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(BandPass_Image.T)
        elif selection == "FFT Magnitude":
            imageFFT = np.fft.fft2(imageGRAY)
            phaseFFT = np.angle(imageFFT)
            magnitudeFFT = np.absolute(imageFFT)
            magnitudeFFTShifted = np.fft.fftshift(np.log10(1 + magnitudeFFT))
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(magnitudeFFTShifted.T)

        else:
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(imageGRAY.T)

    def load_histogram(self):
        global imageGRAYH, imageRGBH, MaxH, MinH, imageSizeH, heightH, widthH, imageSourceH, ClickedH, histogram
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        imageSourceH = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        # To make sure the application doesn't crash if no image is loaded
        if imageSourceH:
            imageRGBH = cv2.imread(imageSourceH)
            imageGRAYH = color.rgb2gray(imageRGBH)  # np.dot(image[..., :3], [0.299, 0.587, 0.114])
            MaxH = np.max(imageGRAYH)
            MinH = np.min(imageGRAYH)
            imageSizeH = np.shape(imageGRAYH)
            heightH = np.size(imageGRAYH, 0)
            widthH = np.size(imageGRAYH, 1)
            name = '(' + str(imageSizeH[0]) + 'X' + str(imageSizeH[1]) + ')'
            histogram = np.zeros((2, 256), dtype=int)
            for i in range(256):
                histogram[0, i] = i
            # Linear Scaling & Histogram Calculating
            a = -1 * MinH
            b = 255 / (MaxH - MinH)
            for i in range(imageSizeH[0]):
                for j in range(imageSizeH[1]):
                    intensity = math.floor(b * (imageGRAYH[i, j] + a))
                    histogram[1, intensity] = histogram[1, intensity] + 1
            self.label_15.setText(OpenedFile(imageSourceH))
            self.label_16.setText(name)
            self.graphicsView_4.setImage(imageGRAYH.T)
            self.graphicsView_6.clear()
            self.graphicsView_6.plot(histogram[0, :], histogram[1, :])
            self.radioButton.setChecked(False)
            self.radioButton_2.setChecked(False)
            ClickedH = 1

    def histogram_equalization(self):
        global imageGRAYH, MaxH, MinH, imageSizeH, heightH, widthH, ClickedH
        if ClickedH == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Histogram,new_Gray_image = Histogram_Equalization (imageGRAYH)
        # Create the new equalized image
        for i in range(imageSizeH[0]):
            for j in range(imageSizeH[1]):
                intensity = int(new_Gray_image[i, j])
                new_Gray_image[i, j] = Histogram[3, intensity]
        self.graphicsView_5.setImage(new_Gray_image.T)
        self.graphicsView_7.clear()
        self.graphicsView_7.plot(Histogram[0, :], Histogram[3, :])

    def load_histogram_matching(self):
        global imageGRAYHM, imageRGBHM, MaxHM, MinHM, imageSizeHM, heightHM, widthHM, imageSourceHM, ClickedHM
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        imageSourceHM = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        # To make sure the application doesn't crash if no image is loaded
        if imageSourceHM:
            imageRGBHM = cv2.imread(imageSourceHM)
            imageGRAYHM = color.rgb2gray(imageRGBHM)  # np.dot(image[..., :3], [0.299, 0.587, 0.114])
            MaxHM = np.max(imageGRAYHM)
            MinHM = np.min(imageGRAYHM)
            imageSizeHM = np.shape(imageGRAYHM)
            heightHM = np.size(imageGRAYHM, 0)
            widthHM = np.size(imageGRAYHM, 1)
            histogramHM = np.zeros((2, 256), dtype=int)
            for i in range(256):
                histogramHM[0, i] = i
            # Linear Scaling & Histogram Calculating
            a = -1 * MinHM
            b = 255 / (MaxHM - MinHM)
            for i in range(imageSizeHM[0]):
                for j in range(imageSizeHM[1]):
                    intensity = math.floor(b * (imageGRAYHM[i, j] + a))
                    histogramHM[1, intensity] = histogramHM[1, intensity] + 1
            self.graphicsView_5.setImage(imageGRAYHM.T)
            self.graphicsView_7.clear()
            self.graphicsView_7.plot(histogramHM[0, :], histogramHM[1, :])
            ClickedHM = 1
            self.radioButton.setChecked(False)
            self.radioButton_2.setChecked(False)

    def histogram_matching(self):
        global imageGRAYH, imageRGBH, MaxH, MinH, imageSizeH, heightH, widthH, imageSourceH, ClickedH, \
            imageGRAYHM, imageRGBHM, MaxHM, MinHM, imageSizeHM, heightHM, widthHM, imageSourceHM, histogramHM, ClickedHM
        if ClickedH == 0 or ClickedHM == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        Histogram_Source,new_Gray_image_Source = Histogram_Equalization(imageGRAYHM)
        Histogram_Input, new_Gray_image_Input = Histogram_Equalization(imageGRAYH)

        size = np.shape(imageGRAYH)
        for i in range(size[0]):
            for j in range(size[1]):
                intensity = int(new_Gray_image_Input[i, j])
                new_Gray_image_Input[i, j] = Histogram_Source[3, intensity]
        self.graphicsView_5.setImage(new_Gray_image_Input.T)

    def gaussian_kernel(self):
        global imageGRAY, imageHSV, imageRGB, Max, Min, imageSize, height, width, imageSource, Clicked, \
            GaussianK, GaussianSTD
        if Clicked == 0:
            QMessageBox.about(self, "Error!", "Please choose an image")
            return
        if self.lineEdit.text().isdigit() and self.lineEdit_2.text().isdigit():
            GaussianK = int(self.lineEdit.text())
            GaussianSTD = int(self.lineEdit_2.text())
            Gaussian_Kernel_1 = signal.gaussian(GaussianK, std=GaussianSTD).reshape(GaussianK, 1)
            Gaussian_Kernel_2 = np.outer(Gaussian_Kernel_1, Gaussian_Kernel_1)
            Gaussian_Image = signal.convolve2d(imageGRAY, Gaussian_Kernel_2, boundary='symm', mode='same')
            self.graphicsView_3.clear()
            self.graphicsView_3.setImage(Gaussian_Image.T)
        else:
            QMessageBox.about(self, "Error!", "Enter valid numbers")
            return

    # References [10][11]
    def circle_detection(self):
        imagergb = cv2.imread('ball.PNG')
        image = color.rgb2gray(imagergb)
        imageCanny = Canny_Edge_Detection(image)
        R_max = 20
        R_min = 18
        result = detectCircles(imageCanny, 5, 15, radius=[R_max, R_min])
        fig = plt.figure()
        plt.imshow(imagergb)
        circleCoordinates = np.argwhere(result)  # Extracting the circle information
        circle = []
        for r, x, y in circleCoordinates:
            circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
        plt.show()
        fig.savefig('BallCircle.png')
        img = "BallCircle.png"
        name = '(' + str(image.shape[0]) + 'X' + str(image.shape[1]) + ')'
        self.label_21.setText('ball.PNG')
        self.label_20.setText(name)
        self.label_22.setPixmap(QPixmap(img).scaled(self.label_22.width(), self.label_22.height()))
        self.graphicsView_8.setImage(image.T)
        self.graphicsView_9.setImage(color.rgb2gray(cv2.imread('BallCircle.png')).T)

    def line_detection(self):
        """options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        imageLocation = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        # To make sure the application doesn't crash if no image is loaded
        if imageLocation:"""
        imageLocation = 'House.jpg'
        imageR = cv2.imread(imageLocation)
        imageG = color.rgb2gray(imageR)
        imageEDGE = Canny_Edge_Detection(imageG)
        accumulator, thetas, rhos = houghLine(imageEDGE)
        name = '(' + str(imageG.shape[0]) + 'X' + str(imageG.shape[1]) + ')'
        detectLines(imageR, accumulator, thetas, rhos, 0.5)
        self.label_24.setText(name)
        self.label_23.setText('House.jpg')
        self.graphicsView_11.setImage(accumulator.T)
        self.graphicsView_12.setImage(imageG.T)
        self.graphicsView_13.setImage(color.rgb2gray(cv2.imread('Lines.png')).T)
        self.label_25.setPixmap(QPixmap('Lines.png').scaled(self.label_25.width(), self.label_25.height()))




if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    app = QApplication(sys.argv)
    widget = CV()
    widget.show()
    sys.exit(app.exec_())
