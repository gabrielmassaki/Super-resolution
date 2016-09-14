from cv2 import *
import numpy as np
import sys
import GradientFieldTransformation as GFT

# Image to 8-bits unsigned int
def im2uint8(image):
    minVal = np.min(image.ravel())
    maxVal = np.max(image.ravel())
    out = image.astype('int') * (255 / (maxVal - minVal))
    return out

# Image to double
def im2double(image):
    minVal = np.min(image.ravel())
    maxVal = np.max(image.ravel())
    out = (image.astype('float') - minVal) / (maxVal - minVal)
    return out

# Downsampling
def downsampling(image):
  return pyrDown(image)

# Upsampling
def upsampling(image):
  return pyrUp(image)

# Creates a LR image by applying a blur then downsampling
def createLR(image, kernel_size):
  image = blur(image, (kernel_size, kernel_size))
  return downsampling(image)

def main():
  # Variables
  isRGB = True
  imageName = 'original.jpg'

  # Stop conditions
  tolRMSE = 10 ^ (-5) # max RMSE between the previous HR and the current HR
  maxIteration = 50 # max number of iterations

  # Filtro para suavizar a imagem antes de calcular o gradiente.
  nfilter = 5 # size of the gaussian nfilter x nfilter
  sigma = nfilter / 2

  maxProfile = 20 # max size of the gradient profile

  beta = 0.05 # gradient factor
  tal = 0.2 # weight of the derivative in gradient descent algorithm

  # Loads the image
  LR = imread(imageName)

  # Converts RGB to gray scale
  if isRGB:
    LR = cvtColor(LR, COLOR_BGR2GRAY)
#  LR = im2double(LR)

  # Gradient field transformation
  expectedComplexGrad = GFT.GradientFieldTransformation(LR, maxProfile)

  # Estimates the HR image using bicubic interpolation
  initialHR = resize(LR, None, fx = 2, fy = 2, interpolation = INTER_CUBIC)

  HR = initialHR
  counter = 0
  rmse = tolRMSE + 1
  
  while counter < maxIteration:
    currentHR = HR
    
    # Calculating the current HR gradient field
    gradX = Sobel(currentHR, CV_16S, 1, 0, ksize = 5)
    gradY = Sobel(currentHR, CV_16S, 0, 1, ksize = 5)

    # Converting to CV_32F
    gradX = np.float32(gradX)
    gradY = np.float32(gradY)

    curGradMag = magnitude(gradX, gradY)
    curGradDir = phase(gradX, gradY, True)

    curComplexGrad = curGradMag * np.exp(1j * curGradDir / 180 * np.pi)
    
    # Calculating the special factor

    # Generates the LR image based on the current HR image and calculates the difference between the LR images
    specialFactor = createLR(currentHR, 5)    
    
    # Applies the upsample to get a HR image and allow the comparission between gradients.
    # Then applies the blur filter (idk why)
    specialFactor = blur(upsampling(specialFactor), (5, 5))
    
    # Calculating the gradient factor
    gradientFactor = abs(curComplexGrad) ** 2 - abs(expectedComplexGrad) ** 2
    
    # Normalizing the gradient factor to -1 <= x <= 1
    gradientFactor[gradientFactor > 1] = 1
    gradientFactor[gradientFactor < -1] = -1

    # Updates the HR value
    HR = currentHR - tal * (specialFactor - beta * gradientFactor)
    
    # Calculates the mean square error between the previous HR image and the current one, i.e., checks if the estimate changed
    # For some reason the value has to converge to uint8. If not, the rmse increases intead of decreasing and algorithm doesnt converge    
    rmse = im2uint8(currentHR) - im2uint8(HR);
    rmse = sqrt(sum(sum(rmse ** 2)) / rmse.size)
    
    counter = counter + 1

  # Saving the results
  imwrite('result.png', HR)  

  # Showing the results
  namedWindow("Display window", WINDOW_AUTOSIZE)
  imshow("Display window", LR)
  waitKey(0)
  imshow( "Display window", HR)
  waitKey(0)

if __name__ == "__main__":
  main()
