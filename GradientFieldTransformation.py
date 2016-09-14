from cv2 import *
import numpy as np
from scipy import special

# Matches sigmaL and sigmaH according to what was learned
# Input:
# sigmaL: LR sharpness (LR bicubic interpolated)
# Output:
# sigmaH: HR sharpness
def sigmaL2sigmaH(sigmaL):
  sigmaH = sigmaL
  for i in range(sigmaH.shape[0]):
    for j in range(sigmaH.shape[1]):
      if sigmaH[i][j] > 0.5 and sigmaH[i][j] <= 1.5:
        sigmaH[i][j] = 0.325 + 1.58 * (sigmaL[i][j] - 1)
  for i in range(sigmaH.shape[0]):
    for j in range(sigmaH.shape[1]):
      if sigmaH[i][j] > 1.5 and sigmaH[i][j] <= 2.0:
        sigmaH[i][j] = 1.115 + 0.77 * (sigmaL[i][j] - 1.5)
  for i in range(sigmaH.shape[0]):
    for j in range(sigmaH.shape[1]):
      if sigmaH[i][j] > 2.0 and sigmaH[i][j] <= 2.5:
        sigmaH[i][j] = 1.5 + 1 * (sigmaL[i][j] - 2)
  for i in range(sigmaH.shape[0]):
    for j in range(sigmaH.shape[1]):
      if sigmaH[i][j] > 2.5 and sigmaH[i][j] <= 3.0:
        sigmaH[i][j] = 2 + 1.04 * (sigmaL[i][j] - 2.5)
  for i in range(sigmaH.shape[0]):
    for j in range(sigmaH.shape[1]):
      if sigmaH[i][j] > 3.0 and sigmaH[i][j] <= 3.5:
        sigmaH[i][j] = 2.52 + 1.08 * (sigmaL[i][j] - 3)
  for i in range(sigmaH.shape[0]):
    for j in range(sigmaH.shape[1]):
      if sigmaH[i][j] < 3.5:
        sigmaH[i][j] = 3.06 + 1.17 * (sigmaL[i][j] - 3.5)
  return sigmaH

# Transforms the LR image gradient field into the expected HR gradient field
# 
# This function defines the gradient profiles from the LR image, associates all the border pixels with its corresponding sharpness and makes the final gradient fields transformation
#
# Input:
#   maxProfile: max gradient profile lenght for each side
#
# Output:
#   HRGradComplex: expected HR gradient field in the complex number format

def GradientFieldTransformation(LR, maxProfile):
  # Variables
  lambdaValue = 1.6 # Value for lambdaL and lambdaH
  
  # Estimates the HR image using bicubic interpolation
  LREstimated = resize(LR, None, fx = 2, fy = 2, interpolation = INTER_CUBIC)
  
  # Detecting the borders using Sobel method
  gradX = Sobel(LREstimated, CV_16S, 1, 0, ksize = 5)
  absGradX = convertScaleAbs(gradX)
  gradY = Sobel(LREstimated, CV_16S, 0, 1, ksize = 5)
  absGradY = convertScaleAbs(gradY)
  borderPixels = addWeighted(absGradX, 0.5, absGradY, 0.5, 0)

  # Converting to CV_32F
  gradX = np.float32(gradX)
  gradY = np.float32(gradY)

  # Gradient Magnitude and Direction
  gradMagnitude = magnitude(gradX, gradY)
  gradDirection = phase(gradX, gradY, True)

  sharpnessMap = np.zeros((LREstimated.shape[0], LREstimated.shape[1]))

  # Calculating the gradient profile for each border pixel and its sharpness
  for row in range(borderPixels.shape[0]):
    for col in range(borderPixels.shape[1]):
      # Checks if the pixel is a border one
      if borderPixels[row][col] == 1:
        # Profile: col 0: gradient intensity, col 1: distance to border pixel
        profile = np.zeros((2 * maxProfile + 1, 2))
        j = maxProfile
        for i in range(maxProfile):
          profile[i][1] = j
          j = j - 1
        for i in range(maxProfile + 1, 2 * maxProfile + 1):
          profile[i][1] = j
          j = j + 1
       
        # Insert pixel into the border profile
        profile[maxProfile + 1, 0] = gradMagnitude[row, col]
        
        # Finds the profile in the positive gradient direction
        profileRow = row
        profileCol = col
        distance = 0
        while distance < maxProfile:
            direction = gradDirection[profileRow][profileCol]
            gradientMag = gradMagnitude[profileRow][profileCol]
            # Checks the position of the next profile pixel
            if -175 < direction and direction <= -45:
                profileRow = profileRow + 1
            elif -45 < direction and direction <= 45:
                profileCol = profileCol + 1
            elif 45 < direction and direction <= 175:
                profileRow = profileRow - 1
            else:
                profileCol = profileCol - 1
            
            # Checks if the next profile pixel is inside the image, if it is not, then it's the end of the profile in this direction
            if profileRow >= LREstimated.shape[0] or profileRow < 0:
                break
            if profileCol >= LREstimated.shape[1] or profileCol < 0:
                break
            
            # Checks if the gradient modulus is lower, if it is, updates the value, otherwise finishes the profile in this direction
            if gradientMag > gradMagnitude[profileRow][profileCol]:
                profile[maxProfile + 1 + distance][0] = gradMagnitude[profileRow][profileCol]
                distance = distance + 1
            else:
                break
        # Finds the profile in the negative gradient direction
        profileRow = row
        profileCol = col
        distance = 0
        while distance < maxProfile:
            direction = -gradDirection[profileRow][profileCol]
            gradientMag = gradMagnitude[profileRow][profileCol]
            
            # Checks the next profile pixel position
            if -175 < direction and direction <= -45:
              profileRow = profileRow + 1
            elif -45 < direction and direction <= 45:
              profileCol = profileCol + 1
            elif 45 < direction and direction <= 175:
              profileRow = profileRow - 1
            else:
              profileCol = profileCol - 1
            
            # Checks if the next profile pixel is inside the image, if it is not, then it's the end of the profile in this direction
            if profileRow >= LREstimated.shape[0] or profileRow < 0:
              break
            if profileCol >= LREstimated.shape[1] or profileCol < 0:
              break
            
            # Checks if the gradient modulus is lower, if it is, updates the value, otherwise finishes the profile in this direction
            if gradientMag > gradMagnitude[profileRow][profileCol]:
                profile[maxProfile - distance + 1][0] = gradMagnitude[profileRow][profileCol]
                distance = distance + 1
            else:
                break
        
        # Calculating sharpness
        sharpness = np.sqrt(np.sum(np.multiply(profile[:][0], profile[:][1] ** 2)) / np.sum(profile[:][1]))
        sharpnessMap[row][col] = sharpness

  # Locates the border pixel related to each pixel
  #
  # Initialize the ratioAux matrix which helps in obtaining the ratio matrix, which transforms the LREstimated's gradient field into the HR gradient field
  #
  #
  ratioAux = np.zeros((LREstimated.shape[0], LREstimated.shape[1], 2))
  ratioAux[:][:][1] = np.inf

  for row in range(LREstimated.shape[0]):
    for col in range(LREstimated.shape[1]):
      # Checks if the pixel is a border one
      if borderPixels[row][col] == 1:
        ratioAux[row][col][0] = sharpnessMap[row][col]
        ratioAux[row][col][1] = 0
      else:
        # Looks for the border pixel in the positive gradient direction
        profileRow = row
        profileCol = col
        posDistance = 0
        while posDistance < maxProfile:
          direction = gradDirection[profileRow][profileCol]
          gradientMag = gradMagnitude[profileRow][profileCol]
          
          # Checks the position of the next profile pixel
          if -175 < direction and direction <= -45:
            profileRow = profileRow + 1
          elif -45 < direction and direction <= 45:
            profileCol = profileCol + 1
          elif 45 < direction and direction <= 175:
            profileRow = profileRow - 1
          else:
            profileCol = profileCol - 1
          
          # Checks if the next profile pixel is inside the image, if it is not, then it's the end of the profile in this direction
          if profileRow >= LREstimated.shape[0] or profileRow < 0:
            break
          if profileCol >= LREstimated.shape[1] or profileCol < 0:
            break
          
          # Checks if the gradient modulus is higher, if it is, updates the value, otherwise finishes the profile in this direction
          if gradientMag < gradMagnitude[profileRow][profileCol]:
            # Checks if the pixel is a border one
            if borderPixels[profileRow][profileCol] == 1:
              ratioAux[row][col][0] = sharpnessMap[profileRow][profileCol]
              ratioAux[row][col][1] = posDistance
              break
            else:
              posDistance = posDistance + 1
          else:
            break
        
        # Looks for the border pixel in the negative gradient direction
        profileRow = row
        profileCol = col
        negDistance = 0
        while negDistance < maxProfile:
          direction = -gradDirection[profileRow][profileCol]
          gradientMag = gradMagnitude[profileRow][profileCol]
          
          # Checks the position of the next profile pixel
          if -175 < direction and direction <= -45:
            profileRow = profileRow + 1
          elif -45 < direction and direction <= 45:
            profileCol = profileCol + 1
          elif 45 < direction and direction <= 175:
            profileRow = profileRow - 1
          else:
            profileCol = profileCol - 1
          
          # Checks if the next profile pixel is inside the image, if it is not, then it's the end of the profile in this direction
          if profileRow >= LREstimated.shape[0] or profileRow < 0:
            break
          if profileCol >= LREstimated.shape[1] or profileCol < 0:
            break
          
          # Checks if the gradient modulus is higher, if it is, updates the value, otherwise finishes the profile in this direction
          if gradientMag < gradMagnitude[profileRow][profileCol]:
            # Checks if the pixel is a border one
            if borderPixels[profileRow][profileCol] == 1:
              # Checks which border pixel is closer
              if negDistance < posDistance:
                ratioAux[row][coluna][0] = sharpnessMap[profileRow][profileCol]
                ratioAux[row][coluna][1] = negDistance
              break
            else:
              negDistance = negDistance + 1
          else:
            break

  # Calculating the ratio matrix, which transforms the LREstimated gradient field into the HR gradient field
  sigmaL = np.zeros((ratioAux.shape[0], ratioAux.shape[1]))
  dist = np.zeros((ratioAux.shape[0], ratioAux.shape[1]))

  for i in range(sigmaL.shape[0]):
    for j in range(sigmaL.shape[1]):
      sigmaL[i][j] = ratioAux[i][j][0]

  for i in range(dist.shape[0]):
    for j in range(dist.shape[1]):
      dist[i][j] = ratioAux[i][j][1]
  
  sigmaH = sigmaL2sigmaH(sigmaL)

  alfaLambda = np.sqrt(special.gamma(3 / lambdaValue) / special.gamma(1 / lambdaValue))

  # Where the distance is inf, ratio = 1
  ratio = np.ones(dist.shape)
  # Calculates the ratio only where dist != inf and sigmaL != 0
  for i in range(dist.shape[0]):
    for j in range(dist.shape[1]):
      if dist[i][j] != np.inf and sigmaL[i][j] != 0:
        ratio[i][j] = sigmaL[i][j] / sigmaH[i][j] * np.exp(-(alfaLambda * abs(dist[i][j]) / sigmaH[i][j]) ** lambdaValue + (alfaLambda * abs(dist[i][j]) / sigmaL[i][j]) ** lambdaValue)

  # Transforms the LREstimated gradient field to complex numbers format

  LRGradComplex = gradMagnitude
  for i in range(gradMagnitude.shape[0]):
    for j in range(gradMagnitude.shape[1]):      
      LRGradComplex[i][j] = gradMagnitude[i][j] * np.exp(1j * gradDirection[i][j] / 180 * np.pi)

  # Transforms the gradient field using the found ratio
  
  HRGradComplex = ratio ** LRGradComplex

  return HRGradComplex
