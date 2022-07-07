import cv2 
import os
import numpy as np
from skimage.metrics import structural_similarity

def exportMask(folder, file, mask):
    path = currentDir + '/' + folder + '_mask'
    file = file.replace('.jpg','.png')
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path,file), mask)
    
def exportNightMask(subdir, nightFiles, latestDayTimeMask):
     for file in range(len(nightFiles)):
         lastElement = len(nightFiles) - 1
         exportMask(subdir,nightFiles[lastElement],latestDayTimeMask)
         nightFiles.pop()
              
def findContours(thresholdImg, nrow, ncol):
    mask = np.zeros((nrow, ncol), dtype=np.uint8)
    area = nrow * ncol
    halfA = 0.4 * area
    halfY = ncol / 2
    
    contours, hier = cv2.findContours(thresholdImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    for cnt in contours:
        match = cv2.matchShapes(cnt, c, 1, 0.0)
        (x,y,w,h) = cv2.boundingRect(cnt)
        bottomVertex = y + h
        if y == 0:              #Ensure it begins from the topmost part of the image
            if match == 0.0:    #Biggest contour
                if cv2.contourArea(cnt) >= halfA and x == 0:             #Ensure that it occupies most of the image and starts from top left OR
                    cv2.drawContours(mask,[cnt],-1,(255),cv2.FILLED)
                elif bottomVertex <= halfY:                             #Ensure that it sits in the upper part of the image
                    cv2.drawContours(mask,[cnt],-1,(255),cv2.FILLED)
            elif cv2.contourArea(cnt) > 1000 and bottomVertex <= halfY:  #Filters out smaller contours that are less likely to be sky
                cv2.drawContours(mask,[cnt],-1,(255),cv2.FILLED)
    return mask
    
def getGroundTruth(batchName, rRow, rCol):
    for root, subdirs, files in os.walk(currentDir):
        for subdir in subdirs:
            if subdir == "GroundTruth":
                subdirPath = os.path.join(root,subdir)
                subdirPath = '/'.join(subdirPath.split('\\')) 
                for file in os.listdir(subdir):
                    if file.startswith(batchName):
                        groundPath =  cv2.imread(subdirPath + '/' + file)
                        groundTruth = cv2.resize(groundPath, (rRow, rCol))
                        return groundTruth
                    

if __name__ == '__main__':               
    currentDir = '/'.join(os.getcwd().split('\\'))     
    rRow, rCol = 640, 480
    for root, subdirs, files in os.walk(currentDir):
        for subdir in subdirs:
            if "GroundTruth" not in subdir and "mask" not in subdir:
                subdirPath = os.path.join(root,subdir)
                subdirPath = '/'.join(subdirPath.split('\\')) 
                print(f'Processing {subdir}...')
                latestDayTimeMask = 0
                dayMaskAvailable = False
                nightFiles = []
                for file in os.listdir(subdir):
                    if file.endswith('.jpg'):
                        img = cv2.imread(subdirPath + '/' + file)   
                        # =============================================================================
                        # Image Preprocessing - Resizing & removing noise
                        # =============================================================================
                        img_resized = cv2.resize(img, (rRow,rCol))
                        
                        blue_channel = img_resized[:,:,0] 
                        blue_gaussian = cv2.GaussianBlur(blue_channel,(3,3),0)
                        
                        val, th = cv2.threshold(blue_gaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        [nrow, ncol] = blue_channel.shape 
                        # =============================================================================
                        # Find general contours
                        # =============================================================================
                        mask = findContours(th, nrow, ncol)
                                    
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))    
                        kernelEllipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
                        sE = np.array([[0, -1, 0],
                                        [-1, 5,-1],
                                        [0, -1, 0]])
                        
                        total = np.sum(mask)
                        if total == 0: #Night images
                            #If first image in the folder is night time, use the first subsequent day time mask as its mask
                            if not dayMaskAvailable:    
                                nightFiles.append(file)
                            else:
                                mask = latestDayTimeMask
                                exportMask(subdir,file,mask)
                        else:
                            #Get edges from sharpened image for better results
                            blue_sharpened = cv2.filter2D(blue_channel, -1, sE)
                            ext_sky_sharpened = cv2.bitwise_and(blue_sharpened, mask)
                            ext_sky_blur = cv2.bitwise_and(blue_gaussian, mask)
                            edges = cv2.Canny(ext_sky_sharpened, (val/2), 255)
                            edgesBlur = cv2.Canny(ext_sky_blur, (val/2), 255)
                            
                            # Compute difference between the two extracted edges - can determine if its clouds or actual objects
                            (score, diff) = structural_similarity(edges, edgesBlur, full=True)
                            similarity = float(f'{score*100:.2f}')
                            
                            if similarity <= 97:
                                closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
                                mask1 = np.zeros((nrow, ncol), dtype=np.uint8)
                                # =============================================================================
                                # Find contours in extracted sky region to cover objects which have been left out
                                # =============================================================================
                                skyContour, hier1 = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for c in skyContour:
                                    cv2.drawContours(mask1, [c],-1,(255),cv2.FILLED)
                                    
                                mask1 = cv2.bitwise_not(mask1)
                                added = cv2.bitwise_not(cv2.bitwise_and(mask, mask1)) #Inverse the image for opening
                                
                                #Using ellipse kernel for better smoothing
                                maskOpen = cv2.morphologyEx(added, cv2.MORPH_OPEN, kernelEllipse, iterations=1) 
                                maskOpen = cv2.bitwise_not(maskOpen)
                                latestDayTimeMask = maskOpen
                                dayMaskAvailable = True
                                exportMask(subdir,file,maskOpen)
                                
                                if dayMaskAvailable:
                                    exportNightMask(subdir, nightFiles, latestDayTimeMask)
                            else:
                                maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelEllipse, iterations=2) 
                                latestDayTimeMask = maskOpen
                                dayMaskAvailable = True
                                exportMask(subdir,file,maskOpen)
                                if dayMaskAvailable:
                                    exportNightMask(subdir, nightFiles, latestDayTimeMask)
                
    # =============================================================================
    # Evaluating masks     
    # =============================================================================
    for root, subdirs, files in os.walk(currentDir):
        for subdir in subdirs:                
            if "mask" in subdir:
                folder = subdir.split('_')
                batchName = folder[0]
                totalIOU = 0     
                totalFalsePos = 0
                groundTruth = getGroundTruth(batchName, rRow, rCol)
                groundTruth_area = np.count_nonzero(groundTruth)
                subdirPath = os.path.join(root,subdir)
                subdirPath = '/'.join(subdirPath.split('\\')) 
                subdirLength = len(os.listdir(subdirPath))
                print(f'Processing {subdir}...')
                if groundTruth_area == 0: 
                    for file in os.listdir(subdir):
                        mask = cv2.imread(subdirPath + '/' + file)  
                        mask_area = np.count_nonzero(mask)
                        area = rRow * rCol
                        falsePositives = (mask_area / area) 
                        totalFalsePos += falsePositives
                    avgFalsePos = totalFalsePos / subdirLength
                    print("No sky region in this set. Calculating false positives: ")
                    print(f'{subdir} false positives = {avgFalsePos*100:.2f}%')
                else:
                    for file in os.listdir(subdir):
                        mask = cv2.imread(subdirPath + '/' + file)  
                        mask_area = np.count_nonzero(mask)
                        intersection = np.count_nonzero( np.logical_and( mask, groundTruth) )
                        iou = intersection/(mask_area+groundTruth_area-intersection)
                        totalIOU += iou 
                    accuracy = totalIOU / subdirLength
                    print(f'{subdir} accuracy = {accuracy*100:.2f}%')
                


