#!/usr/bin/evn python
# Author : CHINMAYA kHAMESRA

#Importing the dependencies 
import os
import cv2
import numpy as np 
from copy import deepcopy

class Img:
    def __init__(self):
        self.grey = []
        self.rgb = []
        self.corners = []
        self.anms = []
        self.feature = []
        self.featurei = []
        self.keypoint = []
        self.offsetrow = 0 
        self.offsetcol = 0 

# Defining ANMS 
def anms(indexrange, features):
    # Evaluating the good features 
    for img in indexrange:
        img.anms = cv2.goodFeaturesToTrack(img.grey, features, 0.01, 10)
        img.anms = img.anms.astype('int32')
        rgb_anms = deepcopy(img.rgb)
        fineimage = img.anms

        for c in range(len(fineimage)):
            [col,row] = fineimage[c][0]
            updatedwidth = cv2.circle(rgb_anms, (col,row), 3, (0,0,255), 1)

        #Evaluate the new image 
        new = []
        for r in range(len(img.anms)):
            [col,row] = img.anms[r,0]
            if 0 < col < img.grey.shape[1] and 0 < row < img.grey.shape[0]:
                rowl, rowh = row-20, row+20
                coll, colh = col-20, col+20
                #Checking the boundaries
                if img.grey[rowl:rowh,col:colh].all() == 0 or img.grey[rowl:rowh,col-1:colh].all() == 0 or img.grey[rowl:rowh,col+1:colh].all() == 0:
                    pass
                elif img.grey[row:rowh,coll:colh].all() == 0 or img.grey[row-1:rowh,coll:colh].all() == 0 or img.grey[row+1:rowh,coll:colh].all() == 0:
                    pass
                elif img.grey[rowl:row,coll:colh].all() == 0 or img.grey[rowl:row-1,coll:colh].all() == 0 or img.grey[rowl:row+1,coll:colh].all() == 0:
                    pass
                elif img.grey[rowl:rowh,coll:col].all() == 0 or img.grey[rowl:rowh,coll:col-1].all() == 0 or img.grey[rowl:rowh,coll:col+1].all() == 0:
                    pass
                else:
                    new.append(img.anms[r])

        img.anms = np.array(new, dtype=int)
    return indexrange

def featuredescriptor(indexrange, iterr, patch = 25):
    offset = patch//2
    # Evaluating the descriptors 

    for r in range(len(indexrange)):
        feature1 = []
        featured1 = []
        img = indexrange[r]
        img_dim = img.grey.shape

        for m in range(len(img.anms)):			
            [col,row] = img.anms[m][0]

            colmin, colmax = col-offset, col+offset
            rowmin, rowmax = row-offset, row+offset

            if colmin>=0 and colmax<=img_dim[0]-1 and rowmin>=0 and rowmax<=img_dim[1]-1:
                feature = img.grey[rowmin:rowmax,colmin:colmax]
                feature = cv2.GaussianBlur(feature,(5,5),cv2.BORDER_DEFAULT)

                if m < 20 and save:
                    cv2.imwrite(f'{out}/feature_4141_{iterr}.jpg', feature)
                feature = cv2.resize(feature,(8,8))

                if m < 20 and save:
                    cv2.imwrite(f'{out}/feature_88_{iterr}.jpg', feature)
                    iterr+=1

                feature = np.mean(np.reshape(feature, (64,1)), axis = 1)
                
                mean = np.mean(feature,axis=0)
                std = np.std(feature, axis = 0)

                feature = (feature - mean)/std

                # Appending the features 
                feature1.append(feature)
                featured1.append([col,row])

        img.feature = feature1
        img.featurei = featured1

        #Evaluating the key points  
        img.keypoint = keypoint(img.featurei)
    return indexrange

def keypoint(featurei):
    arr = []
    for i in range(len(featurei)):
        arr.append(cv2.KeyPoint(float(featurei[i][0]),float(featurei[i][1]),1.0))
    return arr

def featurematching(image1, image2, threshold, iterr):

    indexfp = []
    feature1_out = []
    feature2_out = []

    features1, feature1index = [], []
    features2, feature2index = [], []

    # Matching of the features 
    if len(image1.feature) < len(image2.feature):
        features2 = image2.feature[:len(image1.feature)][:]
        feature2index = image2.featurei[:len(image1.featurei)][:]
        features1 = image1.feature
        feature1index = image1.featurei

    if len(image1.feature) > len(image2.feature):
        features1 = image1.feature[:len(image2.feature)][:]
        feature1index = image1.featurei[:len(image2.featurei)][:]
        features2 = image2.feature
        feature2index = image2.featurei

    keypoint1 ,keypoint2 = image1.keypoint[:len(features1)] , image2.keypoint[:len(features2)]


    bf = cv2.BFMatcher()

    matches = bf.knnMatch(np.asarray(features1,np.float32),np.asarray(features2,np.float32),2)
    good = []
    original_threshold = threshold

    while len(good)<70:
        for i,j in matches: 

            if i.distance/j.distance <= threshold: 
                
                good.append([i])
                feature1_col, feature1_row= feature1index[i.queryIdx]
                feature2_col, feature2_row = feature2index[i.trainIdx]

                indexfp.append([feature1_col, feature1_row, feature2_col, feature2_row])
                feature1_out.append([feature1_col,feature1_row])
                feature2_out.append([feature2_col,feature2_row])
        threshold += 0.1
    threshold = original_threshold

    # KNN to draw the matches 
    sameimage = cv2.drawMatchesKnn(image1.rgb, keypoint1, image2.rgb, keypoint2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("feature_matching", sameimage)
    return feature1_out, feature2_out, good

# RANSAC
def RANSAC(feature1, feature2, good, inliertol, iter=100):

    value = []
    index = []
    good_new = []

    tolerance  = inliertol
    i = 0

    while len(index)<10:
        for _ in range(iter):

            # Evaluating the pairs 
            pair = np.random.randint(0,len(feature1),size=4)


            final = np.float32([feature1[p] for p in pair])
            initial = np.float32([feature2[p] for p in pair])
            approximate = cv2.getPerspectiveTransform(initial, final)

            for p in pair:
                h = np.dot(approximate, np.array([feature1[p][0], feature1[p][1], 1]))
                
                if h[2] != 0:
                    h_x = h[0]/h[2]
                    h_y = h[1]/h[2]
                else:
                    h_x = h[0]/0.000001
                    h_y = h[1]/0.000001
                h = np.array([h_x, h_y])
                
                # Normalize
                norm = np.linalg.norm(feature2[p]-h)
                value.append(norm)

                if norm > 1.0 and norm < inliertol:
                    if feature1[p] not in index and feature2[p] not in index:
                        index.append(feature1[p])
                        index.append(feature2[p])
                        good_new.append(good[p])

         
        # Updating the tolerance  
        difference = abs(inliertol-min(value))+30.0
        inliertol += difference

    inliertol = tolerance 
    final = np.float32([index[i] for i in range(0,len(index),2)])
    initial = np.float32([index[i]for i in range(1,len(index),2)])
    new_homo, _ = cv2.findHomography(initial, final, method=0, ransacReprojThreshold=0.0)
    return np.asarray(index), good_new, new_homo

def wrapimage(image1, image2, H, offsetrow, offsetcol):

    #Dimensions of the images 
    height1,width1 = image1.shape[:2]
    height2,width2 = image2.shape[:2]

    #Points  
    point1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
    point2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
    update = cv2.perspectiveTransform(point2, H)

    # Adding both the point
    point = np.concatenate((point1, update), axis=0)

    [xmin, ymin] = np.int32(point.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(point.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 

    result = cv2.warpPerspective(image2, Ht.dot(H), (xmax-xmin+offsetrow, ymax-ymin+offsetcol))

    rowl, coll = max(t[1]-offsetrow,0) , max(t[0]-offsetcol, 0)
    
    rowh = height1+rowl
    colh = width1+t[0]
    
    if coll-colh != image1.shape[1]:
        coll = t[0]
        colh = width1+t[0]
    if rowl-rowh != image1.shape[0]:
        rowl = t[1]
        rowh = height1+t[1]
    
    indexr = 0
    for row in range(rowl, rowh):
        indexc = 0
        for col in range(coll, colh):
            if image1[indexr,indexc,:].all() != 0:
                result[row,col,:] = image1[indexr,indexc,:]
            indexc += 1
        indexr += 1
    
    offset_y = t[0]-offsetcol
    offset_x = t[1]-offsetrow
    return result, offset_x, offset_y

def stitch(image1, image2, iterr):

    feature1, feature2, good = featurematching(image1, image2, threshold=feature_threshold, iterr=iterr)
    print(np.shape(feature1))
    # RANSAC image
    ransaindexcdx, good_new, homography = RANSAC(feature1, feature2, good, RANSAC_threshold)
    print('RANSAC:',np.shape(ransaindexcdx))
    # Combined image 
    sameimage = cv2.drawMatchesKnn(image1.rgb, image1.keypoint, image2.rgb, image2.keypoint, good_new, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("ransac", sameimage)
    # Wrapped image 
    wrapedimage, offsetrow, offsetcol = wrapimage(image1.rgb, image2.rgb, homography, image1.offsetrow, image1.offsetcol)
    cv2.imshow("wrapedimage", wrapedimage)

    attachimage = Img()
    attachimage.offsetrow += offsetrow
    attachimage.offsetcol += offsetcol
    attachimage.rgb = wrapedimage
    attachimage.grey = cv2.cvtColor(wrapedimage, cv2.COLOR_BGR2GRAY)

    for img in [attachimage]:
        img.corners = cv2.cornerHarris(img.grey, 2, 3, 0.04)

    print("stiched corners:",np.shape(attachimage.corners))
    [attachimage] = anms([attachimage], NumFeatures)
    attachimage.anms = attachimage.anms[10:]
    
    rgb_anms = deepcopy(attachimage.rgb)
    fineimage = attachimage.anms
    for m in range(len(fineimage)):
        [c,r] = fineimage[m][0]
        updatedwidth = cv2.circle(rgb_anms, (c,r), 3, (0,0,255), 1)

    cv2.imshow("anms", updatedwidth)
    [attachimage] = featuredescriptor([attachimage], iterr)
    return attachimage

#Running the main function 
def main():
    
    global out, feature_threshold, RANSAC_threshold, NumFeatures, save, iterr
    NumFeatures, iterr, feature_threshold, RANSAC_threshold= 100, 0, 0.80, 100.0
    save = False 
    in_path="/home/chinmay/Downloads/YourDirectoryID_p1/Phase1/Data/Train/Set1"
    out = '../Data/Outputs'

    if not os.path.isdir(out):
        os.makedirs(out)
    out = f'{out}/{in_path.split("/")[-1]}'
    if not os.path.isdir(out):
        os.makedirs(out)

    # Read the directories 
    img_files = os.listdir(in_path)
    img_files.sort()
    indexrange = []
    for f in img_files:
        ## Read image and append
        img = Img()
        img.rgb = cv2.imread(f'{in_path}/{f}')
        img.grey = cv2.cvtColor(img.rgb, cv2.COLOR_BGR2GRAY)
        img.grey = cv2.GaussianBlur(img.grey,(3,3),cv2.BORDER_DEFAULT)
        indexrange.append(img)

    start = len(indexrange) 

    # Harris corner detector 
    for img in indexrange:
        img.corners = cv2.cornerHarris(img.grey, 2, 3, 0.04)

    print("Shape of output corner:",np.shape(indexrange[0].corners),np.shape(indexrange[1].corners),np.shape(indexrange[2].corners))

    for m in range(len(indexrange)):
        img = indexrange[m]
        rgb_corners = deepcopy(img.rgb)
        array_corner = img.corners
        array_corner[array_corner<0] = 0
        mean = np.mean(array_corner)
        for r in range(len(array_corner)):
            for c in range(len(array_corner[0])):
                if array_corner[r][c] > mean:
                    img_new = cv2.circle(rgb_corners, (c,r), 1, (0,0,255), 1)

    indexrange = anms(indexrange, NumFeatures)

    for i in range(len(indexrange)):
        img = indexrange[i]
        rgb_anms = deepcopy(img.rgb)
        fineimage = img.anms
        for m in range(len(fineimage)):
            [c,r] = fineimage[m][0]
            updatedwidth = cv2.circle(rgb_anms, (c,r), 3, (0,0,255), 1)

    indexrange = featuredescriptor(indexrange, iterr)

    anchoindexrmg = start//2
    iterr = 0

    for i in range(anchoindexrmg,0,-1):
        if i == anchoindexrmg:
            image1 = indexrange[i] 
        image2 = indexrange[i-1] 
        attachimage = stitch(image1, image2, iterr)
        iterr+=1
        image1 = attachimage
    
    image1 = attachimage 
    for i in range(anchoindexrmg, anchoindexrmg+1):
        image2 = indexrange[i+1] 
        attachimage = stitch(image1, image2, iterr)
        iterr += 1
        image1 = attachimage
    cv2.imshow("final",attachimage.rgb)


if __name__ == '__main__':
    main()

