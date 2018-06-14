#-*- coding: UTF-8 -*-
import cv2
import numpy as np
def func1():
    img_data=cv2.imread('e.jpg')
    gray=cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
    face=cv2.imread('xiaogou.png',0)

    w = face.shape[1]
    h=face.shape[0]
    print(face.shape)
    print(w,h)
    res = cv2.matchTemplate(gray,face,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_data, pt, (pt[0] + w, pt[1] + h), (7,249,151), 2)
    #显示图像
    cv2.imshow('Detected',img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch,url3):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray
    print(vis.shape)
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    leng=vis.shape[1]
    width=vis.shape[0]

    new_leng=401
    new_width=int(401/leng*width)
    vis=cv2.resize(vis,(new_leng,new_width))
    cv2.imwrite(url3,vis)


def mysurf(url1="statics/images/666.jpg",url2="statics/images/888.jpg",url3=None):
    img1_gray = cv2.imread(url1)
    img2_gray = cv2.imread(url2)
    # SURF进行特征识别
    sift = cv2.xfeatures2d.SURF_create()

    # SIFT进行特征识别
    # sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # BFmatcher with default parms

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)

    drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:50],url3)


if __name__=='__main__':
     pass



