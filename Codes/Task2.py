import cv2
import numpy as np

def stitch():
    image1 = cv2.imread(r"C:\Users\anast\Desktop\PROGRAMMING\ImageProcessing\Project\ImageProcessing\Scenes\S19.png",cv2.IMREAD_COLOR)
    image2 = cv2.imread(r"C:\Users\anast\Desktop\PROGRAMMING\ImageProcessing\Project\ImageProcessing\Scenes\S18.png",cv2.IMREAD_COLOR)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {})

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(image1, homography_matrix, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    result[:, 0:image2.shape[1]] = image2
    cv2.imwrite("Stitched_Image.png", result)
    return 
    

stitch()