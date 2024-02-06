# -*- coding: utf-8 -*-


import matplotlib.cm as cm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import cv2
import base64


def load_images(directory):
    return [cv2.imread(file_) for file_ in glob.glob('data/*.jpg)')]


def resize_image(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return (width,height)

def run_algorithm(product_image_path ,shelf_image_path,scale_shelf=False,scale_percent=0,encircle_width = 35):
    

    img1 = cv2.imread(product_image_path,0)
    img2  = cv2.imread(shelf_image_path,0)

    ## RGB version of the images are used to show last result on colored one as requested.

    img1_rgb = cv2.imread(product_image_path,1)
    img2_rgb = cv2.imread(shelf_image_path,1)
    
    
    THRESHOLD_MATCH_COUNT = 10
    dim_img1 = resize_image(img1,500)
    dim_img2 = resize_image(img2,scale_percent)

    if scale_shelf==True:
        img2 = cv2.resize(img2,dim_img2,interpolation=cv2.INTER_AREA)
        img2_rgb=cv2.resize(img2_rgb,dim_img2,interpolation=cv2.INTER_AREA)

    img1 = cv2.resize(img1, dim_img1, interpolation = cv2.INTER_AREA)
    img1_rgb=cv2.resize(img1_rgb, dim_img1, interpolation = cv2.INTER_AREA)

    #     Initiate SIFT detector, in upgraded versions of OpenCV this function is patented and not free
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    ## Eliminating not good matches by defining threshold
    if len(good)>THRESHOLD_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    #     polyline_params = dict(isClosed=True,
    #                        color=(255,0,0),
    #                        thickness=10,
    #                        lineType=cv2.LINE_AA,
    #                        shift=0)
    #     img2 = cv2.polylines(img2,[np.int32(dst)],True,(0, 255 ,0),4, cv2.LINE_AA)



    else:
        MIN_MATCH_COUNT = 10
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    ## Defining parameters of draw funciton for insterest points of both image


    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # matchesMask = mask.ravel().tolist()

    # h,w = img1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
    # dst = cv2.perspectiveTransform(pts,M)




    draw_params = dict(matchColor = (0,255,0,0), # draw matches in green color
                       singlePointColor =None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)


    plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    matched_image = cv2.drawMatches(img1_rgb,kp1,img2_rgb,kp2,good,None,**draw_params)

    ## Here, we are encircling the search image on target one.

    cv2.line(matched_image, (int(dst[0,0,0] + img1.shape[1]), int(dst[0,0,1])),\
        (int(dst[1,0,0] + img1.shape[1]), int(dst[1,0,1])), (255,0,0), encircle_width)
    cv2.line(matched_image, (int(dst[1,0,0] + img1.shape[1]), int(dst[1,0,1])),\
        (int(dst[2,0,0] + img1.shape[1]), int(dst[2,0,1])), (0,0,255), encircle_width)
    cv2.line(matched_image, (int(dst[2,0,0] + img1.shape[1]), int(dst[2,0,1])),\
        (int(dst[3,0,0] + img1.shape[1]), int(dst[3,0,1])), (0,0,255), encircle_width)
    cv2.line(matched_image, (int(dst[3,0,0] + img1.shape[1]), int(dst[3,0,1])),\
            (int(dst[0,0,0] + img1.shape[1]), int(dst[0,0,1])), (255,0,), encircle_width)

    cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    # print(dst[0,0,0],dst[0,0,1],dst[1,0,0],dst[1,0,1])


    # Convert the image to base64
    _, buffer = cv2.imencode('.png', matched_image)
    matched_image_base64 = base64.b64encode(buffer).decode('utf-8')



    # Assuming img1 is the image on which the quadrilateral is drawn
    total_image_area = img2.shape[0] * img2.shape[1]

    # Coordinates of the quadrilateral
    quad_coords = [
        (int(dst[0, 0, 0] + img1.shape[1]), int(dst[0, 0, 1])),
        (int(dst[1, 0, 0] + img1.shape[1]), int(dst[1, 0, 1])),
        (int(dst[2, 0, 0] + img1.shape[1]), int(dst[2, 0, 1])),
        (int(dst[3, 0, 0] + img1.shape[1]), int(dst[3, 0, 1])),
    ]

    # Calculate the area of the quadrilateral using Shoelace formula
    quad_area = 0.5 * abs(
        (quad_coords[0][0] * (quad_coords[1][1] - quad_coords[3][1])) +
        (quad_coords[1][0] * (quad_coords[2][1] - quad_coords[0][1])) +
        (quad_coords[2][0] * (quad_coords[3][1] - quad_coords[1][1])) +
        (quad_coords[3][0] * (quad_coords[0][1] - quad_coords[2][1]))
    )

    relative = np.array([[int(dst[0, 0, 0] + img1.shape[1]), int(dst[0, 0, 1])],
     [int(dst[1, 0, 0] + img1.shape[1]), int(dst[1, 0, 1])],
      [int(dst[2, 0, 0] + img1.shape[1]), int(dst[2, 0, 1])],
       [int(dst[3, 0, 0] + img1.shape[1]), int(dst[3, 0, 1])]])

    # Calculate the centroid of the matched region along the y-axis
    centroid_matched_y = np.mean(relative[:, 1])

    # Calculate the centroid of the entire image along the y-axis
    centroid_entire_image_y = img2.shape[0] / 2  # Assuming the y-axis is the vertical axis

    # Calculate the distance along the y-axis
    distance_y = abs(centroid_entire_image_y - centroid_matched_y)

    # Normalize the distance to a value between 0 and 1
    normalized_distance_y = distance_y / img2.shape[0]

    # The closer to 0, the more centrally located the matched region is along the y-axis
    #print(f"The normalized distance along the y-axis is {normalized_distance_y:.2f}")

        # Calculate the percentage of area
    percentage_area = (quad_area / total_image_area) * 100

    #print(f"The quadrilateral occupies {percentage_area:.2f}% of the entire image.")
        
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # average_intensity = cv2.mean(gray_image)[0]
    # _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    shelf_image = cv2.imread(shelf_image_path)

    hsv_image = cv2.cvtColor(shelf_image, cv2.COLOR_BGR2HSV)

    # Extract the saturation channel
    saturation_channel = hsv_image[:,:,1]

    # Calculate the average saturation
    average_saturation = np.mean(saturation_channel)

    print(f"Average Saturation: {average_saturation}")

    # Set thresholds for different lighting conditions
    good_lighting_threshold = 50
    moderate_lighting_threshold = 30
    poor_lighting_threshold = 10

    # Determine lighting conditions
    if average_saturation > good_lighting_threshold:
        lighting_conditions = "Good lighting conditions"
    elif average_saturation > moderate_lighting_threshold:
        lighting_conditions = "Moderate lighting conditions"
    elif average_saturation > poor_lighting_threshold:
        lighting_conditions = "Poor lighting conditions"
    else:
        lighting_conditions = "Very poor lighting conditions"
    
    if normalized_distance_y < 0.2:
        incentive_message = "Offer more incentives to the shopkeeper for placing the product along the central shelf (along the customers eye level)"
        incentive = 0.25
    elif normalized_distance_y < 0.35 and normalized_distance_y > 0.2:
        incentive_message = "Offer less incentives to the shopkeeper for placing the product not at the corners of the shelf but also not at the center."
        incentive = 0.1
    else:
        incentive_message = "Don't offer any incentive to the shopkeeper. The product is not placed very well and is along the corners."
        incentive = 0

    return normalized_distance_y, percentage_area, matched_image_base64, lighting_conditions, incentive_message
    # print(f"Incentive value = {incentive*100}%")

    # plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)),plt.show()




