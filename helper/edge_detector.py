import json
import multiprocessing
import os
import random

import cv2
import numpy as np
import scipy.signal
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numpyencoder import NumpyEncoder
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from matplotlib import rc

def brute_edge_finder(stick_dir, stick):
    """
    Function to detect edges in an image by a brute force approach

    Input:
        stick_dir: directory of the image that should be analysed
        stick: name of the file

    Output:
        detected rects: list of all detected rectangles in the input image
    """

    img_cntr = 0
    detected_rects = []
    # read stick file
    img_orig = cv2.imread(stick_dir + "/" + stick)
    # copy file for drawing the found rectangles
    img4draw = img_orig.copy()

    # border_id: Use image without border | add black border on bottom and top | add white border to bottom and top
    for border_id in [0, 1, 2]:
        if border_id == 1:
            d_y = 20
            img = cv2.copyMakeBorder(img_orig, top=d_y, bottom=d_y, left=0,
                                     right=0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif border_id == 2:
            d_y = 20
            img = cv2.copyMakeBorder(img_orig, top=d_y, bottom=d_y, left=0,
                                     right=0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            img = img_orig.copy()
            d_y = 0
        # convert to hsv colorspace
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        height, width, _ = img.shape

        edges = np.zeros(shape=[height, width], dtype=np.uint8)
        # Channels: Increase contrast by multiplying single channels
        for channel in [0, 1, 2, 3, 4, 5, 6]:

            for contrast in range(1, 5, 1):
                if channel < 3:
                    # increase contrast of H, S or V channel
                    gray = hsv[:, :, channel]*contrast
                elif channel < 6:
                    # increase contrast of R, G, or B channel
                    gray = img[:, :, channel-3]*contrast
                else:
                    # use grayscaled image with increased contrast
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)*contrast

                # Application of different filters
                for filter_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    if filter_id == 1:
                        gray = cv2.fastNlMeansDenoising(gray, None)
                    elif filter_id == 2:
                        gray = cv2.bilateralFilter(gray, 7, 50, 50)
                    elif filter_id == 3:
                        gray = cv2.blur(gray, (9, 9))
                    elif filter_id == 4:
                        gray = cv2.morphologyEx(
                            gray, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                    elif filter_id == 5:
                        filter = np.array(
                            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                        gray = cv2.filter2D(gray, -1, filter)
                    elif filter_id == 6:
                        gray = cv2.morphologyEx(
                            gray, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                    elif filter_id == 7:
                        gray = cv2.medianBlur(gray, 5)
                    elif filter_id == 8:
                        gray = cv2.medianBlur(gray, 9)
                    elif filter_id == 9:
                        gray = cv2.bilateralFilter(gray, 9, 75, 75)
                    elif filter_id == 10:
                        gray = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    elif filter_id == 11:
                        gray = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
                    elif filter_id == 12:
                        gray = cv2.GaussianBlur(gray, (5, 5), 0)

                    # Variation of thresholds of the Canny edge detection
                    for sat_val in [0, 1, 10, 20, 40]:
                        # sat_val = 0
                        for apertureSize in [3, 5, 7]:
                            edges = cv2.Canny(gray, sat_val, 120,
                                              apertureSize=apertureSize)

                            # Find contours in Canny output image
                            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                            # Analyse all found contours
                            for c in cnts:
                                area = cv2.contourArea(c)
                                # only continue at certain area size of the contour
                                if area > ((height-d_y)*(height-d_y))*0.2:
                                    approx = cv2.approxPolyDP(
                                        c, 0.01 * cv2.arcLength(c, True), True)

                                    x, y, w, h = cv2.boundingRect(approx)
                                    rect = [x, y, w, h]
                                    aspectRatio = float(w)/h

                                    # only continue if bounding rectangle of the contour is a square
                                    if aspectRatio < 1.06 and aspectRatio > 0.94:
                                        # check if rect is on left side
                                        if (rect[0] > 0.2*width):
                                            rect[1] -= d_y
                                            # Add to list of found rectangles
                                            detected_rects.append(rect)
                                            # cv2.rectangle(
                                            #     img4draw, (x, y-d_y), (x+w, y-d_y+h), (0, 255, 0), 2)
                                            img_cntr += 1
                                            # print("Success Parameters: ", apertureSize,
                                            #       sat_val, filter_id, contrast, channel, border_id)
    return detected_rects


def cluster_rects(detected_rects, image):
    """
    Cluster with K Means center points for the detected rectangles in the respective image

    Input:
        detected_rects: list of found rectangles (rectangle = x, y, w, h)
        image:  Image file where the rectangles where detected in
    Output
        centers: list containing the calculated center points 
    """
    height, width, _ = image.shape
    X = []
    field_width = []
    for rect in detected_rects:
        X.append((rect[0]+rect[2]/2, rect[1]+rect[3]/2))
        field_width.append(rect[2])

    # check how many clusters are here
    residual = []
    for n_clusters in range(2, 12):
        kmeans = KMeans(n_clusters=n_clusters)
        X = np.array(X)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        residual.append(kmeans.inertia_)
        centers.sort(0)
        allowed_dis = min(
            np.median(centers[1:, 0]-centers[0:-1, 0]), height)
        min_dis = np.min(centers[1:, 0]-centers[0:-1, 0])

        if min_dis < 0.7*allowed_dis and n_clusters > 8:
            n_clusters += -1
            break
    else:
        n_clusters = 11

    # get number of meaningfull clusters
    idx = [idx for idx in range(len(residual))
           if residual[idx] <= residual[-1]*1.05]
    #n_clusters = 11
    kmeans = KMeans(n_clusters=n_clusters)
    X = np.array(X)
    kmeans.fit(X)
    # centers of clusterd data
    centers = kmeans.cluster_centers_

    centers.sort(0)

    distances = centers[1:, 0]-centers[0:-1, 0]

    median_dis = np.median(distances)
    med_field_width = np.median(field_width)

    del_idx = []
    sus_idx = []

    # check results and delete suspicious centers
    for dist in distances:
        if dist < 0.4 * med_field_width:
            idx = np.where(distances == dist)
            for i in range(len(idx[0])):
                del_idx.append(int(idx[0][i]))
        elif dist > 0.4 * med_field_width and dist < med_field_width:
            idx = np.where(distances == dist)
            for i in range(len(idx[0])):
                del_idx.append(int(idx[0][i] + 1))
    if del_idx:
        centers = np.delete(centers, del_idx, axis=0)

    distances = centers[1:, 0]-centers[0:-1, 0]
    median_dis = np.median(distances)

    # Fill the gaps
    while len(centers) < 11:

        if np.max(distances) > 1.5 * np.median(distances) and np.max(distances) < 2.5*np.median(distances):
            # found one space
            idx = np.where(distances == np.max(distances))
            add_center = centers[idx[0]].copy()
            add_center[0][0] += np.max(distances)/2
            centers = np.insert(centers[:], int(idx[0][0]+1), add_center, 0)
            centers.sort(0)
            distances = centers[1:, 0]-centers[0:-1, 0]
        elif np.max(distances) > 2.5 * med_field_width and np.max(distances) < 3.5*np.median(distances):
            idx = np.where(distances == np.max(distances))
            add_center_1 = centers[idx[0]].copy()
            add_center_1[0][0] += np.max(distances)/3
            add_center_2 = add_center_1.copy()
            add_center_2[0][0] += np.max(distances)/3
            centers = np.insert(centers[:], int(idx[0][0]+1), add_center_1, 0)
            centers = np.insert(centers[:], int(idx[0][0]+2), add_center_2, 0)
            centers.sort(0)
            distances = centers[1:, 0]-centers[0:-1, 0]
        elif np.max(distances) > 3.5*np.median(distances):
            idx = np.where(distances == np.max(distances))
            add_center_1 = centers[idx[0]]
            add_center_1[0][0] += np.max(distances)/4
            add_center_2 = add_center_1.copy()
            add_center_2[0][0] += np.max(distances)/4
            add_center_3 = add_center_2.copy()
            add_center_3[0][0] += np.max(distances)/4
            centers = np.insert(centers[:], int(idx[0][0]+1), add_center_1, 0)
            centers = np.insert(centers[:], int(idx[0][0]+2), add_center_2, 0)
            centers = np.insert(centers[:], int(idx[0][0]+3), add_center_3, 0)
            centers.sort(0)
            distances = centers[1:, 0]-centers[0:-1, 0]

        else:
            break

    # if centers are missing and image is big enough: add points to end -> are more often missing
    if (len(centers) < 11):
        delta = 11 - len(centers)
        median_dis = np.median(distances)

        while (len(centers) < 11):
            new_point = centers[-1].copy()
            new_point[0] = new_point[0] + median_dis
            if(new_point[0] < int(width*0.95)):
                centers = np.append(centers, [new_point], axis=0)
            else:
                break

    # if still missing points: add to front
    if(len(centers) < 11):
        while (len(centers) < 11):
            new_point = centers[0].copy()
            new_point[0] = new_point[0] - median_dis
            centers = np.append(centers, [new_point], axis=0)
            centers.sort(0)
    centers.sort(0)

    # fig.savefig("./results/contour/" + stick[0:-4] + '.png', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', papertype=None, format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    # plt.close(fig)

    return centers


def find_saved_rects(stick):
    """
    Function to find already detected rectangles within the json files 

    Input:
        stick: filename of the stick image

    Output: 
        data: dict containing the rectangles
    """
    json_dir = os.getcwd() + "\\results\\rectangles\\"
    json_name = os.path.splitext(stick)[0] + ".json"

    if os.path.isfile(json_dir + json_name):
        data = json.load(open(json_dir + json_name))
    else:
        print("Saved rects not found")
        data = []
    return data


def plot_rect(rects, centers, image):
    """
    Plots the stick image, the found rectangles and the calculated center points

    Input:
        rects: list with all rectangles
        centers: list containing the center points
        image: image of the stick
    """
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img4draw = image.copy()
    height, width, _ = image.shape
    for rectangle in rects:
        x, y, w, h = rectangle
        y = int((height/2) - (h/2))
        cv2.rectangle(img4draw, (x, y), (x+w, y+h), (0, 255, 0), 2)

    fig = plt.figure(figsize=[6.2, 6])
    plt.subplot(311)
    plt.imshow(image)
    plt.title("Extracted stick image", fontsize=12)

    plt.subplot(312)
    plt.imshow(img4draw)
    plt.title("Found squares during edge detection", fontsize=12)

    plt.subplot(313)
    plt.imshow(image)
    plt.scatter(centers[:, 0], centers[:, 1], c="#00ff00")
    plt.title("Calculated center points of test fields", fontsize=12)

    plt.show()

    plt.close(fig)


def debug_edge_detector():
    """
    Helper function to test and debug the edge detector

    """

    stick_dir = os.getcwd() + "\\results\\found_sticks\\"
    all_sticks = os.listdir(stick_dir)
    random.shuffle(all_sticks)
    # print(len(all_sticks))
    #
    check_dir = os.getcwd() + "\\results\\rectangles"
    done = os.listdir(check_dir)

    for stick in all_sticks:
        rects_json = dict()
        filename = os.path.splitext(stick)[0] + ".json"
        if not (filename in done):
            print("not found")
            break
            print("Getting Rectangles: "+filename)
            detected_rects = brute_edge_finder(stick_dir, stick)
            rects_json["rectangles"] = detected_rects
            full_filename = "./results/rectangles/" + filename
            with open(full_filename, 'w') as fp:
                json.dump(rects_json, fp, indent=4, cls=NumpyEncoder)
            fp.close()
        else:
            print("Already in set: " + filename)

        # stick = '1604422025295-16044218898194693381376830618597.jpg'
        stick_image = cv2.imread(stick_dir+stick)
        data = find_saved_rects(stick)
        found_rects = data["rectangles"]
        if len(found_rects) > 11:
            cents = cluster_rects(found_rects, stick_image)
            plot_rect(found_rects, cents, stick_image)


    # stick = "1605367130751-image.jpg"
    # img = cv2.imread(stick_dir+stick)
    # centers = cluster_rects(rects, img)
if __name__ == "__main__":
    debug_edge_detector()
