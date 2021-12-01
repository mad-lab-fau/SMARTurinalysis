import cv2
import numpy as np
import time
import imutils
import math
import os

from matplotlib import pyplot as plt
from helper import support as spt

###################################################################
#### Object Detector for Urine sticks and their reference card ####
###################################################################


# maximum number of features that are searched during the feature detection
MAX_FEATURES = 10000

# Directory of all Subjects
DATA = os.path.join(os.path.dirname(__file__), "../control_urine")

# Reference Images:
card_file = os.path.join(os.path.dirname(
    __file__), "../referenceimages/RefCard.jpg")
RefCard = cv2.imread(card_file)

stick_file = os.path.join(os.path.dirname(
    __file__), "../referenceimages/UrineStick.jpg")
Stick = cv2.imread(stick_file)


def detect_refcard(image):
    """
    Function that detects the Reference Card

    Input:
      image: Photo, that will be analysed

    Output:
      Ref_card: Cropped image of the detected reference card
      corners: Coordinates of the corner points of the Reference Card in the original image

    """
    Ref_card, corners = find_object(image, RefCard, 0.15)
    return Ref_card, corners


def detect_stick(image, corners_refcard):
    """
    Function, that finds and crops the UrineStick in the photograph

    Input:
      image: Photograph, that is analysed
      corners_refcard: Coordinates of the corners of the detected Reference Card in order to fill it black
                      for the easier detection of the Urine Stick

    Output:
      detected_stick: Image of the detected Stick

    """

    # Measurements of the original Reference Card
    h, w, _ = RefCard.shape
    corners_refcard[0] = corners_refcard[0]*0.95
    corners_refcard[1][0] = corners_refcard[1][0]*0.95
    corners_refcard[1][1] = corners_refcard[1][1]*1.05
    corners_refcard[2] = corners_refcard[2]*1.05
    corners_refcard[3][0] = corners_refcard[3][0]*1.05
    corners_refcard[3][1] = corners_refcard[3][1]*0.95

    # Fill the RefCard black to improve Stick Detection
    cv2.fillPoly(image, [corners_refcard], 0)

    img_h, img_w, _ = image.shape
    T = np.float32([[1, 0, img_w/2], [0, 1, img_h/2]])
    img_mod = cv2.warpAffine(image, T, (img_w*2, img_h*2))

    # Detect the stick in the image
    detected_stick, stick_corners = find_object(img_mod, Stick, 0.1)

    # Warping often deforms the stick too much:
    # Alternative solution that uses the Corner coordinates

    # Calculation of the Bounding rectangle
    rectangle = cv2.minAreaRect(stick_corners)

    # Calculation of the Rotation and the rotated image (support function)
    detected_stick = spt.crop_rect(img_mod, rectangle)

    # Correction of the Orientation of the Stick
    im_size = detected_stick.shape
    angle_is = 0

    if im_size[1] < im_size[0]:
        detected_stick = imutils.rotate_bound(detected_stick, -90)
        angle_is = 90

    delta = stick_corners[2] - stick_corners[1]
    angle_target = math.atan2(delta[1], delta[0])/math.pi*180

    angle_is = angle_is + rectangle[2]

    if abs(angle_is-angle_target) > 70:
        detected_stick = imutils.rotate_bound(detected_stick, 180)

    return detected_stick, stick_corners


def find_object(image, reference, good_match_percent):
    """
    Function to find matching features

    Input:
      image: Photograph, that shall be analysed
      reference: Reference photograph of the object that is searched
      good_match_percent: Part of the best matches that will be used for further analysis

    Output:
      detected_object: warped and cropped image of the detected object
      corners: coordinates of the object in the input photograph(image)
    """

    # Grayscaling Images
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Detecting ORB features and computing descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(image_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_gray, None)

    # Feature Matching (Brute Force)
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by quality
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove worse matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]
    print("Number of matches: {}".format(len(matches)))

    # Draw best matches and write to directory
    drawn_matches = cv2.drawMatches(
        image, keypoints1, reference, keypoints2, matches, None)
    directory = os.getcwd() + "/results/matches/" + \
        os.path.basename(os.path.dirname(path))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = "/" + str(time.time()) + "_matches.jpg"
    cv2.imwrite(directory + filename, drawn_matches)

    # Extract coordinates of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Calculate the homography matrix
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    h_rev, mask_rev = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Calculate the coordinates of the corners
    height, width, channel = reference.shape
    top_left = h_rev.dot(np.array([0, 0, 1]))
    down_left = h_rev.dot(np.array([0, height, 1]))
    down_right = h_rev.dot(np.array([width, height, 1]))
    top_right = h_rev.dot(np.array([width, 0, 1]))

    corner_coord = np.array([top_left[:2]/top_left[2],
                             down_left[:2]/down_left[2],
                             down_right[:2]/down_right[2],
                             top_right[:2]/top_right[2]], dtype='int32')

    # Use homography to warp perspective and crop the detected object
    detected_object = cv2.warpPerspective(image, h, (width, height))
    return detected_object, corner_coord


def filter_refcard(img_refcard):
    
    kernel = np.ones((5, 5), np.uint8)
    image_filtered = cv2.medianBlur(img_refcard, 7)
    image_filtered = cv2.morphologyEx(image_filtered, cv2.MORPH_OPEN, kernel)
    image_filtered = cv2.morphologyEx(image_filtered, cv2.MORPH_CLOSE, kernel)
    return image_filtered


def main():
    print("Starting ...")
    global path
    detected_coordinates = dict()

    # all images are stored in DATA:
    subjects = os.listdir(DATA)
    sum_saved_refs = 0
    sum_errors = 0

    # Iterate through subjects
    for subject_id in subjects:
        subject_dir = DATA + "/" + subject_id
        steps = os.listdir(subject_dir)

        # Iterate through steps
        for step in steps:
            image_dir = subject_dir + "/" + step

            if os.path.isdir(image_dir):

                try:
                    # list files in subdirectory
                    files = os.listdir(image_dir)
                    times_created = []

                    # find original image if other images present
                    for filename in files:
                        times_created.append(os.path.getctime(
                            image_dir + "/" + filename))
                    
                    # find oldest image in the directory 
                    minpos = times_created.index(min(times_created))
                    path = image_dir
                    image = cv2.imread(path + "/" + files[minpos])

                    detected_coordinates[files[minpos]] = dict()

                    image_filtered = image

                    # detect the reference card
                    ref_card, corners = find_object(image, RefCard, 0.1)
                    
                    # write image of detected ref_card into the results folder
                    path_aligned = os.getcwd() + "/results/aligned_objects/" + subject_id
                    if not os.path.exists(path_aligned):
                        os.makedirs(path_aligned)
                    card_file = "/Step_" + step + "_aligned_reference.jpg"
                    print("Saving aligned Reference : ",  card_file)
                    cv2.imwrite(path_aligned + card_file, ref_card)
                    sum_saved_refs += 1

                    # Detect the urine stick 
                    stick, stick_corners = detect_stick(image, corners)
                    detected_coordinates[files[minpos]
                                         ]["stick_coordinates"] = stick_corners
                except:
                    print(image_dir + " An error occured.")
                    sum_errors += 1
                    
    print("Number of saved Refcards: ", sum_saved_refs)

    # Save coordinates of detected objects to json file
    spt.store_results(detected_coordinates, "./results/corners.json")
    print("Finished.")


if __name__ == "__main__":
    main()
