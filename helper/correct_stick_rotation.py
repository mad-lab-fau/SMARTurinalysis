import cv2
import os
import numpy as np
import time
from statistics import mean
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Folder with reference stick
stick_file = os.path.join(os.path.dirname(os.path.dirname(
    __file__)), "../referenceimages/UrineStick.jpg")

Stick = cv2.imread(stick_file)


def correct_rotation(image, reference, good_match_percent):
    # Grayscaling Images
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Detecting ORB features and computing descriptors.
    orb = cv2.ORB_create()
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
    if matches == 0:
        return image
    # Extract coordinates of good matches
    match_points = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        match_points[i, :] = keypoints1[match.queryIdx].pt
    if (len(match_points) < 1):
        return image
    else:
        mean_x = mean(match_points[:, 0])
        height, width, _ = image.shape
        if (mean_x > width/2):
            rotated_stick = cv2.rotate(image, cv2.ROTATE_180)
        else:
            rotated_stick = image
        return rotated_stick


if __name__ == "__main__":
    STICK_DIR = askdirectory(title='Select folder')
    files = os.listdir(STICK_DIR)
    print("Correcting rotation of sticks")
    for image_file in files:
        imagepath = os.path.join(STICK_DIR, image_file)
        image = cv2.imread(imagepath)
        stick_corrected = correct_rotation(image, Stick, 0.1)
        cv2.imwrite(imagepath, stick_corrected)

    print("Finished.")
