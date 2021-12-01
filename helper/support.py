import json
import numpy as np
import os
from numpyencoder import NumpyEncoder
import cv2

from helper import field_extraction as fex
from helper import color_calculation as colcalc
def store_results(results, filename):
    """
    Create or open a JSON file where the calculated test results are saved

    Input:
        test_results: dictionary containing the calculated results

    Output:
        JSON File: saved to "./results/results.json"
    """
    with open(filename, 'w') as fp:
        json.dump(results, fp, indent=4, cls=NumpyEncoder)
    fp.close()
    print("Results stored: ", filename)


def sort_imgs(image_list, subject_id, EVAL_TIMES):
    """
    Function to extract and group the corresponding images in a list of all images
    with names like: Step_30_aligned_reference.jpg

    Input:
        image_list: Contains all filenames of the images as string
        subject_id: id of the respective subject.
    Output:
        step_pairs: dictionary with all found image pairs corresponding to one step
                    e.g. {30: [Step_30_aligned_stick.jpg, Step_30_aligned_reference.jpg]}
    """
    step_pairs = dict()
    for step in EVAL_TIMES:
        step_images = []

        if step == 120:
            try:
                step_images.append(next(img for img in image_list if (
                    img[5:8] == str(step) and img[-13:-4] == "reference")))
                step_images.append(next(img for img in image_list if (
                    img[5:8] == str(step) and img[-9:-4] == "stick")))
                step_pairs[step] = step_images
                # step_pairs.append(step_images)
            except:
                print("At least one image not found for: {}.".format(subject_id))
        else:
            try:
                step_images.append(next(img for img in image_list if (
                    img[5:7] == str(step) and img[-13:-4] == "reference")))
                step_images.append(next(img for img in image_list if (
                    img[5:7] == str(step) and img[-9:-4] == "stick")))
                step_pairs[step] = step_images
            except:
                print("At least one image not found for: {}.".format(subject_id))

    return step_pairs


def crop_rect(img, rect):
    """
    Function, that crops a rectangle from an image

    Input:
      img: Whole image
      rect: Parameters of the rectangle that will be cut out: in the form center

    Output:
      detectedStick
    """

    # Shape of the Image
    height, width = img.shape[0], img.shape[1]

    # Create border, to prevent cutting off image parts during rotation
    # T = np.float32([[1, 0, width/2], [0, 1, height/2]])
    # img_mod = cv2.warpAffine(img, T, (width*2, height*2))

    # Parameters of the rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]

    center, size = tuple(map(int, center)), tuple(map(int, size))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop


def check_refcard(reffield):
    """
    Checks wether the reference card was detected properly by comparing with the original reference

    Input:
        reffield: list of separated reference field of the detected reference card

    Output:
        ref_detected: Boolean, indicating, wether the card was detected properly or not
    """
    ref_detected = True
    sus_fields = 0
    # Reference Image:
    card_file = os.getcwd()+ "/referenceimages/RefCard.jpg"

    # read original reference card and seperate the single fields
    ref_card = cv2.imread(card_file)
    ref_reffield = fex.separate_reffields(ref_card)
    if (ref_reffield == 0):
        print("Refcard sus. ")

        return False
    # calculate colors for all fields
    hue_ref_refs = colcalc.calculate_colors(ref_reffield, 0)
    hue_refs = colcalc.calculate_colors(reffield, 0)

    # compare each field, if the difference is too big: field is sus
    for i in range(len(hue_ref_refs)):
        for j in range(len(hue_ref_refs[i])):

            delta = abs(hue_ref_refs[i][j]-hue_refs[i][j])

            if (delta > 60):
                sus_fields += 1
    # if too many sus fields detected: reference card not detected properly
    if sus_fields > 10:
        ref_detected = False
        # cv2.imshow('Refcard', reffield[-1][-1])
        # cv2.waitKey(0)
    return ref_detected
