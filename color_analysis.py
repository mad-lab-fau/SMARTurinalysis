import math
import os

import cv2
import numpy as np
import scipy.signal
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpyencoder import NumpyEncoder

from helper import color_calculation as colcalc
from helper import field_extraction as fex
from helper import image_selection as slc
from helper import support as spt

##################################################################
######### Color Analyzer for the detected Urine sticks  ##########
##################################################################

# Evaluation times as specified by the manufacturer
EVAL_TIMES = [30, 40, 45, 60, 120]

# Directory of original images
DATA = os.path.join(os.path.dirname(__file__), "../data")

RESULTS_DIR = os.getcwd() + "/results/aligned_objects_subjects"


def compare_colors(index, hue_values_stick, hue_values_refcard, max_hsv_stick, max_hsv_ref, hsv_coord_ref, hsv_coord_stick):
    """
    Function, to compare the colors of the a testfield with the reference card
    
    Input:
        index_stick: Index of the Parameter that will be compared:
                        0 - Glucose
                        1 - Bilirubin
                        2 - Ketone
                        3 - Specific gravity
                        4 - Blood
                        5 - pH
                        6 - Proteine
                        7 - Urobilinogen
                        8 - Nitrite
                        9 - Leucocytes
        hue_values_stick: Hue Values that were calculated for the testfields of dipstick
        hue_values_refcard: Calculated hue values of the whole reference card
        max_hsv_stick: Values of H, S and V channel for the testfields 
        max_hsv_ref: Values of H, S and V channel for the reference card 
        hsv_coord_ref: Calculated coordinates of the colorfields
        hsv_coord_stick: Calculated coordinates of the test fields

    Output:
        comparisons: list of similarity values for the respective Parameter, ranging from 0 to 1
        comparisons_abs: absolute difference of the hue values
        matching_factors: calculated matching factor of the max_hsv values 
        hsv_distance: Distance of the color coordinates
        
    """
    comparisons = []
    comparisons_abs = []
    matching_factors = []
    hsv_distance = []

    hue_stick = hue_values_stick[index]

    rgb_stick = max_hsv_stick[index]

    coord_stick = hsv_coord_stick[index]

    for j in range(len(hue_values_refcard[index])):

        # Calculate similarity and save in array
        hue_ref = hue_values_refcard[index][j]

        if hue_stick == 0 and hue_ref == 0:
            comparisons.append(0.0)
            comparisons_abs.append(0)
        else:
            #similarity = min(hue_ref, hue_stick)/max(hue_ref, hue_stick)

            sim_abs = abs(hue_stick - hue_ref)
            comparisons_abs.append(sim_abs)
            similarity = 1 - sim_abs/180
            comparisons.append(similarity)

        # Matching Factor RGB
        rgb_ref = max_hsv_ref[index][j]
        # Calculate Matching Factor acc. to Ra et al. "Smartphone-Based Point-of-Care Urinalysis under variable illumination"
        delta_red = abs(rgb_stick[0] - rgb_ref[0])
        delta_green = abs(rgb_stick[1] - rgb_ref[1])
        delta_blue = abs(rgb_stick[2] - rgb_ref[2])
        # m_factor = 1 - ((delta_red + delta_green + delta_blue)/(255*3))
        m_factor = 1 - ((0.6429*delta_red + 0.1786 *
                         delta_green*100/255 + 0.1786*delta_blue*100/255)/(180+100+100))
        matching_factors.append(m_factor)

        # Coordinate things
        coord_ref = hsv_coord_ref[index][j]
        # get distance by Pythagoras
        distance = math.sqrt(math.pow(coord_ref[0]-coord_stick[0], 2)+math.pow(
            coord_ref[1]-coord_stick[1], 2)+math.pow(coord_ref[2]-coord_stick[2], 2))
        distance = 1 - \
            (distance/(math.sqrt(math.pow(512, 2)+math.pow(255, 2))))
        hsv_distance.append(distance)
    return comparisons, comparisons_abs, matching_factors, hsv_distance


def analyze_blood(testfield_image):
    """
    Function to determine if the blood is hemolysed.

    Input:
        testfield_img: Image of the testfield for blood.

    Output:
        blood_status:
            0: no darker spots were found, no blood present or blood is hemolysed
            1: few dark spots were found, result corresponds to the second reference field of blood
            2: many dark spots were found, result corresponds to the third reference field of blood
    """
    blood_status = 0
    hist = cv2.calcHist([testfield_image], [0], None, [180], [0, 256])
    hist_flat = hist.flatten()
    peak_index, p_values = scipy.signal.find_peaks(hist_flat, height=1000)

    if len(peak_index) > 2:
        peak_values = list(p_values.values())
        peaks = peak_values[0]
        sorted_peaks = sorted(peaks, reverse=True)
        if (sorted_peaks[1]/sorted_peaks[0]) > 0.9:
            blood_status = 2
        else:
            blood_status = 1
    return blood_status



def calculate_results(subject, eval_time, testfields, reffields):
    """
    Function to print out the results of the colorimetric analysis

    Input:
        subject: ID of the subject
        eval_time: Evaluation time, after which the input image was taken:
                    30: Glucose, Bilirubin
                    40: Ketone
                    45: Specific gravity
                    60: Blood, pH, Proteine, Urobilinogen, Nitrite
                    120: Leucocytes

        test_fields: List of images of the separated testfields

        reffields: List of images of the separates reference fields

    """

    analytes = ['Glucose', 'Bilirubin', 'Ketone', 'SpecificGravity', 'Hemoglobin', 'pHValue', 'Protein',
                'Urobilinogen', 'Nitrite', 'Leukocytes']

    # Calculate the hue values for each field
    hue_stick = colcalc.calculate_colors(testfields, 1)
    hue_ref = colcalc.calculate_colors(reffields, 0)

    # Calculate mean rgb values for matching factor
    max_hsv_ref, max_hsv_stick = colcalc.calculate_hsv(
        reffields, testfields)

    # Caculate coordinates
    hsv_coord_ref, hsv_coord_stick = colcalc.calc_color_coordinates(
        reffields, testfields)

    # results lists
    # sim_values = []
    # mf_values = []
    print('Calculating results for the evaluation time: {} of {}'.format(
        eval_time, subject))
    TEST_RESULTS[subject][eval_time] = dict()
    print('Saving results for {}.'.format(subject))

    for i in range(len(analytes)):
        TEST_RESULTS[subject][eval_time][analytes[i]] = dict()

        # Compare calculated color values and save them
        sim, sim_abs, match_f, hsv_dist = compare_colors(
            i, hue_stick, hue_ref, max_hsv_stick, max_hsv_ref, hsv_coord_ref, hsv_coord_stick)
        # sim_values.append(sim)
        # mf_values.append(match_f)

        # Save Hue Value of the stick into dictionary
        TEST_RESULTS[subject][eval_time][analytes[i]
                                         ]["Hue_Stick"] = hue_stick[i]
        # Save Hue Value of the testfield into dictionary
        TEST_RESULTS[subject][eval_time][analytes[i]
                                         ]["Hue_Ref"] = hue_ref[i]

        # Save similarities into dictionary
        TEST_RESULTS[subject][eval_time][analytes[i]
                                         ]["Similarities"] = sim

        # Save absolute similarities into dictionary
        TEST_RESULTS[subject][eval_time][analytes[i]
                                         ]["Similarities_absolute"] = sim_abs

        # Save similarities into dictionary
        TEST_RESULTS[subject][eval_time][analytes[i]
                                         ]["Matching_factor"] = match_f
        # Save color distances into dictionary
        TEST_RESULTS[subject][eval_time][analytes[i]
                                         ]["Color_distances"] = hsv_dist
        # Analyze Blood (if dots are present)
        if i == 4:
            blood_status = analyze_blood(testfields[4])
            if blood_status == 2:
                TEST_RESULTS[subject][eval_time][analytes[i]
                                                 ]["Blood_Status"] = "Non-hemolysed blood"
            elif blood_status == 1:
                TEST_RESULTS[subject][eval_time][analytes[i]
                                                 ]["Blood_Status"] = "Traces of non-hemolysed blood"
            else:
                TEST_RESULTS[subject][eval_time][analytes[i]
                                                 ]["Blood_Status"] = "neg or hemolysed"
    return TEST_RESULTS


def calculate_folder():
    """
    Calculates results for all aligned objects in results directory

    Output:
        JSON file with calculated results
    """
    print("Starting color analysis ...")

    all_subjects = os.listdir(RESULTS_DIR)
    global TEST_RESULTS

    TEST_RESULTS = dict()

    for subject_id in all_subjects:
        print(subject_id)
        subject_dir = RESULTS_DIR + "/" + subject_id
        TEST_RESULTS[subject_id] = dict()
        aligned_images = os.listdir(subject_dir)

        # Support function to sort the list of images inside the folder
        img_dict = spt.sort_imgs(aligned_images, subject_id, EVAL_TIMES)

        # Iterate through all steps
        for step in EVAL_TIMES:
            if (not step in img_dict):
                print("No images for time step {} of {}".format(step, subject_id))
                continue
            ref_file = subject_dir + "/" + img_dict[step][0]
            stick_file = subject_dir + "/" + img_dict[step][1]

            # Check if images can be read
            if (not cv2.haveImageReader(stick_file) or not cv2.haveImageReader(ref_file)):
                print("Error reading aligned images for {} at step {}".format(
                    subject_id, step))
                continue
            reference = cv2.imread(ref_file)
            stick = cv2.imread(stick_file)

            testfields = fex.separate_testfields(
                img_dict[step][1], subject_dir, stick, subject_id, str(step))
            if not testfields:
                print("Testfields were not separated")
                continue


            reffields = fex.separate_reffields(reference)
            if (testfields == 0 or reffields == 0):
                print("Testfields or reffields could not be found, check {} at time step {}".format(
                    subject_id, step))
                continue

            ref_detected = spt.check_refcard(reffields)

            if(not ref_detected):
                print("Warning! Not sure if refcard was detected properly, please check for {} at time step {}".format(
                    subject_id, step))
            
            # Calculate test results for subject
            TEST_RESULTS = calculate_results(
                subject_id, int(step), testfields, reffields)

    # Store results into result folder
    spt.store_results(TEST_RESULTS, "./results/results.json")
    print('Finished.')


if __name__ == "__main__":
    calculate_folder()
