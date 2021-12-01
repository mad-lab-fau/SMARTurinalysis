import os
import json
import tkinter
from tkinter import messagebox

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from helper import edge_detector as edt
from numpyencoder import NumpyEncoder


def separate_reffields(img_ref):
    """
    This function separates the reference fields of the Reference Card

    Input:
        image: image of the detected reference card
    Output:
        ref_fields: List with the separated image parts, starting with fields for glucose, ending with leucocytes
    """
    row, col, _ = img_ref.shape

    # Measured positions and distances
    start_x = 0.186
    start_y = 0.956
    next_col = 0.104
    row_height = 0.039

    # percentual positions of rows from glucose to leucocytes
    next_row = [0, 0.089, 0.157, 0.237, 0.295,
                0.377, 0.435, 0.513, 0.581, 0.644]

    ref_fields = []

    # Seperate all reference fields (whites excluded)
    for r in range(0, 10):
        row_imgs = []
        for c in range(0, 7):
            lower_row = int(start_y*row-next_row[r]*row)
            upper_row = int(start_y*row-next_row[r]*row-row_height*row)
            lower_col = int((start_x+c*next_col)*col)
            upper_col = int((start_x+(c+1)*next_col)*col)
            new_img = img_ref[upper_row:lower_row, lower_col:upper_col]
            height, width, _ = new_img.shape
            new_img = new_img[int(0.2*height):int(0.8*height),
                              int(0.2*width):int(0.8*width)]
            row_imgs.append(new_img)
        if (r == 0 or r == 2 or r == 6):
            # glucose, ketone, protein
            del row_imgs[1]
        if (r == 1):
            # bilirubin
            del row_imgs[1:4]
        if (r == 7):
            # urobilinogen
            del row_imgs[2:4]
        if (r == 8):
            # nitrite
            del row_imgs[1:4]
            del row_imgs[-2]
        if (r == 9):
            del row_imgs[1:3]
        ref_fields.append(row_imgs)
        if (np.count_nonzero(row_imgs[0]) == 0):
            ref_fields = 0
            break
    return ref_fields


def separate_testfields(filename, stick_dir, img_stick, subject, step):
    """
    Separates the 10 testfields of the Urine Stick into single image parts

    Input:
         filename: stick filename
         stick_dir: directory of the stick file
         img_stick: image of the detected urine stick
         subject: Subject Id
         step: time step (30, 40 45, 60 or 120)
    Output:
        testfields: List with the separated image parts, starting with glucose, ending with leucocytes
     """
    image_dir = os.getcwd()+"\\data\\" + subject + "\\" + step
    dest_file = os.listdir(image_dir)

    # Calculate x coordinates of testfields
    height, width, _ = img_stick.shape

    data = edt.find_saved_rects(dest_file[0])
    found_rects = np.array(data["rectangles"])

    status = data["found"]
    if not status:
        testfields = []
        return testfields

    if (len(found_rects) > 11):
        try:
            centers = edt.cluster_rects(found_rects, img_stick)
        except:
            print("Clustering impossible for subject:  {}, Step: {}".format(
                subject, step))
            return []

    elif (len(found_rects) == 0):
        found_rects = edt.brute_edge_finder(stick_dir, filename)
        try:
            centers = edt.cluster_rects(found_rects, img_stick)
        except:
            print("Clustering impossible for subject:  {}, Step: {}".format(
                subject, step))
            return []
    else:
        print("Not enough rectangles")
        testfields = []
        return testfields

    # edt.plot_rect(found_rects, centers, img_stick)

    # # Check if Box is detected
    # stat = tkinter.messagebox.askyesno(
    #     title='None', message='Picture detected correctly')

    # new_json = dict()
    # new_json["rectangles"] = found_rects
    # if not stat:
    #     new_json["found"] = False
    #     testfields = []

    # else:
    #     new_json["found"] = True

    # Define the width of the testfields
    field_width = width*0.04
    field_height = height*0.6
    testfields = []

    # Separate testfields (skip ID mark)
    for i in range(1, 11):
        left_boundary = centers[i][0] - field_width/2
        right_boundary = centers[i][0] + field_width/2
        upper_boundary = centers[i][1] - field_height/2
        lower_boundary = centers[i][1] + field_height/2
        field = img_stick[int(upper_boundary):int(
            lower_boundary), int(left_boundary):int(right_boundary)]
        testfields.append(field)

    # full_filename = os.getcwd() + "/results/contour/" + \
    #     os.path.splitext(dest_file[0])[0] + ".json"
    # with open(full_filename, 'w') as fp:
    #     json.dump(new_json, fp, indent=4, cls=NumpyEncoder)
    # fp.close()

    return testfields


def separate_testfields_old(img_stick):
    """
    Separates the 10 testfields of the Urine Stick into single image parts

    Input:
         img_stick: image of the detected urine stick
    Output:
        testfields: List with the separated image parts, starting with glucose, ending with leucocytes
     """

    # Calculate x coordinates of testfields
    # try:
    height, width, _ = img_stick.shape
    try:
        x_coord = find_testfields(img_stick, 50)
    except:
        print("Optimizer not working")
        return 0

    n = 1

    try:
        while (x_coord[-1]-x_coord[0] < 0.5*width):
            x_coord = find_testfields(img_stick, 50+5*n)
            n += 1
            if n == 15:
                break
    except:
        print("An error during minimizing occurred")

    # Define the width of the testfields
    field_width = width*0.05
    testfields = []

    # Separate testfields
    for i in range(0, 10):
        left_boundary = x_coord[i]
        right_boundary = x_coord[i] + field_width
        field = img_stick[:, int(left_boundary):int(right_boundary)]
        testfields.append(field)
    return testfields


def slice_stick(image):
    """
    This function slices the image of the urine stick into 100 slices

    Input:
        image: image of the detected urine stick
    Output:
        testfields: List with the separated image slices
    """
    row, col, _ = image.shape

    # Start position of the Urine Stick
    start_pos = 0

    # Slice the Urine Stick, only the middle 30%
    length_pos = 1
    slices = []
    for i in range(0, 100):
        slices.append(
            image[int(row*0.2):int(row*0.8), int((i/100)*col):int(((i+length_pos)/100)*col)])
    return slices


def find_testfields(img_stick, start_value):
    """
    DEPRECATED: Finds the positions of the testfields of the urine stick

    Input:
        img_stick: Stick image
        start_value: start point for the analysis of the hue hist

    Output:
        x_pos: x-positions of the found test fields

    """
    stick_sliced = slice_stick(img_stick)
    hue_values = []
    ctr = 0

    x = np.arange(0, 180, 1)
    y = np.arange(0, len(stick_sliced), 1)
    X, Y = np.meshgrid(x, y)
    Z = X * 0
    Z2 = X * 0
    Z3 = X * 0
    maxvals = [y, []]

    # Iterate through every slice
    for segment in stick_sliced:
        # Convert Color froom BGR to HSV
        img_hsv = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)

        # Calculate the Histogram of the Hue values
        hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 256])

        hue_values.append(hist.argmax())

        Z[ctr, :] = hist[:, 0]
        hist = cv2.calcHist([img_hsv], [1], None, [180], [0, 256])
        Z2[ctr, :] = hist[:, 0]
        maxvals[1].append(max(hist[25:, 0]))
        hist = cv2.calcHist([img_hsv], [2], None, [180], [0, 256])
        Z3[ctr, :] = hist[:, 0]
        # Plot histogram
        # ax.plot(range(0, 180), hist, np.multiply(np.ones(180), ctr), color='r')
        ctr += 1

    maxvals[1] = np.array(maxvals[1])
    maxvals[0] = np.append(maxvals[0], 1000)
    maxvals[1] = np.append(maxvals[1], 0)

    # find good start values
    tmp = np.max(Z2[:, start_value:], 1)
    loc_max = np.max(tmp[0:35])
    cntr = 0
    while loc_max == 0:
        loc_max = np.max(tmp[0:35+cntr])
        cntr += 1
    idx = np.where(tmp == loc_max)
    idx = np.min(idx)
    x0 = [float(idx), 6.1111*float(idx)/30]

    x1 = scipy.optimize.minimize(fun_min, x0, args=maxvals,
                                 method='Nelder-Mead', options={'gtol': 1e-6, 'disp': True})
    f = interp1d(maxvals[0], maxvals[1])

    # Found postions of testfields in percent
    x_pos = np.linspace(x1.x[0], min(x1.x[0]+x1.x[1]*9, 99), 10)

    # print("Found positions in percent: {}".format(x_pos))

    ynew = f(x_pos)
    # plt.figure()
    # plt.plot(maxvals[0], maxvals[1])
    # plt.plot(x_pos, ynew, 'bo')
    # plt.xlim([0, 110])

    row, col, _ = img_stick.shape

    x_pos = x_pos * (col/100)
    ynew = x_pos*0 + row*0.5
    fig1 = plt.figure()
    plt.imshow(img_stick)
    plt.plot(x_pos, ynew, 'bo')
    plt.title("Stick with found positions")

    # print(x1)
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    # ax.set_title("Hue")
    # ax.set_xlabel("H Channel")
    # ax.set_ylabel("Position")
    # ax.set_zlabel("Distribution")

    # fig3 = plt.figure()
    # ax1 = fig3.add_subplot(111, projection='3d')
    # ax1.plot_surface(X, Y, Z2, cmap=cm.coolwarm)
    # ax1.set_title("Saturation")
    # ax1.set_xlabel("S Channel")
    # ax1.set_ylabel("Position")
    # ax1.set_zlabel("Distribution")

    # fig4 = plt.figure()
    # ax2 = fig4.add_subplot(111, projection='3d')
    # ax2.plot_surface(X, Y,  Z3, cmap=cm.coolwarm)
    # ax2.set_title("Value")
    # ax2.set_xlabel("V Channel")
    # ax2.set_ylabel("Position")
    # ax2.set_zlabel("Distribution")

    #print("Found positions for testfields: {}".format(x_pos))

    plt.show()
    return x_pos


def fun_min(x, args):

    f = interp1d(args[0], args[1])
    xnew = np.linspace(x[0], x[0]+x[1]*9, 10)
    xnew = xnew[[0, 3, 4, 6, 7]]
    ynew = f(xnew)
    val = -np.sum(ynew)

    return val
