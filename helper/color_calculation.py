import cv2
import matplotlib.pyplot as plt
import math


def calculate_colors(colorfields, field_type):
    """
    Converts the image to HSV Color space and calculates the color

    Input:
        colorfields: List, containing the images of the separated testfields/reference fields
        field_type: Stick (1) or Reference Card (0)

    Output:
        hue_values: list of hue values of the images in the input list

    """
    hue_values = []
    if field_type == 0:
        for row in range(len(colorfields)):
            hue_ref = []
            for img in colorfields[row]:

                # Convert Color froom BGR to HSV
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Calculate the Histogram of the Hue values
                hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 179])
                hue_ref.append(hist.argmax())

                # Plot histogram
                # plt.plot(hist, color='b')
                # plt.xlim([0, 180])
                # plt.title("Histogram Hue Values")
                # plt.xlabel("H Channel")
                # plt.ylabel("Distribution")
                # plt.show()
            hue_values.append(hue_ref)
    else:
        for img in colorfields:
            # Convert Color froom BGR to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Calculate the Histogram of the Hue values
            hist = cv2.calcHist([img_hsv], [0], None, [180], [0, 179])
            hue_values.append(hist.argmax())
    return hue_values


def calculate_hsv(colorfields_ref, colorfields_stick):
    """
    Function to calculate the hsv values 

    Input:
        colorfields_ref: List, containing the images of the separated reference fields
        colorfields_stick: List, containing the images of the separated testfields

    Output:
        ref_hsv: List containing the calculated HSV values of the ref card
        stick_hsv: List containing the calculated HSV values of the stick 

    """
    ref_hsv = []
    for row in range(len(colorfields_ref)):
        colors_ref = []
        for img in colorfields_ref[row]:
            # Convert Color froom BGR to RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #
            # Calculate the color values
            hue = img[:, :, 0]
            sat = img[:, :, 1]
            value = img[:, :, 2]
            hist = cv2.calcHist([img], [0], None, [180], [0, 179])
            hist1 = cv2.calcHist([img], [1], None, [100], [0, 255])
            hist2 = cv2.calcHist([img], [2], None, [100], [0, 255])
            colors_ref.append([hist.argmax(), hist1.argmax(), hist2.argmax()])
        ref_hsv.append(colors_ref)
    stick_hsv = []
    for img in colorfields_stick:
        # Convert Color from BGR to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = img[:, :, 0]
        sat = img[:, :, 1]
        value = img[:, :, 2]
        # Calculate the Histogram of the Hue values
        hist = cv2.calcHist([img], [0], None, [180], [0, 179])
        hist1 = cv2.calcHist([img], [1], None, [100], [0, 255])
        hist2 = cv2.calcHist([img], [2], None, [100], [0, 255])

        stick_hsv.append([hist.argmax(), hist1.argmax(), hist2.argmax()])
    return ref_hsv, stick_hsv


def calc_color_coordinates(colorfields_ref, colorfields_stick):
    """
    Function to calculate the color coordinates of the point in hsv colorspace

    Input:
        colorfields_ref: List, containing the images of the separated reference fields
        colorfields_stick: List, containing the images of the separated testfields

    Output:
        ref_coord: contains all coordinates (cylinder coordinates transformed to kartesian) of the reference card
        stick_coord: contains all coordinates (cylinder coordinates transformed to kartesian) of the stick

    """
    ref_coord = []
    for row in range(len(colorfields_ref)):
        hue_ref = []
        for img in colorfields_ref[row]:
            # Convert Color froom BGR to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Calculate the Histogram of the Hue values
            hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 179])
            hist_s = cv2.calcHist([img_hsv], [1], None, [255], [0, 255])
            hist_v = cv2.calcHist([img_hsv], [2], None, [255], [0, 255])

            h = hist_h.argmax()
            s = hist_s.argmax()
            v = hist_v.argmax()

            x = s * math.cos(h/180*math.pi)
            y = s * math.sin(h/180*math.pi)
            z = v

            hue_ref.append([x, y, z])

            # Plot histogram
            # plt.plot(hist, color='b')
            # plt.xlim([0, 180])
            # plt.title("Histogram Hue Values")
            # plt.xlabel("H Channel")
            # plt.ylabel("Distribution")
            # plt.show()
        ref_coord.append(hue_ref)
    stick_coord = []
    for img in colorfields_stick:
        # Convert Color froom BGR to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Calculate the Histogram of the Hue values
        hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 179])
        hist_s = cv2.calcHist([img_hsv], [1], None, [255], [0, 255])
        hist_v = cv2.calcHist([img_hsv], [2], None, [255], [0, 255])

        h = hist_h.argmax()
        s = hist_s.argmax()
        v = hist_v.argmax()

        x = s * math.cos(h/180*math.pi)
        y = s * math.sin(h/180*math.pi)
        z = v

        stick_coord.append([x, y, z])

    return ref_coord, stick_coord
