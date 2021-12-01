import cv2
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from tkinter import *
import os


def select_image():
    """
    Function to select a folder, where the results will be stored in 

    Output: 
        dirname: Path to the image directory
        original_image: Selected image

    """
    Tk().withdraw()
    selected_image = filedialog.askopenfile()
    original_image = cv2.imread(selected_image.name)
    dirname = os.path.split(selected_image.name)[0]
    print("All files are saved into: {}".format(dirname))

    return dirname, original_image



def select_folder():
    """
    Function to select a directory

    Output:
        dirname: Path to the selected directory
    """

    Tk().withdraw()
    dirname = filedialog.askdirectory()
    print("Selected directory: {}".format(dirname))
    return dirname
