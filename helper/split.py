import json
import os
import random

import numpy as np
from numpyencoder import NumpyEncoder
from sklearn.model_selection import KFold, train_test_split

#######################################################################################
# Helper script, that splits the dataset into 3 training, testing and validation sets #
#######################################################################################


def find_annotations(list_filenames, all_annotations):
    """
    Function that generates list of annotations according to the given filenames

    Input:
        list_filenames: List of filenames, that shall be found in all_annotations
        all_annotations: List of dictionaries, containing annotations for all images
    """
    found_annotations = []
    for filename in list_filenames:
        annot = next(
            item for item in all_annotations if item["filename"] == filename)
        found_annotations.append(annot)
    return found_annotations


# Directory of the dataset
dataset_dir = "../All_imgs"

# Import all Annotations
annotations = json.load(
    open("./annotations/00_all_annotations.json"))

# Convert to list, dict keys are not necessary
# annotations=list(annotations.values())

count_labelled = 0
for a in annotations:
    if a['regions']:
        count_labelled += 1
    else:
        print("Unlabelled: ", a["filename"])
print("Labelled Images: ", count_labelled)
# Sort out images, that are not labelled.
annotations = [a for a in annotations if a['regions']]


# Generate list of all filenames
filenames = []
for i in range(len(annotations)):
    filenames.append(annotations[i]["filename"])

# sort list of filenames
filenames_sorted = sorted(filenames)

img_study = filenames_sorted[:158]

# Move single images to back, to sort subject-wise
img_study.append(img_study.pop(img_study.index(
    '1605368808626-16053687969671489409661.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1605368883340-16053688596131861194117.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1605368909089-1605368898417-13225481.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1605368963241-1605368953900-1538427202.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1605369055926-1605369042130-18521828.jpg')))
img_study.append(img_study.pop(img_study.index('1604572813636-IMG_0038.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604347536258-2020-11-0221.04.527950054788212704818.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604347691476-2020-11-0221.07.584019032657385811007.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604347893130-2020-11-0221.11.076903911206488238889.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604348016461-2020-11-0221.13.194761313437091000228.jpg')))
img_study.append(img_study.pop(img_study.index('1604650211649-image.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604657909309-16046578641335239298983456179214.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604827230965-1604827216916888892255.jpg')))
img_study.append(img_study.pop(img_study.index(
    '1604827171341-1604827152877-491031706.jpg')))
img_study.append(img_study.pop(img_study.index('1605347803572-image.jpg')))
img_study.append(img_study.pop(
    img_study.index('1605347826532-image.jpg')))
img_study.append(img_study.pop(
    img_study.index('1605347871643-image.jpg')))
img_study.append(img_study.pop(
    img_study.index('1605347937731-image.jpg')))


img_rest = filenames_sorted[158:]

img_controlurine = []
img_controlurine.append(img_rest[:5])
img_controlurine.append(img_rest[5:11])
img_controlurine.append(img_rest[11:19])
img_controlurine.append(img_rest[19:25])
img_controlurine.append(img_rest[25:32])
img_controlurine.append(img_rest[32:37])
img_controlurine.append(img_rest[37:43])
img_controlurine.append(img_rest[43:48])
img_controlurine.append(img_rest[48:53])

img_controlurine.append(img_rest[66:81])
img_controlurine.append(img_rest[81:86])
img_controlurine.append(img_rest[86:92])
img_controlurine.append(img_rest[92:98])
img_controlurine.append(img_rest[104:119])
img_controlurine.append(img_rest[138:147])
img_controlurine.append(img_rest[147:152])
img_controlurine.append(img_rest[152:162])
img_controlurine.append(img_rest[162:168])
img_controlurine.append(img_rest[168:174])
img_controlurine.append(img_rest[174:180])
joined_imgs = [img_rest[189]] + img_rest[180:184]
img_controlurine.append(joined_imgs)
img_controlurine.append(img_rest[184:189])
img_controlurine.append([img_rest[190]])
random.shuffle(img_controlurine)

annot_cu = []
for i in range(len(img_controlurine)):
    annot_temp_cu = find_annotations(img_controlurine[i], annotations)
    annot_cu = annot_cu + annot_temp_cu

# Split for 3-fold cross validation
kf = KFold(n_splits=3, shuffle=False)

k = 0
for train_index, test_index in kf.split(annot_cu):
    if(k == 0):
        cu_train_val_k1 = [annot_cu[i] for i in train_index]
        cu_test_k1 = [annot_cu[i] for i in test_index]
        k += 1
    elif(k == 1):
        cu_train_val_k2 = [annot_cu[i] for i in train_index]
        cu_test_k2 = [annot_cu[i] for i in test_index]
        k += 1
    elif (k == 2):
        cu_train_val_k3 = [annot_cu[i] for i in train_index]
        cu_test_k3 = [annot_cu[i] for i in test_index]

# Bundle subject images, shuffle randomly and divide into 2/3 and 1/3
index = 0
subjects = []
for i in range(0, len(img_study), 5):
    subjects.append(img_study[i:i+5])
    index += 1
random.shuffle(subjects)

# Find annotations of subjects:
annot_subjects = []
for i in range(len(subjects)):
    annot_temp = find_annotations(subjects[i], annotations)
    annot_subjects = annot_subjects + annot_temp

k = 0
for train_index, test_index in kf.split(annot_subjects):
    if(k == 0):
        subject_train_val_k1 = [annot_subjects[i] for i in train_index]
        subject_test_k1 = [annot_subjects[i] for i in test_index]
        k += 1
    elif(k == 1):
        subject_train_val_k2 = [annot_subjects[i] for i in train_index]
        subject_test_k2 = [annot_subjects[i] for i in test_index]
        k += 1
    elif (k == 2):
        subject_train_val_k3 = [annot_subjects[i] for i in train_index]
        subject_test_k3 = [annot_subjects[i] for i in test_index]


# Summarize random images
img_random = []
img_random = img_random + img_rest[98:104]
img_random = img_random + img_rest[53:62]
img_random = img_random + img_rest[62:66]
img_random = img_random + img_rest[119:138]

# Generate list of annotations of images not generated within the at-home study, nor in the control urine study
annot_rest = find_annotations(img_random, annotations)

# Split into training testing and validation sets
# train_random_k1, test_random_k1 = train_test_split(
#     annot_rest, test_size=0.33, random_state=42)
# train_random_k1, val_random_k1 = train_test_split(
#     train_random_k1, test_size=0.2, random_state=42)

# Split for 3-fold cross validation
kf_random = KFold(n_splits=3, random_state=42, shuffle=True)

k = 0
for train_index, test_index in kf_random.split(annot_rest):
    if(k == 0):
        train_val_random_k1 = [annot_rest[i] for i in train_index]
        test_random_k1 = [annot_rest[i] for i in test_index]
        k += 1
    elif(k == 1):
        train_val_random_k2 = [annot_rest[i] for i in train_index]
        test_random_k2 = [annot_rest[i] for i in test_index]
        k += 1
    elif (k == 2):
        train_val_random_k3 = [annot_rest[i] for i in train_index]
        test_random_k3 = [annot_rest[i] for i in test_index]

# Split all training sets into train and validation set
train_val_border_cu = round(len(cu_train_val_k1)*0.8)
train_val_border_subject = round(len(subject_train_val_k1)*0.8)
train_val_border_random = round(len(train_val_random_k1)*0.8)

train_val_border_cu_k2 = round(len(cu_train_val_k2)*0.8)
train_val_border_subject_k2 = round(len(subject_train_val_k2)*0.8)
train_val_border_random_k2 = round(len(train_val_random_k2)*0.8)

train_val_border_cu_k3 = round(len(cu_train_val_k3)*0.8)
train_val_border_subject_k3 = round(len(subject_train_val_k3)*0.8)
train_val_border_random_k3 = round(len(train_val_random_k3)*0.8)

# Control urine training sets
cu_train_k1 = cu_train_val_k1[:train_val_border_cu]
cu_train_k2 = cu_train_val_k2[:train_val_border_cu_k2]
cu_train_k3 = cu_train_val_k3[:train_val_border_cu_k3]

# Subjects training sets
subject_train_k1 = subject_train_val_k1[:train_val_border_subject]
subject_train_k2 = subject_train_val_k2[:train_val_border_subject_k2]
subject_train_k3 = subject_train_val_k3[:train_val_border_subject_k3]

# Random images training sets
random_train_k1 = train_val_random_k1[:train_val_border_random]
random_train_k2 = train_val_random_k2[:train_val_border_random_k2]
random_train_k3 = train_val_random_k3[:train_val_border_random_k3]

# Control urine validation sets
cu_val_k1 = cu_train_val_k1[train_val_border_cu:]
cu_val_k2 = cu_train_val_k2[train_val_border_cu_k2:]
cu_val_k3 = cu_train_val_k3[train_val_border_cu_k3:]

# Subjects validation sets
subject_val_k1 = subject_train_val_k1[train_val_border_subject:]
subject_val_k2 = subject_train_val_k2[train_val_border_subject_k2:]
subject_val_k3 = subject_train_val_k3[train_val_border_subject_k3:]

# Random images training sets
random_val_k1 = train_val_random_k1[train_val_border_random:]
random_val_k2 = train_val_random_k2[train_val_border_random_k2:]
random_val_k3 = train_val_random_k3[train_val_border_random_k3:]

# Final lists of annotations for traing, validation and testset
training_set_k1 = subject_train_k1 + cu_train_k1 + random_train_k1
validation_set_k1 = subject_val_k1 + cu_val_k1 + random_val_k1
test_set_k1 = subject_test_k1 + cu_test_k1 + test_random_k1

training_set_k2 = subject_train_k2 + cu_train_k2 + random_train_k2
validation_set_k2 = subject_val_k2 + cu_val_k2 + random_val_k2
test_set_k2 = subject_test_k2 + cu_test_k2 + test_random_k2

training_set_k3 = subject_train_k3 + cu_train_k3 + random_train_k3
validation_set_k3 = subject_val_k3 + cu_val_k3 + random_val_k3
test_set_k3 = subject_test_k3 + cu_test_k3 + test_random_k3


# Check length of sets:
print("1: Trainingset: {}, Testset: {}, Validationset: {}".format(
    len(training_set_k1), len(test_set_k1), len(validation_set_k1)))
print("2: Trainingset: {}, Testset: {}, Validationset: {}".format(
    len(training_set_k2), len(test_set_k2), len(validation_set_k2)))
print("3: Trainingset: {}, Testset: {}, Validationset: {}".format(
    len(training_set_k3), len(test_set_k3), len(validation_set_k3)))

# filename_test = "./00_test_k1.json"
# with open(filename_test, 'w') as fp:
#     json.dump(test_set_k1, fp, indent=4, cls=NumpyEncoder)
# fp.close()

# filename_train = "./00_train_k1.json"
# with open(filename_train, 'w') as fp:
#     json.dump(training_set_k1, fp, indent=4, cls=NumpyEncoder)
# fp.close()

# filename_val = "./00_val_k1.json"
# with open(filename_val, 'w') as fp:
#     json.dump(validation_set_k1, fp, indent=4, cls=NumpyEncoder)
# fp.close()


filename_test = "./00_test_k2.json"
with open(filename_test, 'w') as fp:
    json.dump(test_set_k2, fp, indent=4, cls=NumpyEncoder)
fp.close()

filename_train = "./00_train_k2.json"
with open(filename_train, 'w') as fp:
    json.dump(training_set_k2, fp, indent=4, cls=NumpyEncoder)
fp.close()

filename_val = "./00_val_k2.json"
with open(filename_val, 'w') as fp:
    json.dump(validation_set_k2, fp, indent=4, cls=NumpyEncoder)
fp.close()

filename_test = "./00_test_k3.json"
with open(filename_test, 'w') as fp:
    json.dump(test_set_k3, fp, indent=4, cls=NumpyEncoder)
fp.close()


filename_train = "./00_train_k3.json"
with open(filename_train, 'w') as fp:
    json.dump(training_set_k3, fp, indent=4, cls=NumpyEncoder)
fp.close()

filename_val = "./00_val_k3.json"
with open(filename_val, 'w') as fp:
    json.dump(validation_set_k3, fp, indent=4, cls=NumpyEncoder)
fp.close()
