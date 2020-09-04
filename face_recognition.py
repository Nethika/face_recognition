
"""
Face Recognition
================

Inspired by :
http://dlib.net/face_recognition.py.html

This code uses dlib's face detector model, face landmark calculation model and dlib's face recognition tool. 

The face detector model detects the faces in a given image. 

The face landmark calculation model (shape predictor) calcultes 68 face landmarks for each face detected, so we can precisely localize the features (eyes, nose, mouth etc) of the face, and finally feed the landmarks (shape) into the face recognition model.

The face recognition tool maps
an image of a human face to a 128 dimensional vector space where images of
the same person are near to each other and images from different people are
far apart.  Therefore, you can perform face recognition by mapping faces to
the 128D space and then checking if their Euclidean distance is small
enough. 

When using a distance threshold of 0.6, the dlib model obtains an accuracy
of 99.38% on the standard LFW face recognition benchmark, which is
comparable to other state-of-the-art methods for face recognition as of
February 2017. This accuracy means that, when presented with a pair of face
images, the tool will correctly identify if the pair belongs to the same
person or is from different people 99.38% of the time.

With our testings, we found that a distance threshold of 0.58 will help in identifying the difference between siblings.

Saving face hashes (128 dimensional vector):
============================================

`faces_json.json` file save the hashes calculated for each face recognized. 
If the new face is similar to a previously uploaded face, a mean hash will be calculated and 
`faces_json.json` will be updated accordingly.

"""


import sys
import os
import dlib
import glob
from skimage import io
from imutils.face_utils import FaceAligner
import imutils
import numpy as np
from scipy.spatial import distance
import json
import string

# Load Models
## Face landmarks model:
predictor_path = "shape_predictor_68_face_landmarks.dat" 
sp = dlib.shape_predictor(predictor_path)
## Face recognition model(# calculates 128D vector (hash) for an image)
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
# model from dlib to detect faces:
detector = dlib.get_frontal_face_detector()
# to allign the face    
fa = FaceAligner(sp, desiredFaceWidth=512)     

# Image Location
user = "user1"
new_images_path = user

# Read new Images
new_images = glob.glob(os.path.join(new_images_path, "*.jpg"))

# write/read to/from json
json_file= os.path.join(user, 'faces_json.json')

# Match with new image
## In a loop:
#for new_image_path in new_images:
   ######################## 

## One at a time:
#new_image_path=os.path.join(user, 'maroon_bells.jpg')
new_image_path=new_images[0]

print("Processing file: {}".format(new_image_path))
img = io.imread(new_image_path)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))  
features =[]
for k, d in enumerate(dets):
    faceAligned = fa.align(img, img, d)
    dets2 = detector(faceAligned, 1)          # face detector model
    shape_new = sp(faceAligned, dets2[0])     # face landmarks model
    face_descriptor = facerec.compute_face_descriptor(faceAligned, shape_new)     # face recognition model
    features.append(face_descriptor)

features = np.array(features) 

# Read from json file
if os.path.exists(json_file):
    face_data = json.load(open(json_file))
else:
    face_data =[]

# Threshold set to identify different faces.  Might need tuning in case of siblings, sun glass pictures etc.  
threshold=0.58

#
for i in range(len(features)): #faces in new image
    inface_hash= features[i]
    match_dict={}
    print (i)
    for j in range(len(face_data)):
        face_id = face_data[j]['face_id']
        face_freq = face_data[j]['frequency']
        face_hash = json.loads(face_data[j]['hash'])
        dist = distance.euclidean(inface_hash,face_hash)
        print(face_id , face_freq, dist)
        if dist < threshold:
            match_dict[j] = dist
    if match_dict:
        indx = min(match_dict, key=match_dict.get)
        min_hash = json.loads(face_data[indx]['hash'])
        # find New Mean for hash
        new_mean = np.mean([inface_hash,min_hash],axis=0)
        #update hash
        face_data[indx]['hash'] = str(new_mean.tolist())
        #update frequency
        face_data[indx]['frequency'] += 1  
        print("Matched with:")
        print(face_data[indx]['face_id'])
    else:     #new face
        print("No match! -> New face:")
        face_id = str(len(face_data)+1).zfill(4)
        print(face_id)
        tempt_dict={'face_id': face_id, 'frequency': 1,'hash':str(inface_hash.tolist())}
        face_data.append(tempt_dict)
            
#update json file
with open(json_file, 'w') as facefile:
    json.dump(face_data, facefile)
    

######################## 