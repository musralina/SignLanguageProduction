import numpy as np
import os
import math
import time
import mediapipe as mp
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import csv
import warnings
warnings.filterwarnings("ignore")


model_name = "musralina/helsinki-opus-de-en-fine-tuned-wmt16-finetuned-src-to-trg"
# notebook_login()


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Pre-trained model: {model_name} is downloaded from the Huggingface platform')

for i in range(3):
    src_text = input("Enter the text: ")
    input_ids = tokenizer.encode(src_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print('From the given text the Glosses are as follows: ', decoded)
    input_glosses = decoded.split(' ')

    phoenix_dataset_dev = '/Users/alua/Desktop/PHOENIX-2014-T.dev.corpus.csv'
    phoenix_dataset_test = '/Users/alua/Desktop/PHOENIX-2014-T.test.corpus.csv'
    phoenix_dataset_train = '/Users/alua/Desktop/PHOENIX-2014-T.train.corpus.csv'
    # Load the Phoenix dataset
    phoenix_dev = pd.read_csv(phoenix_dataset_dev, sep="|")
    phoenix_test = pd.read_csv(phoenix_dataset_test, sep="|")
    phoenix_train = pd.read_csv(phoenix_dataset_train, sep="|")

    dev_img_path = '/Volumes/My Passport for Mac/2_Master2/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev'
    test_img_path = '/Volumes/My Passport for Mac/2_Master2/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test'
    train_img_path = '/Volumes/My Passport for Mac/2_Master2/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train'


    csv_file = 'img_number.csv'
    all_img = []

    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip the header row if present
        next(reader)
        
        # Read each row of the CSV file
        for row in reader:
            # Access the values in each row
            img_dict = {
                row[0]:row[1]
            }
            all_img.append(img_dict)

    train_gloss_lookup = []

    for gloss in input_glosses:
        for i in range(len(phoenix_train.orth)):
            sentence = [gloss for gloss in phoenix_train.orth[i].split(' ')]
            if str(gloss) in sentence:
                train_gloss_lookup.append({phoenix_train.name[i]:sentence})
                break

    dev_gloss_lookup = []
    for gloss in input_glosses:
        for i in range(len(phoenix_dev.orth)):
            sentence = [gloss for gloss in phoenix_dev.orth[i].split(' ')]
            if str(gloss) in sentence:
                dev_gloss_lookup.append({phoenix_dev.name[i]:sentence})

    test_gloss_lookup = []

    for gloss in input_glosses:
        for i in range(len(phoenix_test.orth)):
            sentence = [gloss for gloss in phoenix_test.orth[i].split(' ')]
            if str(gloss) in sentence:
                test_gloss_lookup.append({phoenix_test.name[i]:sentence})
                break

    gloss_to_draw_dev = []
    def get_gloss_to_draw(kp):
        gloss_to_draw_dev = []
        for gloss in (set(input_glosses)):
            for dict1 in kp:
                for k, v in dict1.items():   
                    for i in range(len(v)):
                        if gloss==v[i]:
                            gloss_to_draw_dev.append({gloss : [k, i]})
                            break
            
        
        # print(gloss_to_draw_dev)
        return gloss_to_draw_dev
    gloss_to_draw_train = get_gloss_to_draw(train_gloss_lookup)
    gloss_to_draw_dev = get_gloss_to_draw(dev_gloss_lookup)
    gloss_to_draw_test = get_gloss_to_draw(test_gloss_lookup)
    gloss_all = [gloss_to_draw_dev, gloss_to_draw_train, gloss_to_draw_test]

    def create_unique_list_of_dicts(lists):
        unique_dict = {}
        keys = []
        for list1 in lists:
            for dictionary in list1:
                for key, value in dictionary.items():
                    if key not in keys:
                        unique_dict[key] = value
                        keys.append(key)
        return unique_dict

    combined_list = gloss_all
    result = create_unique_list_of_dicts(combined_list)

    result_sorted = {}
    for g in input_glosses:
        for gloss, value in result.items():
            if g == gloss:
                result_sorted[gloss] = value
            else:
                continue

    result_sorted_all = {}
    for k, v in result_sorted.items():
        for dict_dev in all_img:
            for k1, v1 in  dict_dev.items():
                if str(v[0]) == str(k1):
                    result_sorted_all[k] = [v[0], v[1], v1]
                    break

                  
    # Holistic model
    mp_holistic = mp.solutions.holistic
    # Drawing utilities
    mp_drawing = mp.solutions.drawing_utils 

    path_train = "/Volumes/My Passport for Mac/2_Master2/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train/"
    path_test = "/Volumes/My Passport for Mac/2_Master2/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/"
    path_dev = "/Volumes/My Passport for Mac/2_Master2/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev/"

    csv_file = 'folders.csv'

    # Initialize an empty dictionary
    data = {}

    # Read the data from the CSV file
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        current_list = None
        current_values = []
        for row in reader:
            if len(row) > 0:
                if row[0].startswith('folders_'):
                    current_list = row[0]
                    current_values = []
                else:
                    current_values.append(row[0])
                    data[current_list] = current_values

    folders_train = data['folders_train']
    folders_test = data['folders_test']
    folders_dev = data['folders_dev']

    def mediapipe_detection(image, model):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

    def draw_styled_landmarks(image, results):
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def get_keypoints(img):
    # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            frame = img
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
        return image, results

    def draw_keypoints_example(img):
        image, results = get_keypoints(img)
        # Draw landmarks
        draw_styled_landmarks(image, results)
        image_height, image_width = 500, 400
        image2 = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        draw_landmarks(image2, results)
        cv.imshow(" ",image2)

    f_train = {}
    f_test = {}
    f_dev = {}



    for f_tr in folders_train:
        f_train[f_tr] = f_tr.split('/')
    for f_t in folders_test:
        f_test[f_t] = f_t.split('/')
    for f_d in folders_dev:
        f_dev[f_d] = f_d.split('/')
    
    for k1, v1 in result_sorted_all.items():
        train_stat = True
        test_stat = True
        for path, file in f_dev.items():
            if v1[0] in file:
                images = [image for image in os.walk(path)]
                framesNumber = int(v1[2])
                glossNumber = v1[1] + 1
                fr = int(framesNumber/glossNumber)

                for i, v in enumerate(images[0][2]):

                    if i > (v1[1] * fr - 4) and (i < v1[1] * fr + 8):
                        img_path = str(path_dev+'/'+ v1[0] +'/'+str(v))
                        # print(img_path)
                        img = cv.imread(img_path)
                        # print(v)
                        if img is not None:
                            draw_keypoints_example(img)
                        cv.waitKey(120)
                break
            for path, file in f_train.items():
                if (v1[0] in file) & train_stat == True:
                    images = [image for image in os.walk(path)]
                    framesNumber = int(v1[2])
                    glossNumber = v1[1] + 1
                    fr = int(framesNumber/glossNumber)
                    for j, v in enumerate(images[0][2]):
                        if j > (v1[1] * fr - 4) and (j < v1[1] * fr + 8):

                            img_path = str(path_train +'/'+ v1[0] +'/'+str(v))
                            img = cv.imread(img_path)
                            #  print('train', v)
                            if img is not None:
                                draw_keypoints_example(img)
                            cv.waitKey(120)
                    train_stat = False
                    break
                for path, file in f_test.items():
                    if (v1[0] in file) & test_stat == True:
                        images = [image for image in os.walk(path)]
                        framesNumber = int(v1[2])
                        glossNumber = v1[1] + 1
                        fr = int(framesNumber/glossNumber)
                        for i, v in enumerate(images[0][2]):
                            if i > (v1[1] * fr - 4) and (i < v1[1] * fr + 8):

                                img_path = str(path_test +'/'+ v1[0] +'/'+str(v))
                                img = cv.imread(img_path)
                                # print('train', v)
                                if img is not None:
                                    draw_keypoints_example(img)
                                cv.waitKey(120)
                        test_stat = False
                        break
                    break






