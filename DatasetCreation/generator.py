import h5py
import json
from keras.preprocessing.image import load_img
import numpy as np
import scipy.misc
import os

desired_objects = {
'car': 0,
'chair': 1,
'door': 2,
'leaves': 3,
'light': 4,
'person': 5,
'plate': 6,
'pole': 7,
'shirt': 8,
'sign': 9,
'table': 10,
'train': 11,
'tree': 12,
'trees': 13,
'wall': 14,
'window': 15}

questions = {1:32,2:33,3:34,4:35,5:36,6:37,7:38,8:39}

TARGET_IMG_SIZE = (400,400,3)
QUESTION_OFFSET = len(desired_objects)

def get_qa():
    # car,chair,door,leaves,light,person,plate,pole,shirt,sign,table,train,tree,trees,wall,window,yes, no,
    answer_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # objects (0-15), subject (16-31), question (32-39)
    question_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return question_array,answer_array

data= json.load(open('objects.json'))

for img in data:
    # setting ID of image
    image_id = img['image_id']
    # data for image
    object_coordinates = {}
    # key: name, value: frequency
    objects_freq = {}
    for obj in img['objects']:
        # combine man & women into person category
        if obj['names'][0] == 'man' or obj['names'][0] == 'woman':
            obj['names'][0] = 'person'
        # frequency count of objects
        if obj['names'][0] in desired_objects:
            # get bounding box coords
            x,y,h,w = obj['x'],obj['y'],obj['h'],obj['w']
            if obj['names'][0] in objects_freq:
                # increment frequency
                objects_freq[obj['names'][0]] += 1
                # append x,y,h,w
                object_coordinates[obj['names'][0]].append([x,y,h,w])
            else:
                # set initial count
                objects_freq[obj['names'][0]] = 1
                # set initial
                object_coordinates[obj['names'][0]] = [[x,y,h,w]]

    question_answer = {'questions':[],'answers':[],'locations':[]}

    # load img, get size of original
    complete_location = '../../VG_100K/'+str(image_id)+'.jpg'
    if os.path.isfile(complete_location):
        img = np.array(load_img(complete_location))
        prev_h,prev_w,_ = img.shape
        img = np.array(load_img(complete_location,target_size=TARGET_IMG_SIZE))
    else:
        complete_location = '../../VG_100K_2/'+str(image_id)+'.jpg'
        img = np.array(load_img(complete_location))
        prev_h,prev_w,_ = img.shape
        img = np.array(load_img(complete_location,target_size=TARGET_IMG_SIZE))

    print 'prev',prev_h,prev_w
    scipy.misc.imsave('images/'+str(image_id)+'.jpg', img)

    for obj in objects_freq:
        # resizing coordinates
        for i in range(len(object_coordinates[obj])):
            x,y,h,w = object_coordinates[obj][i]
            rescaled_x = (x * TARGET_IMG_SIZE[0] ) / prev_w
            rescaled_y = (y * TARGET_IMG_SIZE[0] ) / prev_h
            rescaled_w = (w * TARGET_IMG_SIZE[0] ) / prev_w
            rescaled_h = (h * TARGET_IMG_SIZE[0] ) / prev_h

            object_coordinates[obj][i] = [rescaled_x,rescaled_y,rescaled_h,rescaled_w]

        if objects_freq[obj] == 1:
            # question 1
            q,a = get_qa()
            q[desired_objects[obj]] = 1
            q[desired_objects[obj]+QUESTION_OFFSET] = 1
            q[questions[1]] = 1
            # answer 1
            a[desired_objects[obj]] = 1
            question_answer['questions'].append(q)
            question_answer['answers'].append(a)
            question_answer['locations'].append(object_coordinates[obj][0])

            # question 2
            q,a = get_qa()
            x,y,_,_ = object_coordinates[obj][0]
            q[desired_objects[obj]] = 1
            q[desired_objects[obj]+QUESTION_OFFSET] = 1
            q[questions[2]] = 1
            if x < 200:
                a[-2] = 1
            else:
                a[-1] = 1
            question_answer['questions'].append(q)
            question_answer['answers'].append(a)
            question_answer['locations'].append(object_coordinates[obj][0])
        for obj_2 in objects_freq:
            if obj_2 == obj:
                continue
            if objects_freq[obj] == 1 and objects_freq[obj_2] == 1:
                # question 3
                q,a = get_qa()
                q[desired_objects[obj]] = 1
                q[desired_objects[obj_2]] = 1
                q[desired_objects[obj]+QUESTION_OFFSET] = 1
                q[questions[3]] = 1

                obj_x = object_coordinates[obj][0][0]
                obj_2_x = object_coordinates[obj_2][0][0]
                if obj_x < obj_2_x:
                    a[-2] = 1
                else:
                    a[-1] = 1
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(object_coordinates[obj][0])
            # TODO: question 5 in here
            # iterate through all coordinates to compare distance
            # 

            # question 4
            q,a = get_qa()
            for obj_3 in objects_freq:
                if obj_3 == obj_2 or obj == obj_2 or obj == obj_3:
                    continue
                if objects_freq[obj] == 1 and objects_freq[obj_2] == 1 and objects_freq[obj_3] == 1:
                    obj_x = object_coordinates[obj][0][0]
                    obj_2_x = object_coordinates[obj_2][0][0]
                    obj_3_x = object_coordinates[obj_3][0][0]

                    obj_idx = {0:obj,1:obj_2,2:obj_3}
                    obj_sort = np.argsort([obj_x,obj_2_x,obj_3_x])
                    left_obj = obj_idx[obj_sort[0]]
                    middle_obj = left_obj = obj_idx[obj_sort[1]]
                    right_obj = obj_idx[obj_sort[2]]

                    q[desired_objects[left_obj]] = 1
                    q[desired_objects[right_obj]] = 1
                    q[desired_objects[middle_obj]+QUESTION_OFFSET] = 1
                    q[questions[4]] = 1

                    a[desired_objects[middle_obj]] = 1
                    question_answer['questions'].append(q)
                    question_answer['answers'].append(a)
                    question_answer['locations'].append(object_coordinates[middle_obj][0])



grp = f.create_group(id)
grp['image'] = I
grp['question'] = Q[j, :]
grp['answer'] = A[j, :]
grp['location'] = L[j, :]
