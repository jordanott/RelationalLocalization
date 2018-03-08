from keras.preprocessing.image import load_img
from visualize import visualize_qa
import progressbar
import h5py
import json
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
'wall': 13,
'window': 14}

questions = {1:30,2:31,3:32,4:33,5:34,6:35,7:36,8:37}

TARGET_IMG_SIZE = (400,400,3)
QUESTION_OFFSET = len(desired_objects)

def get_qa():
    # car,chair,door,leaves,light,person,plate,pole,shirt,sign,table,train,tree,trees,wall,window,yes, no,
    answer_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # objects (0-15), subject (16-31), question (32-39)
    question_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return question_array,answer_array

data= json.load(open('objects.json'))

dataset_size = len(data)
count = 0
question_count = 0
image_count = 0
f = h5py.File('data.hy', 'w')
id_file = open('id.txt', 'w')

# progress bar
bar = progressbar.ProgressBar(maxval=100,
                              widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                       progressbar.Percentage()])
bar.start()

for img_data in data:
    # setting ID of image
    image_id = img_data['image_id']
    # data for image
    object_coordinates = {}
    # key: name, value: frequency
    objects_freq = {}
    for obj in img_data['objects']:
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

    # print 'prev',prev_h,prev_w
    scipy.misc.imsave('images/'+str(image_id)+'.jpg', img)
    #import pprint
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(objects_freq)
    for obj in objects_freq:
        # resizing coordinates
        for i in range(len(object_coordinates[obj])):
            x,y,h,w = object_coordinates[obj][i]
            rescaled_x = (x * TARGET_IMG_SIZE[0] ) / prev_w
            rescaled_y = (y * TARGET_IMG_SIZE[0] ) / prev_h
            rescaled_w = (w * TARGET_IMG_SIZE[0] ) / prev_w
            rescaled_h = (h * TARGET_IMG_SIZE[0] ) / prev_h

            object_coordinates[obj][i] = [rescaled_x,rescaled_y,rescaled_h,rescaled_w]
    for obj in objects_freq:
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
            #visualize_qa(1,obj,object_coordinates[obj][0],img,obj)
            # question 2
            q,a = get_qa()
            x,y,_,_ = object_coordinates[obj][0]
            q[desired_objects[obj]] = 1
            q[desired_objects[obj]+QUESTION_OFFSET] = 1
            q[questions[2]] = 1
            if x < 200:
                a[-2] = 1
                #visualize_qa(2,'yes',object_coordinates[obj][0],img,obj)
            else:
                a[-1] = 1
                #visualize_qa(2,'no',object_coordinates[obj][0],img,obj)
            question_answer['questions'].append(q)
            question_answer['answers'].append(a)
            question_answer['locations'].append(object_coordinates[obj][0])

            obj_coords = np.array(object_coordinates[obj][0])[:2]
            question_7_8 = []
            max_dist_obj = None
            min_dist_obj = None
            max_dist_coords = None
            min_dist_coords = None
            max_dist = -1
            min_dist = 10000
            for obj_2 in objects_freq:
                # question 7 and 8
                if obj_2 != obj and objects_freq[obj_2] > 1:
                    obj_2_coords = np.array(object_coordinates[obj_2])
                    distances = np.linalg.norm(obj_coords - obj_2_coords[:,:2], axis=1)
                    max_idx = np.argmax(distances)
                    min_idx = np.argmin(distances)
                    # question 7
                    q,a = get_qa()
                    q[desired_objects[obj]] = 1
                    q[desired_objects[obj_2]] = 1
                    q[desired_objects[obj]+QUESTION_OFFSET] = 1
                    q[questions[7]] = 1

                    a[desired_objects[obj_2]] = 1
                    question_answer['questions'].append(q)
                    question_answer['answers'].append(a)
                    question_answer['locations'].append(obj_2_coords[min_idx].tolist())
                    #visualize_qa(7,obj_2,obj_2_coords[min_idx].tolist(),img,obj_2,obj)

                    # question 8
                    q,a = get_qa()
                    q[desired_objects[obj]] = 1
                    q[desired_objects[obj_2]] = 1
                    q[desired_objects[obj]+QUESTION_OFFSET] = 1
                    q[questions[8]] = 1

                    a[desired_objects[obj_2]] = 1
                    question_answer['questions'].append(q)
                    question_answer['answers'].append(a)
                    question_answer['locations'].append(obj_2_coords[max_idx].tolist())
                    #visualize_qa(8,obj_2,obj_2_coords[max_idx].tolist(),img,obj_2,obj)
                # question 5 and 6
                elif obj_2 != obj:
                    obj_2_coords = np.array(object_coordinates[obj_2])
                    distances = np.linalg.norm(obj_coords - obj_2_coords[:,:2], axis=1)
                    max_idx = np.argmax(distances)
                    min_idx = np.argmin(distances)
                    if distances[max_idx] > max_dist:
                        max_dist = distances[max_idx]
                        max_dist_obj = obj_2
                        max_dist_coords = obj_2_coords[max_idx].tolist()
                    if distances[min_idx] < min_dist:
                        min_dist = distances[min_idx]
                        min_dist_obj = obj_2
                        min_dist_coords = obj_2_coords[min_idx].tolist()

            # question 5
            if min_dist_obj:
                q,a = get_qa()
                q[desired_objects[obj]] = 1
                q[desired_objects[obj]+QUESTION_OFFSET] = 1
                q[questions[5]] = 1

                a[desired_objects[min_dist_obj]] = 1
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(min_dist_coords)
                #visualize_qa(5,min_dist_obj,min_dist_coords,img,obj)
            # question 6
            if max_dist_obj:
                q,a = get_qa()
                q[desired_objects[obj]] = 1
                q[desired_objects[obj]+QUESTION_OFFSET] = 1
                q[questions[6]] = 1

                a[desired_objects[max_dist_obj]] = 1
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(max_dist_coords)
                #visualize_qa(6,max_dist_obj,max_dist_coords,img,obj)

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
                    #visualize_qa(3,'yes',object_coordinates[obj][0],img,obj,obj_2)
                else:
                    a[-1] = 1
                    #visualize_qa(3,'no',object_coordinates[obj][0],img,obj,obj_2)
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(object_coordinates[obj][0])

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
                    middle_obj = obj_idx[obj_sort[1]]
                    right_obj = obj_idx[obj_sort[2]]

                    q[desired_objects[left_obj]] = 1
                    q[desired_objects[right_obj]] = 1
                    q[desired_objects[middle_obj]+QUESTION_OFFSET] = 1
                    q[questions[4]] = 1

                    a[desired_objects[middle_obj]] = 1
                    question_answer['questions'].append(q)
                    question_answer['answers'].append(a)
                    question_answer['locations'].append(object_coordinates[middle_obj][0])
                    #visualize_qa(4,middle_obj,object_coordinates[middle_obj][0],img,left_obj,right_obj)
    if len(question_answer['questions']):
        image_count += 1
        question_count += len(question_answer['questions'])
        id = '{}'.format(image_id)
        grp = f.create_group(id)
        id_file.write(id+'\n')
        #grp['image'] = I
        grp['question'] = question_answer['questions']
        grp['answer'] = question_answer['answers']
        grp['location'] = question_answer['locations']

    count += 1
    if count % (dataset_size / 100) == 0:
        bar.update(count / (dataset_size / 100))
    if count >= dataset_size:
        bar.finish()
        f.close()
        id_file.close()

print 'Images:',image_count,'Questions: ',question_count
