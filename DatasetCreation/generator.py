from keras.preprocessing.image import load_img
from utils import *
from visualize import *
import progressbar
import h5py
import json
import numpy as np
import scipy.misc
import os

data= json.load(open('objects.json'))

dataset_size = len(data)
count = 0
question_count = 0
image_count = 0
f = h5py.File(DATASET_NAME, 'w')
id_file = open('id.txt', 'w')

# progress bar
bar = progressbar.ProgressBar(maxval=100,
                              widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                       progressbar.Percentage()])
bar.start()
#img_store = np.empty((1,400,400,3))
coords_store = np.empty((1,4))
total_image = np.zeros(TARGET_IMG_SIZE)

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
    # save resized image
    if RESIZE_IMAGES:
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
    for obj in objects_freq:
        if objects_freq[obj] == 1:
            if QUESTIONS[0]: # question 1
                q,a = get_qa()
                q[desired_objects[obj]] = 1
                q[desired_objects[obj]+QUESTION_OFFSET] = 1
                q[questions[1]] = 1
                # answer 1
                a[desired_objects[obj]] = 1
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(object_coordinates[obj][0])
                num_each_question[1] += 1
                #visualize_qa(1,obj,object_coordinates[obj][0],img,obj)
            if QUESTIONS[1]: # question 2
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
                num_each_question[2] += 1

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
                    if QUESTIONS[6]: # question 7
                        q,a = get_qa()
                        q[desired_objects[obj]] = 1
                        q[desired_objects[obj_2]] = 1
                        q[desired_objects[obj]+QUESTION_OFFSET] = 1
                        q[questions[7]] = 1

                        a[desired_objects[obj_2]] = 1
                        question_answer['questions'].append(q)
                        question_answer['answers'].append(a)
                        question_answer['locations'].append(obj_2_coords[min_idx].tolist())
                        num_each_question[7] += 1
                        #visualize_qa(7,obj_2,obj_2_coords[min_idx].tolist(),img,obj_2,obj)

                    if QUESTIONS[7]: # question 8
                        q,a = get_qa()
                        q[desired_objects[obj]] = 1
                        q[desired_objects[obj_2]] = 1
                        q[desired_objects[obj]+QUESTION_OFFSET] = 1
                        q[questions[8]] = 1

                        a[desired_objects[obj_2]] = 1
                        question_answer['questions'].append(q)
                        question_answer['answers'].append(a)
                        question_answer['locations'].append(obj_2_coords[max_idx].tolist())
                        num_each_question[8] += 1
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
            if QUESTIONS[4] and min_dist_obj:
                q,a = get_qa()
                q[desired_objects[obj]] = 1
                q[desired_objects[obj]+QUESTION_OFFSET] = 1
                q[questions[5]] = 1

                a[desired_objects[min_dist_obj]] = 1
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(min_dist_coords)
                num_each_question[5] += 1
                #visualize_qa(5,min_dist_obj,min_dist_coords,img,obj)
            # question 6
            if QUESTIONS[5] and max_dist_obj:
                q,a = get_qa()
                q[desired_objects[obj]] = 1
                q[desired_objects[obj]+QUESTION_OFFSET] = 1
                q[questions[6]] = 1

                a[desired_objects[max_dist_obj]] = 1
                question_answer['questions'].append(q)
                question_answer['answers'].append(a)
                question_answer['locations'].append(max_dist_coords)
                num_each_question[6] += 1
                #visualize_qa(6,max_dist_obj,max_dist_coords,img,obj)

        for obj_2 in objects_freq:
            if obj_2 == obj:
                continue
            # question 3
            if objects_freq[obj] == 1 and objects_freq[obj_2] == 1 and QUESTIONS[2]:
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
                num_each_question[3] += 1
            # question 4
            if QUESTIONS[3]:
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
                        num_each_question[4] += 1
                    #visualize_qa(4,middle_obj,object_coordinates[middle_obj][0],img,left_obj,right_obj)
    if len(question_answer['questions']):
        image_count += 1
        for i in range(len(question_answer['questions'])):
            id = '{}'.format(question_count)
            grp = f.create_group(id)
            id_file.write(id+'\n')
            grp['image'] = '{}.jpg'.format(image_id)
            grp['question'] = question_answer['questions'][i]
            grp['answer'] = question_answer['answers'][i]
            grp['location'] = question_answer['locations'][i]

            question_count += 1

            _coords = np.array(question_answer['locations'][i]).reshape(1,4)
            coords_store = np.append(coords_store,_coords,axis=0)
        #img_store = np.append(img_store,img.reshape(1,400,400,3),axis=0)
        total_image += img
    count += 1
    if count % (dataset_size / 100) == 0:
        bar.update(count / (dataset_size / 100))
    if count >= dataset_size:
        bar.finish()
        f.close()
        id_file.close()
        break

print 'Images:',image_count,'Questions: ',question_count

import pprint
pp = pprint.PrettyPrinter(indent=4)
# print frequency of each question
pp.pprint(num_each_question)

#img_store = np.delete(img_store,0,0)
mean = total_image/float(image_count) #np.mean(img_store,axis=(0,1,2))
#std = np.std(img_store,axis=(0,1,2))

coords_store = np.delete(coords_store,0,0)
c_mean = np.mean(coords_store,axis=0)
c_std = np.std(coords_store,axis=0)

np.savez('img_mean_coords_mean_std',img_mean=mean,coords_mean=c_mean,coords_std=c_std)

from keras.preprocessing.image import load_img
import numpy as np
import h5py

data = np.load('img_mean_coords_mean_std.npz')
img_mean, coords_mean, coords_std = data['img_mean'],data['coords_mean'],data['coords_std']

mean = np.sum(img_mean,axis=(0,1))/float(TARGET_IMG_SIZE[0]**2)
print 'Mean:',mean
store = np.zeros(3)
seen_img = {}
f = h5py.File(DATASET_NAME, 'r')
with open('id.txt','r') as ids:
	for line in ids:
		ID = line.strip()
		img_id = f[ID]['image'].value
		if img_id not in seen_img:
			seen_img[img_id] = 1
			img = np.array(load_img('images/'+img_id))
			diff = np.sum(img-mean,axis=(0,1))/float(TARGET_IMG_SIZE[0]**2)
			store += diff*diff

std = np.sqrt(store/float(len(store)-1))
print 'Std:',std
np.savez('mean_std',img_mean=mean,img_std=std,coords_mean=coords_mean,coords_std=coords_std)
