import h5py
############################
#### NON RELATIONAL ########
############################

# (yes/no) is there x in the image?
    # need one object

# (yes/no) is it on the left?
    # one object

############################
#### RELATIONAL ############
############################
# (yes/no) Is x on the left of y?

# (obj) what obj is in between x?

# (obj) what x is closest to y?

# (obj) what obj is closest to y?

# (obj) what x is farthest from y?

# (obj) what obj is farthest from y?

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

QUESTION_OFFSET = len(desired_objects)
def get_qa():
    # car,chair,door,leaves,light,person,plate,pole,shirt,sign,table,train,tree,trees,wall,window,yes, no,
    answer_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # objects (1-16), subject (17-32), question (33-40)
    question_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return question_array,answer_array
data= json.load(open(objects.json))

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
                object_coordinates[obj['names']].append([x,y,h,w])
            else:
                # set initial count
                objects_freq[obj['names'][0]] = 1
                # set initial
                object_coordinates[obj['names']] = [[x,y,h,w]]

    question_answer = {'questions':[],'answers':[],'locations':[]}
    one_prior = False
    two_or_more_distinct = False
    two_of_something = False
    one_of_something = False

    for obj in objects_freq:
        if objects_freq[obj] == 1:
            # question 1
            q,a = get_qa()
            q[desired_objects[obj]] = 1
            q[desired_objects[obj]+QUESTION_OFFSET] = 1
            q[32] = 1
            # answer 1
            a[desired_objects[obj]] = 1
            question_answer['questions'].append(q)
            question_answer['answers'].append(a)
            question_answer['locations'].append(object_coordinates[obj])

            # question 2
            q,a = get_qa()
            x,y,_,_ = object_coordinates[obj]

            # TODO: get original size of image
            # scale to 400,400
            # save image
            
            if one_prior:
                total_occurences[obj][2] += 1
                two_or_more_distinct = True

            if two_or_more_distinct:
                total_occurences[obj][2] += 1

            one_prior = True
            one_of_something = True

        if objects_freq[obj] > 1:
            two_of_something = True

        if one_of_something and two_of_something:
            total_occurences[obj][3] += 1
            total_occurences[obj][4] += 1


grp = f.create_group(id)
grp['image'] = I
grp['question'] = Q[j, :]
grp['answer'] = A[j, :]
grp['location'] = L[j, :]
