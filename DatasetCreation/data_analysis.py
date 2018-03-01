import json
from collections import OrderedDict

def object_frequency_counts(data):
    objects = {}
    objects_freq = {}
    for img in data:
        img_obj_names = []
        img_id = img['image_id']
        for obj in img['objects']:
            img_obj_names.append(obj['names'][0])

            # frequency count of objects
            if obj['names'][0] in objects_freq:
                objects_freq[obj['names'][0]] += 1
            else:
                objects_freq[obj['names'][0]] = 1
    d_descending = OrderedDict(sorted(objects_freq.items(), key=lambda kv: kv[1], reverse=True))

    f = open('objects.txt','w')
    for key in d_descending.keys():
        f.write(key + ': ' + str(d_descending[key]) + '\n')
    f.close()

def qustion_distribution(data):
    total_occurences = {
    'man':[0,0,0,0,0],
    'person':[0,0,0,0,0],
    'woman':[0,0,0,0,0],
    'people':[0,0,0,0,0],
    'window':[0,0,0,0,0],
    'tree':[0,0,0,0,0],
    'shirt':[0,0,0,0,0],
    'wall':[0,0,0,0,0],
    'sign':[0,0,0,0,0],
    'table':[0,0,0,0,0],
    'pole':[0,0,0,0,0],
    'light':[0,0,0,0,0],
    'car':[0,0,0,0,0],
    'trees':[0,0,0,0,0],
    'plate':[0,0,0,0,0],
    'leaves':[0,0,0,0,0],
    'door':[0,0,0,0,0],
    'train':[0,0,0,0,0],
    'chair':[0,0,0,0,0]}

    data = json.load(open('objects.json'))

    for img in data:
        objects = {}
        objects_freq = {}
        for obj in img['objects']:
            # frequency count of objects
            if obj['names'][0] in total_occurences:
                if obj['names'][0] in objects_freq:
                    objects_freq[obj['names'][0]] += 1
                else:
                    objects_freq[obj['names'][0]] = 1

        one_prior = False
        two_or_more_distinct = False
        two_of_something = False
        one_of_something = False
        # non relational questions -- check if an object appears once
        for obj in objects_freq:
            if objects_freq[obj] == 1:
                total_occurences[obj][0] += 1
                total_occurences[obj][1] += 1

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

    f = open('questions.txt','w')
    for key in total_occurences.keys():
        f.write(key + ': ' + str(total_occurences[key]) + '\n')
    f.close()

data = json.load(open('objects.json'))
