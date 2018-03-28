RESIZE_IMAGES = True
QUESTIONS = [True,True,False,False,True,True,False,False]
TARGET_IMG_SIZE = (300,300,3)
DATASET_NAME = 'data.hy'

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

answers = {
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
'window': 14,
'yes':15,
'no':16
}

obj_look_up = {
6 : 'plate',
2 : 'door',
13 : 'wall',
9 : 'sign',
7 : 'pole',
11 : 'train',
10 : 'table',
1 : 'chair',
8 : 'shirt',
4 : 'light',
3 : 'leaves',
12 : 'tree',
5 : 'person',
14 : 'window',
0 : 'car',
}

ans_look_up = {
6 : 'plate' ,
2 : 'door' ,
8 : 'shirt' ,
16 : 'no' ,
13 : 'wall' ,
0 : 'car' ,
3 : 'leaves' ,
12 : 'tree' ,
14 : 'window' ,
9 : 'sign' ,
5 : 'person' ,
7 : 'pole' ,
11 : 'train' ,
4 : 'light' ,
10 : 'table' ,
1 : 'chair' ,
15 : 'yes' ,
}
questions = {1:30,2:31,3:32,4:33,5:34,6:35,7:36,8:37}
num_each_question = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}

QUESTION_OFFSET = len(desired_objects)

def get_qa():
    # car,chair,door,leaves,light,person,plate,pole,shirt,sign,table,train,tree,wall,window,yes, no,
    answer_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # objects (0-14), subject (15-29), question (30-37)
    question_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return question_array,answer_array
