import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG_COLOR = (180, 180, 150)
COLOR = [
    (0, 0, 210),
    (0, 210, 0),
    (210, 0, 0),
    (150, 150, 0),
    (150, 0, 150),
    (0, 150, 150),
    # add more colors here if needed
]

N_GRID = 4
NUM_COLOR = len(COLOR)
# the number of objects presented in each image
NUM_SHAPE = 4
# avoid a color shared by more than one objects
NUM_SHAPE = min(NUM_SHAPE, NUM_COLOR)
NUM_Q = 5


def color2str(code):
    return {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'yellow',
        4: 'magenta',
        5: 'cyan',
    }[code]


def question2str(qv,i):

    def q_type(q):
        return {
            0: 'is it a circle or a rectangle?',
            1: 'is it closer to the bottom of the image?',
            2: 'is it on the left of the image?',
            3: 'the color of the nearest object?',
            4: 'the color of the farthest object?'
        }[q]
    color = np.argmax(qv[i][:NUM_COLOR])
    q_num = np.argmax(qv[i][NUM_COLOR:])

    return '[Query object color: {}] [Query: {}]'.format(color2str(color),
                                                         q_type(q_num))


def answer2str(av,i,l, prefix=None):

    def a_type(a):
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            6: 'circle',
            7: 'rectangle',
            8: 'yes',
            9: 'no',
        }[np.argmax(a[i])]
    if not prefix:
        return '[Answer: {} @ ({},{})]'.format(a_type(av),l[i][0],l[i][1])
    else:
        return '[{} Answer: {}]'.format(prefix, a_type(av))


def visualize_iqa(img, q, a, l):
    #fig = plt.figure()
    for i in range(NUM_SHAPE*NUM_Q):
        plt.imshow(img)
        plt.title(question2str(q,i))
        plt.xlabel(answer2str(a,i,l))
        plt.show()
    #return fig
