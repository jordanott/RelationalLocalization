import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def answer2str(av,i, prefix=None):

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
        return '[Answer: {} ]'.format(a_type(av))
    else:
        return '[{} Answer: {}]'.format(prefix, a_type(av))


def visualize_iqa(img, q, a, p_a, l, p_l, iou, id=0):
    #fig = plt.figure()
    #for i in range(NUM_SHAPE*NUM_Q):
    i = 0
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    ax.set_title(question2str(q,i))
    x_lab = answer2str(a,i) + answer2str(p_a,i,prefix='Predicted') + ' IoU:' + ' {:0.2f}'.format(iou)
    ax.set_xlabel(x_lab)
    #circle1 = plt.Circle((l[0][0], l[0][1]), 1, color='black')
    rect = patches.Rectangle((l[0],l[1]),l[2],l[3],linewidth=2,edgecolor='g',facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((p_l[0],p_l[1]),p_l[2],p_l[3],linewidth=2,edgecolor='r',facecolor='none')
    #circle2 = plt.Circle((p_l[0][0], p_l[0][1]), 3, color='white',fill=False)
    ax.add_patch(rect)

    plt.savefig('{}.png'.format(id))
    plt.clf()
    #return fig
