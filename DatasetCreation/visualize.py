import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

text_questions = {
    1:'Where is {oi}?',
    2:'Is {oi} on the left?',
    3:'Is {oi} on the left of {oj}?',
    4:'What object is in between {oi} and {oj}?',
    5:'What object is closest to {oi}?',
    6:'What object is farthest from {oi}?',
    7:'What {oi} is closest to {oj}?',
    8:'What {oi} is farthest from {oj}?'
}


def visualize_qa(q,a,location,img,oi,oj=None):
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    if oj:
        ax.set_title('Question '+text_questions[q].format(oi=oi,oj=oj))
    else:
        ax.set_title('Question '+text_questions[q].format(oi=oi))
    ax.set_xlabel('Answer: '+a+' Location: '+str(location))
    # Create a Rectangle patch ~ patches handles (w,h) not (h,w)
    rect = patches.Rectangle((location[0],location[1]),location[3],location[2],linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()
