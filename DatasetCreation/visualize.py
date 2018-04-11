import matplotlib.pyplot as plt
import matplotlib.patches as patches

text_questions = {
    1:'Where is the {oi}?',
    2:'Is the {oi} on the left?',
    3:'Is the {oi} on the left of the {oj}?',
    4:'What object is in between the {oi} and the {oj}?',
    5:'What object is closest to the {oi}?',
    6:'What object is farthest from the {oi}?',
    7:'What {oi} is closest to the {oj}?',
    8:'What {oi} is farthest from the {oj}?'
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

def visualize_prediction(img,q,a,p_a,location,p_l,oi,oj=None,id=0):
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    if oj:
        ax.set_title('Question '+text_questions[q+1].format(oi=oi,oj=oj))
    else:
        ax.set_title('Question '+text_questions[q+1].format(oi=oi))
    print 'Question '+text_questions[q+1].format(oi=oi)
    print 'Answer: '+a+', Predicted: '+p_a+', Location: '+str(location) + ', P Location: '+str(p_l)

    ax.set_xlabel('Answer: '+a+', Predicted: '+p_a)
    # Create a Rectangle patch ~ patches handles (w,h) not (h,w)
    rect = patches.Rectangle((location[0],location[1]),location[3],location[2],linewidth=2,edgecolor='g',facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((p_l[0],p_l[1]),p_l[3],p_l[2],linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    plt.savefig(str(id))
    plt.clf()
