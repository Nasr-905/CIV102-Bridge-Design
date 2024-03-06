import numpy as np
import matplotlib.pyplot as plt

'''Checks to see if the entire train is on the bridge'''
def check_load(x, distrib, span, weight):
    # Calculates the load from the train on the bridge
    P = min((x/distrib),((distrib + span - x)/distrib), 1)*weight

    # Where the train starts
    start = max(0, x - distrib)

    # Where the train ends
    end = min(span, x)

    return [P, start, end]

'''Sets up the bridge, the values can be replaced with inputs'''
def setup():
    # length of bridge
    span = 1200
    
    # A train is not a point load, it's distributed over a distance
    distrib = 960

    # Was the position of the front of the train. But I added 960 to make it the end of train
    x = int(input("what is the position of the train? ")) + 960

    if 0 > x or x - distrib > span:
        x = input('position must be within 0 and '+str(span+distrib))

    weight = 400

    # Applied load from train
    load = check_load(x, distrib, span, weight)
    
    return {'span':span,'load': load[0],'position': x,'distrib': distrib, 'start': load[1], 'end': load[2]}


if __name__ == "__main__":
    bridge = setup()
    span = bridge['span']
    P = bridge['load']
    # x = bridge['position']
    distrib = bridge['distrib']
    start = bridge['start']
    end = bridge['end']
    rx2 = P*(start + (end-start)/2)/1200
    rx1 = P - rx2


    x_val = np.array([0, 0, start, end, span, span])
    V_val = np.array([0, rx1, rx1, rx1 - P, rx1 - P, 0])

    # Optimize this later
    lin_x_val = np.linspace(0, span, 100)

    P_arr = np.full(lin_x_val.shape, P)
    load_arr = np.full(lin_x_val.shape,end-start)
    M_rxs = rx1*lin_x_val
    print((200/(480-0))*(300-0)*0.5*(300-0+0))
    M_P = -1*np.maximum((P/(load_arr))*np.minimum(lin_x_val-start,load_arr),0)*(0.5*np.minimum(lin_x_val-start,load_arr)+np.maximum(0,lin_x_val-end))
    M_val = M_rxs + M_P

    fig, axs = plt.subplots(2)
    axs[0].fill_between(x_val,0,V_val, alpha = 0.5)
    axs[0].set_title('Shear Force Diagram')
    axs[1].fill_between(lin_x_val,0,M_val, alpha = 0.5, facecolor='red')
    axs[1].set_title('Bending Moment Diagram')
    plt.gca().invert_yaxis()
    print(lin_x_val)
    print(M_P)
    plt.show()