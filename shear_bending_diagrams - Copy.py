import numpy as np
import matplotlib.pyplot as plt

'''Checks to see if the entire train is on the bridge'''
def check_load(x, span, weight):
    P = weight/6
    pos = np.array([x+52,x+228,x+392,x+568,x+732,x+908])
    pos = pos[pos >= 0][pos <= span]
    return [P, pos]

def reaction(P, span, points):
    # rx1 is a pin, rx2 is a roller
    # gets the sum of the moments of the point loads then divides by the lever-arm of rx2
    rx2 = (np.sum(points*P))/span
    # Uses vertical force equilibrium (F_y = 0) to find rx1
    rx1 = P*np.prod(points.shape) - rx2
    return [rx1, rx2]


'''Sets up the bridge, the values can be replaced with inputs'''
def setup():
    # length of bridge
    span = 1200
    
    # A train is not a point load, it's distributed over a distance
    distrib = 960

    # Was the position of the front of the train. But I added 960 to make it the end of train
    x = int(input("what is your point of interest? "))

    if -960 >= x or x >= span:
        x = input('position must be within -960 and '+str(span))
    
    weight = 400

    
    # Applied load from train
    load = check_load(x, span, weight)
    rxs = reaction(load[0], span, load[1])
    return {'span':span,'load': load[0],'poi': x,'points': load[1], 'distrib': distrib, 'weight': weight, 'rx1': rxs[0], 'rx2': rxs[1]}

def sfd(points, rx1, rx2):
    V_val = np.zeros([points.shape[0]+3])
    V_val[2] = rx1
    V_val[2:-1] = np.cumsum(V_val[2:-1]-P)
    V_val[0:2] = [0,rx1]
    V_val[-1] = V_val[-2]+ rx2
    return V_val

def bmd(points, P, rx1, rx2, span):
    forces = np.append(rx1, -1*np.append(np.full(points.shape,P), rx2))
    lever_arm = np.append(0, np.append(points, span))
    lever_arm_matrix = np.array([lever_arm]*forces.size)
    
    for i in range(len(lever_arm_matrix)):
        lever_arm_matrix[i] = lever_arm_matrix[i] - lever_arm[i]
        lever_arm_matrix[i][lever_arm_matrix[i] < 0] = 0
    M_val = np.matmul(forces,lever_arm_matrix)
    return lever_arm,M_val

def plot(x,y,x_label,y_label,title,step,invert):
    if step == True:
        plt.step(x,y, where='post')
    else:
        plt.plot(x,y)
    if invert == True:
        plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def envelope(distrib, span, weight, poi):
    train_pos = np.linspace(-52,span-960+52,500)
    bridge_pos= np.linspace(0,span,500)
    M = np.zeros(bridge_pos.shape)
    V = np.zeros(bridge_pos.shape)
    for x in train_pos:
        # Note that the load may not be the complete weight of the train if x < 0 or x > span
        load, points = check_load(x, span, weight)
        rx1, rx2 = reaction(load, span, points)
        V_val = sfd(points, rx1, rx2)
        lever_arm, M_val = bmd(points, load, rx1, rx2, span)
        # plot(lever_arm, M_val, 'Lever Arm', 'Moment', 'Moment Diagram', False, True)
        # Actually for each x value
        
        for i in range(len(bridge_pos)):

            new_V = V_val[np.append(0,lever_arm) <= bridge_pos[i]][-1]
            new_M = np.interp(bridge_pos[i], lever_arm, M_val)
            
            if abs(M[i]) < abs(new_M):
                M[i] = new_M
            if abs(V[i]) < abs(new_V):
                V[i] = abs(new_V)
        # plot(np.append(0,lever_arm),V_val,'Distance','Shear Force','Shear Force Diagram',True, False)
    plot(bridge_pos, M, 'Lever Arm', 'Moment', 'Moment Envelope', False, True)

    print("Max Bending Moment",np.interp(600,bridge_pos,M,))
    plot(bridge_pos, V, 'Lever Arm', 'Shear Force', 'Shear Force Envelope', False, False)
    print("Max Shear Force",np.interp(0,bridge_pos,V,))

    return
def optimize_dim(span):
    girder_b_b = np.arange(75,150,5)
    girder_h_b = 1.27*np.arange(1,5,1)

    girder_b_w = 1.27*np.arange(1,3,1)
    girder_h_w = np.arange(50,200,5)

    girder_b_p = np.arange(5,36,5)
    girder_h_p = 1.27*np.arange(1,3,1)
    
    girder_b_t = np.arange(85,150,5)
    girder_h_t = 1.27*np.arange(1,3,1)
    # girder_dim = [[b_b, h_b,1],[b_w, h_w,2],[b_p, h_p,2],[b_t, h_t,1]]

    # pi_b_w = 1.27
    # pi_h_w = np.arange(0,200,5)

    # pi_b_p = np.arange(0,36,5)
    # pi_h_p = 1.27
    
    # pi_b_t = np.arange(0,200,5)
    # pi_h_t = 1.27
    # pi_dim = [[pi_b_w, pi_h_w,2],[pi_b_p, pi_h_p,2],[pi_b_t, pi_h_t,1]]

    max_FOS = 0
    max_area = 813*1016
    opt_dim = []
    opt_diaphragm_area = 0
    opt_diaphragm_spacing = 0
    opt_total_area = 0
    opt_num_diaphragms = 0
    for bottom_flange_width in girder_b_b:
        for bottom_flange_height in girder_h_b:
            for web_width in girder_b_w:
                for web_height in girder_h_w:
                    for plate_width in girder_b_p:
                        for plate_height in girder_h_p:
                            for top_flange_width in girder_b_t:
                                for top_flange_height in girder_h_t:
                                    dim = dim_analysis([[bottom_flange_width, bottom_flange_height,1],[web_width, web_height,2],[plate_width, plate_height,2],[top_flange_width, top_flange_height,1]])
                                    y_bar = centroidal_axis(dim)
                                    E = 4000
                                    mu = 0.2
                                    M = 68834.67616595917
                                    I = second_moment_area(dim, y_bar)
                                    V = 257.10354041416167
                                    total_area = np.sum(dim[:,0][(dim[:,0] != 1.27*np.arange(1,5,1))]*(dim[:,1][(dim[:,0] != 1.27*np.arange(1,5,1))]/1.27)*dim[:,2][(dim[:,0] != 1.27*np.arange(1,5,1))]) + np.sum(dim[:,1][(dim[:,0] == 1.27*np.arange(1,5,1))]*dim[:,2][(dim[:,0] == 1.27*np.arange(1,5,1))])
                                    total_area = total_area * 1260
                                    test_diaphragm_area = (dim[0,0]-dim[1,0]*dim[1,2])*(dim[1,1]+5)
                                    num_diaphragms = ((max_area-total_area)//test_diaphragm_area)
                                    if  total_area + test_diaphragm_area*8 > max_area :
                                        break
                                    # if num_diaphragms < 4:
                                    #     break
                                    test_diaphragm_spacing = span/7
                                    test_FOS = FOS_check(M, V, dim, I, E, mu, y_bar, test_diaphragm_spacing)[1]
                                    if test_FOS > max_FOS:
                                        max_FOS = test_FOS
                                        opt_dim = dim
                                        opt_diaphragm_area = test_diaphragm_area
                                        opt_diaphragm_spacing = test_diaphragm_spacing
                                        opt_total_area = total_area
                                        opt_num_diaphragms = num_diaphragms
    return max_FOS, opt_dim, opt_diaphragm_area, opt_diaphragm_spacing, opt_total_area, opt_num_diaphragms

def pi_beam_dim():
    # b_w = float(input("What is the width of the web? "))
    # h_w = float(input("What is the height of the web? "))

    # b_p = float(input("What is the width of the backing plate? "))
    # h_p = float(input("What is the height of the backing plate? "))

    # b_t = float(input("What is the width of the top flange? "))
    # h_t = float(input("What is the height of the top flange? "))

    # b_w = 1.27
    # h_w = 73.73

    # b_p = 5
    # h_p = 1.27
    
    # b_t = 100
    # h_t = 1.27
    
    dim = [[b_w, h_w,2],[b_p, h_p,2],[b_t, h_t,1]]
    return dim

def box_girder_dim():
    # b_b = float(input("What is the width of the bottom flange? "))
    # h_b = float(input("What is the height of the bottom flange? "))

    # b_w = float(input("What is the width of the web? "))
    # h_w = float(input("What is the height of the web? "))

    # b_p = float(input("What is the width of the backing plate? "))
    # h_p = float(input("What is the height of the backing plate? "))

    # b_t = float(input("What is the width of the top flange? "))
    # h_t = float(input("What is the height of the top flange? "))

    b_b = 80
    h_b = 1.27

    b_w = 1.27
    h_w = 73.73

    b_p = 5
    h_p = 1.27
    
    b_t = 100
    h_t = 1.27

    dim = [[b_b, h_b,1],[b_w, h_w,2],[b_p, h_p,2],[b_t, h_t,1]]
    return dim

def dim_analysis(dim):
    dim_w_height = []
    # You could do this better using numpy
    for i in range(len(dim)):
        d_area = dim[i][0]*dim[i][1]*dim[i][2]
        d_height = 0
        # Wrong because some of the areas are at the same height
        for j in range(i):
            d_height += dim[j][1]
        d_height += dim[i][1]/2
        dim_w_height.append([dim[i][0],dim[i][1], dim[i][2], d_area, d_height])
    return np.array(dim_w_height)

def centroidal_axis(dim):
    sum = np.sum(dim[:,3]*dim[:,4])
    y_bar = (sum)/(np.sum(dim[:,3]))
    return y_bar

def second_moment_area(dim,y_bar):
    I = np.sum(dim[:,2]*(dim[:,0]*dim[:,1]**3)/12 + dim[:,3]*(dim[:,4]-y_bar)**2)
    return I

def flexural_failure(M, dim, I, y_bar, type):
    if type == "compression":
        max_comp_stress = 6
        y = dim[-1,4] + dim[-1,1]/2 - y_bar
        stress = M*y/I
        factor_of_safety = FOS(max_comp_stress, stress)
    if type == "tension":
        max_ten_stress = 30
        y = y_bar
        stress = M*y/I
        factor_of_safety = FOS(max_ten_stress, stress)
    return factor_of_safety, stress

def glue_shear_failure(V, I, dim, y_bar):
    # I don't think we will be gluing the bottom flage
    max_shear_stress = 2
    Q_1 = np.sum(dim[0,3]*dim[0,2]*(y_bar-dim[0,4]))
    # Check for optimized design!
    b_1 = dim[1,0]*dim[1,2]
    stress_1 = (V*Q_1)/(I*b_1)

    Q_2 = np.sum(dim[3,3]*dim[3,2]*(y_bar-dim[3,4]))
    b_2 = dim[1,0]*dim[1,2] + dim[2,0]*dim[2,2]
    stress_2 = (V*Q_2)/(I*b_2)

    return [FOS(max_shear_stress,stress_1),FOS(max_shear_stress,stress_2)], [stress_1,stress_2]
    # return FOS(max_shear_stress,stress_2), stress_2


def material_shear_failure(V, I, dim, y_bar):
    max_shear_stress = 4
    Q_1 = dim[1,0]*(y_bar - dim[0,1])*dim[1,2]*((y_bar - dim[0,1])/2)+dim[0,3]*(y_bar - dim[0,4])
    b_1 = dim[1,0]*dim[1,2]
    shear_stress = (V*Q_1)/(I*b_1)
    return FOS(max_shear_stress,shear_stress),shear_stress

def flexural_buckling_failure(E,mu,applied_comp_stress,dim,y_bar):
    # Case 1
    k_1 = 4
    t_1 = dim[3,1]
    b_1 = dim[0,0] - dim[1,0]
    case_1_stress = ((k_1*np.pi**2*E)/(12*(1-mu**2)))*((t_1/b_1)**2)
    case_1_FOS = FOS(case_1_stress,applied_comp_stress)

    # Case 2
    k_2 = 0.425
    t_2 = dim[3,1]
    b_2 = (dim[3,0] - dim[0,0] + dim[1,0])/2
    case_2_stress = ((k_2*np.pi**2*E)/(12*(1-mu**2)))*((t_2/b_2)**2)
    case_2_FOS = FOS(case_2_stress,applied_comp_stress)

    # Case 3
    k_3 = 6
    t_3 = dim[1,0]
    b_3 = dim[1,1] + dim[3,1]/2 + dim[0,1] + dim[2,1] - y_bar
    case_3_stress = ((k_3*np.pi**2*E)/(12*(1-mu**2)))*((t_3/b_3)**2)
    case_3_FOS = FOS(case_3_stress,applied_comp_stress)
    return [case_1_FOS,case_2_FOS,case_3_FOS],[case_1_stress,case_2_stress,case_3_stress]

def shear_buckling_failure(E,mu,applied_shear_stress,dim,a):
    # Case 4
    k = 5
    t = dim[1,0]
    b = dim[1,1] + dim[3,1]/2 + dim[0,1]/2
    case_4_stress = ((k*np.pi**2*E)/(12*(1-mu**2)))*((t/b)**2+(t/a)**2)
    case_4_FOS = FOS(case_4_stress,applied_shear_stress)
    return case_4_FOS,case_4_stress

def FOS(max_stress, applied):
    applied = max(abs(applied), 0.000001)
    return max_stress/applied

def FOS_check(M, V, dim, I, E, mu, y_bar, a,i):
    all_FOS = np.array([])
    comp = flexural_failure(M, dim, I, y_bar, "compression")
    all_FOS = np.append(all_FOS,comp[i])
    all_FOS = np.append(all_FOS,flexural_failure(M, dim, I, y_bar, "tension")[i])
    shear = material_shear_failure(V, I, dim, y_bar)
    all_FOS = np.append(all_FOS,shear[i])
    all_FOS = np.append(all_FOS,glue_shear_failure(V, I, dim, y_bar)[i])
    all_FOS = np.append(all_FOS,flexural_buckling_failure(E,mu,comp[int(not i)],dim,y_bar)[i])
    all_FOS = np.append(all_FOS,shear_buckling_failure(E,mu,shear[int(not i)],dim,a)[i])
    all_FOS = np.absolute(all_FOS)
    min_FOS = np.min(all_FOS)
    return all_FOS, min_FOS

'''
Create a calculation for y bar
Create a variable for each of the different lengths of the cross-sections to calculate I
Create a bending moment envelop
Restrict the amount of volume of matboard

'''
if __name__ == "__main__":
    bridge = setup()
    span = bridge['span']
    P = bridge['load']
    points = bridge['points']
    distrib = bridge['distrib']
    weight = bridge['weight']
    rx1 = bridge['rx1']
    rx2 = bridge['rx2']
    poi = bridge['poi']

    # V_val = sfd(points, rx1, rx2)

    # lever_arm,M_val = bmd(points, P, rx1, rx2, span)
    # Max calcs should return the FOS and applied stress
    dim = dim_analysis([[ 75.   ,   1.27 ,   1.],
       [ 1.27  , 133.46   ,   2.],
       [  5.   ,   1.27 ,   2.],
       [ 100.   ,   2.54 ,   1.]])
    print(dim)
    y_bar = centroidal_axis(dim)
    E = 4000
    mu = 0.2
    M = 68834.67616595917
    I = second_moment_area(dim, y_bar)
    V = 257.10354041416167
    # CHANGE DEPENDING ON ITERATION
    diaphragm_spacing = 300
    checks = ["Flexural Compression Failure", "Flexural Tension Failure", "Material Shear Failure", "Glue Shear Failure Bottom", "Glue Shear Failure Top", "Flexural Buckling Failure Case: 1", "Flexural Buckling Failure Case: 2", "Flexural Buckling Failure Case: 3", "Shear Buckling Failure Case 4"]
    factors = FOS_check(M, V, dim, I, E, mu, y_bar, diaphragm_spacing,0)
    flex = FOS_check(M, V, dim, I, E, mu, y_bar, diaphragm_spacing,1)
    print("y_bar",y_bar)
    print("Second Moment of Area",I)
    for i in range(len(factors[0])):
        print(checks[i],":",factors[0][i],"FOS",",",flex[0][i], "MPa")
    print("Minimum FOS:",factors[1])
    

    # print(optimize_dim(span))
    # plot(np.append(0,lever_arm),V_val,'Distance','Shear Force','Shear Force Diagram',True, False)
    # plot(lever_arm,M_val,'Distance','Bending Moment','Bending Moment Diagram',False, True)
    # envelope(distrib, span, weight,poi)