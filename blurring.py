import numpy as np
import cv2
import ot
import math
import matplotlib.pyplot as plt
from ot.utils import *
import pandas as pd
import time

np.random.seed(57)

### GLOBAL VARIABLES
dim1 = 64
dim2 = 64
squares = []
for i in range(dim1):
     for j in range(dim2):
        squares.append((i,j))
cost_matrix = np.ones([dim1*dim2,dim1*dim2])
for (index_1, (i,j)) in enumerate(squares):
    for (index_2, (a,b)) in enumerate(squares):
        cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)



### Spiral Image
img_path = "/Users/rohanbuluswar/Desktop/CS_Research/Spiral.png"
img = cv2.imread(img_path, 0)
img = cv2.resize(img, (dim1,dim2))

reverse_img = np.ones([dim1,dim2])
for x in range(dim1):
    for y in range(dim2):
        reverse_img[x,y] = (255-img[x,y])

tot = np.sum(reverse_img)
init_img = np.ones([dim1,dim2])
for x in range(dim1):
    for y in range(dim2):
        init_img[x,y] = reverse_img[x,y]/tot



def show_sample(matrix, num):
    """
    Visualizes probability distribution by plotting large number of samples.
    Inputs:
        matrix - dim1 x dim2 array representing probability distribution
        num    - integer representing number of desired samples
    Returns: Nothing
    
    """

    probabilities = []
    counts = {}
    for i in range(dim1):
        for j in range(dim2):
            probabilities.append(matrix[i,j])
            counts[(i,j)] = 0

    sample = np.random.choice(len(squares), num, p=probabilities)
    for index, val in enumerate(sample):
        counts[squares[val]] += 1
    new_matrix = np.ones([dim1,dim2])
    m = max(counts.values())

    for i in range(dim1):
        for j in range(dim2):
            if counts[(i,j)] > 0:
                new_matrix[i,j] = 0
            else:
                new_matrix[i,j] = 255
    cv2.imshow("photo", new_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def show_empirical(counts):
    """
    Displays a collection of particles. 
    Input:
        counts - dictionary; keys are indices (i,j), values are # particles
    """
    matrix = np.ones([dim1,dim2])
    for i in range(dim1):
        for j in range(dim2):
            if counts[(i,j)]>0:
                matrix[i,j] = 0
            else:
                matrix[i,j] = 255
    cv2.imshow("photo", matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recover_diagonals(P, lam, M):
    """ 
    Used to recover the vectors u,v from the Sinkhorn algorithm for use in warmstart.
    Inputs:
        P        - dim1*dim2 x dim1*dim2 array, output of the Sinkhorn algorithm
        lam      - float, weighting parameter
        M        - dim1*dim2 x dim1*dim2 cost matrix 
    """
    A = np.ones([dim1*dim2,dim1*dim2])
    for i in range(dim1*dim2):
        for j in range(dim1*dim2):
            A[i,j] = P[i,j] / np.exp(-lam * M[i,j])
    U,S,V =  np.linalg.svd(A) 
    u = U[:,0]
    v = V[:,0]
    return (np.log(u),np.log(v))
    

def recover_diagonals_iter(P, lam, M, iter):
    """ 
    Used to recover the vectors u,v from the Sinkhorn algorithm for use in warmstart.
    Inputs:
        P        - dim1*dim2 x dim1*dim2 array, output of the Sinkhorn algorithm
        lam      - float, weighting parameter
        M        - dim1*dim2 x dim1*dim2 cost matrix 
        iter     - number of iterations performed
    """
    A = np.ones([dim1*dim2,dim1*dim2]) 
    for i in range(dim1*dim2):
        for j in range(dim1*dim2):
            A[i,j] = P[i,j] / np.exp(-lam * M[i,j])
    u = np.ones([dim1*dim2])
    v = np.ones([dim1*dim2])
    for _ in range(iter):
        u = A.dot(v) / np.linalg.norm(v)
        v = A.T.dot(u) / np.linalg.norm(u)
    return (np.array(np.log(u)),np.array(np.log(v)))


"""Performs one step of the SDE given a matrix of particle locations (dictionary) instead of a probability matrix. """
def general_SDE_from_sample(sample, init_matrix, dt, lam, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = None, u_initial = None, v_initial = None):
    """ 
    Performs one step of the SDE.
    Inputs:
        sample             - dictionary; keys are indices (i,j); values are # of particles
        init_matrix        - dim1 x dim2 array giving initial empirical distribution
        dt                 - time step used in discretizing SDE
        lam                - weighting parameter in original optimization problem
        method             - specifies if OT is standard, entropic, or unbalanced
        sinkhorn_reg       - regularization parameter in entropic OT
        reg_og, reg_sample - regularization parameters in unbalanced OT
        warm_start         - specifies if warmstart functionality of entropic OT is used
    Returns: 
        new_counts         - updated dictionary giving positions of particles
        vec_fields         - vector field displaying drift of SDE
    """
    
    ### Initializing list of squares, deriving probability matrix from sample. 
    ### Initializing current and initial probability distributions, and new counts dictionary.
    vec_field = np.zeros((dim1, dim2, 2))

    curr_prob = []
    prob_og = []
    new_counts = {}
    total = sum(sample.values())
    for i in range(dim1):
        for j in range(dim2):
            prob_og.append(init_matrix[i,j])
            new_counts[(i,j)] = 0
    for index, val in enumerate(squares):
        curr_prob.append(sample[val] / total)

    ### Initializing cost matrix and solving optimal transport
    
    if method == "entropic" and warm_start:
        T = ot.sinkhorn(np.array(curr_prob),np.array(prob_og),cost_matrix, sinkhorn_reg, warmstart = (u_initial, v_initial))
    elif method == "entropic":
        T = ot.sinkhorn(np.array(curr_prob),np.array(prob_og),cost_matrix, sinkhorn_reg)
    if method == "standard":
        T = ot.emd(curr_prob,prob_og,cost_matrix)
    if method == "unbalanced" and warm_start:
        T = ot.sinkhorn_unbalanced(curr_prob, prob_og, cost_matrix, reg = sinkhorn_reg, reg_m = (reg_sample,reg_og), warmstart = (u_initial, v_initial))
    elif method == "unbalanced":
        T = ot.sinkhorn_unbalanced(curr_prob, prob_og, cost_matrix, reg = sinkhorn_reg, reg_m = (reg_sample,reg_og))


    ### Moving each particle 
    for index, square in enumerate(squares):
        x_coords = 0
        y_coords = 0
        for _ in range(sample[square]):
            x_0 = square[0]
            y_0 = square[1]
            B_x = np.random.normal(0,dt)
            B_y = np.random.normal(0,dt)
            row = T[index]
            x_sum = 0
            y_sum = 0
            if row.sum() == 0:
                x_sum = x_0
                y_sum = y_0
            elif row.sum() > 0 : 
                for index, (i,j) in enumerate(squares):
                    x_sum += (row[index]*i)/row.sum()
                    y_sum += (row[index]*j)/row.sum()
            new_x = x_0 - dt*(2*x_0-2*x_sum)+lam*np.sqrt(2)*B_x
            new_y = y_0 - dt*(2*y_0-2*y_sum)+lam*np.sqrt(2)*B_y
            if new_x < 0:
                new_x = 0
            elif new_x >= dim1-0.5:
                new_x = dim1-1
                print("warning - x too large")
            else:
                new_x = np.round(new_x,0)
            if new_y < 0:
                new_y = 0
            elif new_y >= dim2-0.5:
                new_y = dim2-1
                print("warning - y too large")
            else:
                new_y = np.round(new_y,0)
            new_counts[(new_x,new_y)]+=1
            x_coords += - dt*(2*x_0-2*x_sum)
            y_coords += - dt*(2*y_0-2*y_sum)
        (i,j) = square
        if sample[square] > 0:
            vec_field[i,j,0] = x_coords/sample[square]
            vec_field[i,j,1] = y_coords/sample[square]
        else:
            vec_field[i,j,0] = 0
            vec_field[i,j,1] = 0
    return (new_counts, vec_field, T)



def general_simulate_SDE(init_sample, dt, lam, steps, N, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = False):
    """
    Simulates the SDE given a sample and the necessary hyperparameters.
    Inputs:
        init_sample        - dictionary; keys are indices (i,j); vals are # particles
        dt                 - float, time step used in discretizing SDE
        lam                - float, weighting parameter in original optimization problem
        steps              - integer, number of steps taken
        N                  - integer, number of copies of each particle taken at beginning
        method             - specifies if OT is standard, entropic, or unbalanced
        sinkhorn_reg       - float, regularization parameter in entropic OT
        reg_og, reg_sample - floats, regularization parameters in unbalanced OT
        warm_start         - specifies if warmstart functionality of entropic OT is used
    Returns: 
        new_matrix_dict    - final positions of particles - dictionary; keys are indices (i,j); vals are # particles
        vec_fields         - list of vector field displaying drift of SDE at each time
    """
    warm = warm_start

    vec_fields = []
    init_matrix_dict = {}
    for i in range(dim1):
        for j in range(dim2):
            init_matrix_dict[(i,j)] =init_sample[(i,j)] * N
    new_matrix_dict = init_matrix_dict
    prob_matrix = np.ones([dim1,dim2])
    total = sum(init_sample.values())
    for i in range(dim1):
        for j in range(dim2):
            prob_matrix[(i,j)] = init_sample[(i,j)] / total

    u = None
    v = None
    for iter in range(steps):
        print("We are in "+ str(iter))
        if iter > 0:
            updated_matrix, new_vec_field, T = general_SDE_from_sample(new_matrix_dict, prob_matrix, dt, lam, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample=reg_sample, u_initial = u, v_initial= v, warm_start = warm)
        if iter == 0:
            updated_matrix, new_vec_field, T = general_SDE_from_sample(new_matrix_dict, prob_matrix, dt, lam, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample=reg_sample, warm_start = False)
        new_matrix_dict = updated_matrix 
        vec_fields.append(new_vec_field)
        
        if warm:
            u,v = recover_diagonals(T, lam, cost_matrix)

    return new_matrix_dict, vec_fields



def general_test_SDE(prob_matrix, dt, lam, steps, N, m, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = False):
    """
    Tests the algorithm given an underlying distribution and the hyperparameters by taking sample, measuring OT decrease.
    Inputs:
        prob_matrix        - array representing underlying distribution
        dt                 - float, time step used in discretizing SDE
        lam                - float, weighting parameter in original optimization problem
        steps              - integer, number of steps taken
        N                  - integer, number of copies of each particle taken at beginning
        method             - specifies if OT is standard, entropic, or unbalanced
        sinkhorn_reg       - float, regularization parameter in entropic OT
        reg_og, reg_sample - floats, regularization parameters in unbalanced OT
        warm_start         - specifies if warmstart functionality of entropic OT is used
    Returns: 
        output_matrix      - dictionary; final positions of particles
        vec_fields         - list of vector field displaying drift of SDE at each time
    Prints/Displays:
        sample_photo       - image showing sample from underlying
        og_dist            - between sample and underlying
        new_dist           - between output empirical and underlying
        percentage change 
    """
    
    ### Take sample from prob_matrix
    probabilities = []
    counts = {}
    for i in range(dim1):
        for j in range(dim2):
            probabilities.append(prob_matrix[i,j])
            counts[(i,j)] = 0
    sample = np.random.choice(len(squares), m, p=probabilities)
    for index, val in enumerate(sample):
        counts[squares[val]] += 1
    
    ### Show our sample

    dummy_matrix =  np.empty((dim1, dim2,3), dtype=np.uint8)
    for i in range(dim1):
        for j in range(dim2):
            if counts[(i,j)] > 0:
                dummy_matrix[i,j] = [255,0,0]
            else:
                dummy_matrix[i,j] = [255,255,255]
    cv2.imshow("sample_photo", dummy_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    output_matrix, vec_fields = general_simulate_SDE(counts, dt, lam, steps, N, method = method, sinkhorn_reg=sinkhorn_reg, reg_og=reg_og, reg_sample = reg_sample)
    
    ### Turn original sample and output sample into probability distributions
    og_emp = []
    output_emp = []
    prob_list = []
    for i in range(dim1):
        for j in range(dim2):
            og_emp.append(counts[(i,j)] / m)  
            output_emp.append(output_matrix[(i,j)]/(N*m))
            prob_list.append(prob_matrix[i,j])
    
    og_dist = ot.emd2(prob_list, og_emp, cost_matrix)
    new_dist = ot.emd2(prob_list, output_emp, cost_matrix)
    change = (new_dist-og_dist) / og_dist
    print("Original OT distance: "+ str(og_dist))
    print("New OT distance: " +str(new_dist))
    print("percentage change = " + str(100*change))
    return output_matrix, vec_fields



def test_SDE_from_sample(sample, prob_matrix, dt, lam, steps, N, method, sinkhorn_reg = 1, reg_og = 1, reg_sample = 1, warm_start = False):
    """
    Tests the algorithm given an underlying distribution, a sample, and the hyperparameters.
    Inputs:
        sample             - dictionary; sample from the underlying distribution
        prob_matrix        - array representing underlying distribution
        dt                 - float, time step used in discretizing SDE
        lam                - float, weighting parameter in original optimization problem
        steps              - integer, number of steps taken
        N                  - integer, number of copies of each particle taken at beginning
        method             - specifies if OT is standard, entropic, or unbalanced
        sinkhorn_reg       - float, regularization parameter in entropic OT
        reg_og, reg_sample - floats, regularization parameters in unbalanced OT
        warm_start         - specifies if warmstart functionality of entropic OT is used
    Returns: 
        output_matrix      - dictionary; final positions of particles
        vec_fields         - list of vector field displaying drift of SDE at each time
    Prints/Displays:
        sample_photo       - image showing sample from underlying
        og_dist            - between sample and underlying
        new_dist           - between output empirical and underlying
        percentage change 
    """
    
    ### Initializing
    probabilities = []
    counts = sample
    init_matrix = np.ones([dim1,dim2])
    for i in range(dim1):
        for j in range(dim2):
            init_matrix[i,j] = counts[(i,j)]

    ### Simulation
    output_matrix, vec_fields = general_simulate_SDE(counts, dt, lam, steps, N, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample = reg_sample)
    
    ### Turn original sample and output sample into probability distributions
    og_emp = []
    output_emp = []
    prob_list = []
    for i in range(dim1):
        for j in range(dim2):
            og_emp.append(counts[(i,j)] / init_matrix.sum())  
            output_emp.append(output_matrix[(i,j)]/(N*init_matrix.sum()))
            prob_list.append(prob_matrix[i,j])
    
    og_dist = ot.emd2(prob_list, og_emp, cost_matrix)
    new_dist = ot.emd2(prob_list, output_emp, cost_matrix)
    change = (-new_dist+og_dist) / og_dist
    print("Original OT distance: "+ str(og_dist))
    print("New OT distance: " +str(new_dist))
    print("percentage decrease = " + str(100*change))
    return (100*change, output_matrix, vec_fields)



def plot_vector_field(field):
    """
    Plots and displays a given vector field:
    Inputs:
        field - an array of shape (dim1, dim2,2) representing a vector field 
    """
    X = []
    Y = []
    U = []
    V = []
    for i in range(dim1):
        for j in range(dim2):
            if field[i,j,0] != 0 or field[i,j,1] !=0:
                X.append(- dim1/2 + i)
                Y.append(dim2/2 - j)
                U.append(field[i,j,0])
                V.append(-field[i,j,1])
    plt.quiver(X, Y, U, V, color='b', units='xy', scale=1) 
    plt.xlim(-65, 65) 
    plt.ylim(-65, 65) 
    plt.grid() 
    plt.show()


def generate_sample(size,init_sample, dt, lam, steps, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = False):
    """
    Tests the algorithm given an underlying distribution and the hyperparameters by taking sample, measuring OT decrease.
    Inputs:
        size               - size of the desired sample
        init_sample        - dictionary representing given empirical distribution
        dt                 - float, time step used in discretizing SDE
        lam                - float, weighting parameter in original optimization problem
        steps              - integer, number of steps taken
        N                  - integer, number of copies of each particle taken at beginning
        method             - specifies if OT is standard, entropic, or unbalanced
        sinkhorn_reg       - float, regularization parameter in entropic OT
        reg_og, reg_sample - floats, regularization parameters in unbalanced OT
        warm_start         - specifies if warmstart functionality of entropic OT is used
    Returns: 
        output_matrix      - dictionary; final positions of sampled points
    """
    
    ### Initially sampling from white noise
    probabilities = []
    counts = {}
    for i in range(dim1):
        for j in range(dim2):
            probabilities.append(noise[i,j])
            counts[(i,j)] = 0
    sample = np.random.choice(len(squares), size, p=probabilities)
    for index, val in enumerate(sample):
        counts[squares[val]] += 1
    
    warm = warm_start


    new_matrix_dict = counts
    prob_matrix = np.ones([dim1,dim2])
    total = sum(init_sample.values())
    for i in range(dim1):
        for j in range(dim2):
            prob_matrix[(i,j)] = init_sample[(i,j)] / total
    u = None
    v = None
    for iter in range(steps):
        print("We are in "+ str(iter))
        if iter > 0:
            updated_matrix, _, T = general_SDE_from_sample(new_matrix_dict, prob_matrix, dt, lam, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample=reg_sample, u_initial = u, v_initial= v, warm_start = warm)
        if iter == 0:
            updated_matrix, _, T = general_SDE_from_sample(new_matrix_dict, prob_matrix, dt, lam, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample=reg_sample, warm_start = False)
        new_matrix_dict = updated_matrix 
        
        if warm:
            u,v = recover_diagonals_iter(T, lam, cost_matrix, 15)

    return new_matrix_dict

    


### CREATING EXAMPLE DISTRIBUTIONS


### Generate Gaussian
mean = [35, 40]
var = [25,15]
gaussian = np.ones([dim1,dim2])
for i in range(dim1):
    for j in range(dim2):
        exponent = -0.5 * (((i-mean[0])**2 / var[0])+((j-mean[1])**2 / var[1]))
        gaussian[i,j] = math.exp(exponent)
total = gaussian.sum()
for i in range(dim1):
    for j in range(dim2):
        gaussian[i,j] = gaussian[i,j] / total


### White noise as Gaussian
mean = [dim1/2, dim2/2]
var = [40,40]
noise = np.ones([dim1,dim2])
for i in range(dim1):
    for j in range(dim2):
        exponent = -0.5 * (((i-mean[0])**2 / var[0])+((j-mean[1])**2 / var[1]))
        noise[i,j] = math.exp(exponent)
total = noise.sum()
for i in range(dim1):
    for j in range(dim2):
        noise[i,j] = noise[i,j] / total

### Generate Mixed Gaussian
mean_1 = [20,20]     
mean_2 = [50,50]       
var_1 = [15,20]       
var_2 = [20,15]        
weight_1 = 0.4            
mixed = np.zeros([dim1,dim2])
for i in range(dim1):
    for j in range(dim2):
        exp_1 =  -0.5 * (((i-mean_1[0])**2 / var_1[0])+((j-mean_1[1])**2 / var_1[1]))
        mixed[i,j] += math.exp(exp_1)
        exp_2 =  -0.5 * (((i-mean_2[0])**2 / var_2[0])+((j-mean_2[1])**2 / var_2[1]))
        mixed[i,j] += math.exp(exp_2)
        
mixed_total = mixed.sum()

for i in range(dim1):
    for j in range(dim2):
        mixed[i,j] = mixed[i,j] / mixed_total
        

### Distribution on Circle
center = [30,30]
rad_lower = 11
rad_upper = 14
circular = np.ones([dim1,dim2])
for i in range(dim1):
    for j in range(dim2):
        if rad_lower**2 < (center[0]-i)**2+(center[1]-j)**2 < rad_upper**2:
            circular[i,j] = 1
        else: 
            circular[i,j] = 0
circ_total = circular.sum()
for i in range(dim1):
    for j in range(dim2):
        circular[i,j] = circular[i,j]/circ_total


### Uniform Distribution
uniform = np.ones([dim1,dim2])
for i in range(dim1):
    for j in range(dim2):
        uniform[i,j] = uniform[i,j] / (dim1*dim2)



#### TESTING Density Estimation

"""start = time.time()
output_matrix, vec_fields = general_test_SDE(gaussian, dt=1, lam=0.1, steps=40, N=3, m=300, method="entropic", sinkhorn_reg = 10000, warm_start=True, reg_og =1, reg_sample = 1)
end = time.time()
print("Time taken  = " +str(end-start) + " seconds")
show_empirical(output_matrix)"""


### TESTING Sampling

probabilities = []
counts = {}
for i in range(dim1):
    for j in range(dim2):
        probabilities.append(gaussian[i,j])
        counts[(i,j)] = 0
sample = np.random.choice(len(squares), 150, p=probabilities)
for index, val in enumerate(sample):
    counts[squares[val]] += 1

show_empirical(counts)
generated = generate_sample(size=600,init_sample=counts, dt=0.1, lam=0.3, steps=40, method="standard", sinkhorn_reg=10000, reg_og = 1, reg_sample = 1, warm_start = False)
show_empirical(generated)
