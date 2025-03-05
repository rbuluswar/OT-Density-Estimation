import numpy as np
import cv2
import ot
import math
import matplotlib.pyplot as plt
from ot.utils import *
import pandas as pd


"""Initializing the image."""
img_path = "/Users/rohanbuluswar/Desktop/CS_Research/Spiral.png"
img = cv2.imread(img_path, 0)
dim1 = 64
dim2 = 64
img = cv2.resize(img, (dim1,dim2))



"""Given an initial image, properly formats it as probability distribution matrix with lighter region corresponding to lower density. """
reverse_img = np.ones([dim1,dim2])
for x in range(dim1):
    for y in range(dim2):
        reverse_img[x,y] = (255-img[x,y])

tot = np.sum(reverse_img)
init_img = np.ones([dim1,dim2])
for x in range(dim1):
    for y in range(dim2):
        init_img[x,y] = reverse_img[x,y]/tot

### cv2.imshow("photo", img)
### cv2.waitKey(0)
### cv2.destroyAllWindows()


"""Currently does nothing, because I am deciding between two options. 
Displaying color/darkness as density relative to maximum, or simply as proportional to the  density itself.
The first option leaves the uniform distribution as completely black, while the second option leaves it completely white."""
def show_photo(matrix):
    new_matrix = np.ones([dim1,dim2])
    for x in range(dim1):
        for y in range(dim2):
            new_matrix[x,y] = (1-matrix[x,y]/np.max(matrix))
    new_matrix = 255*new_matrix
    cv2.imshow("photo_1", new_matrix)
    ###uint_img = new_matrix.astype('uint8')
    ### cv2.imshow("photo", uint_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""Visualizes probability distribution by taking a large number of samples according to it.
Takes as in put a probability distribution matrix and displays an image n samples from it. 
Probably want to choose n~ 16,000, for this case."""
def show_sample(matrix, num):
    squares = []
    probabilities = []
    counts = {}
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
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


"""Takes as an input a probability matrix, initial matrix, and a step size and returns another probability matrix after making one step of discretized PDE.""" 
def PDE_step(matrix, init_matrix, dt):
    squares = []
    prob_curr = []
    prob_og = []
    new_matrix = np.ones([dim1,dim2])

### Construct list of indices representing squares in our grid, and initial and current probability distributions 
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            prob_curr.append(matrix[i,j])
            prob_og.append(init_matrix[i,j])

### Construct cost matrix representing transport cost from each square in our grid to each other square

    cost_matrix = np.ones([dim1*dim2,dim1*dim2])
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)

### Solve optimal transport to obtain coupling matrix, define intermediate step matrix (of which we will take the divergence)
    T = ot.emd(prob_curr,prob_og,cost_matrix)
    x_inter_matrix = np.ones([dim1,dim2])
    y_inter_matrix = np.ones([dim1,dim2])

### Do calculation or grad_log and estimate transport map to find intermediate step matrix
    for i in range(dim1):
        for j in range(dim2):
            if 0< i < dim1-1 and 0< j < dim2-1:

                if matrix[i,j] == 0 or matrix[i+1,j] == 0 or matrix[i,j+1] ==0 or matrix[i-1,j]==0 or matrix[i,j-1]==0:
                    grad_log = (0,0)
                else:
                    grad_log = (0.5*(np.log(matrix[i+1,j])-np.log(matrix[i-1,j])), 0.5*(np.log(matrix[i,j+1])-np.log(matrix[i,j-1]))  )
            
            elif i == dim1-1 and j == 0:
                if matrix[i,j] == 0 or matrix[i-1,j] == 0 or matrix[i,j+1] ==0:
                    grad_log = (0,0)
                else:
                    grad_log = ( np.log(matrix[i,j])-np.log(matrix[i-1,j]), np.log(matrix[i,j+1])-np.log(matrix[i,j])  )

            elif i == 0 and j == dim2-1:
                if matrix[i,j] == 0 or matrix[i+1,j] == 0 or matrix[i,j-1] ==0:
                    grad_log = (0,0)
                else:
                    grad_log = ( np.log(matrix[i+1,j])-np.log(matrix[i,j]), np.log(matrix[i,j])-np.log(matrix[i,j-1])  )
            
            else:
                if matrix[i,j] == 0 or matrix[i-1,j] == 0 or matrix[i,j-1] == 0:
                    grad_log = (0,0)
                else:
                    grad_log = (np.log(matrix[i,j]) - np.log(matrix[i-1,j]), np.log(matrix[i,j])-np.log(matrix[i,j-1]))
           
            index = squares.index((i,j))
            row = T[index]
            x_sum = 0
            y_sum = 0
            if row.sum() == 0:
                x_sum = i
                y_sum = j
            elif row.sum() > 0 : 
                for index, (i,j) in enumerate(squares):
                    x_sum += (row[index]*i)/row.sum()
                    y_sum += (row[index]*j)/row.sum()
            new_x = matrix[i,j]*(1*grad_log[0]+2*i-2*x_sum)
            new_y = matrix[i,j]*(1*grad_log[1]+2*j-2*y_sum)
            x_inter_matrix[i,j] = new_x
            y_inter_matrix[i,j] = new_y

### Take divergence of intermediate step matrix to find new distribution
    for i in range(dim1):
        for j in range(dim2):
            if i < dim1-1 and j < dim2-1:
                dx = x_inter_matrix[i+1,j]-x_inter_matrix[i,j]
                dy = y_inter_matrix[i,j+1]-y_inter_matrix[i,j]
            
            elif i == dim1-1 and j == 0:
                dx = x_inter_matrix[i,j]-x_inter_matrix[i-1,j]
                dy = y_inter_matrix[i,j+1]-y_inter_matrix[i,j]
            
            elif i == 0 and j == dim2-1:
                dx = x_inter_matrix[i+1,j]-x_inter_matrix[i,j]
                dy = y_inter_matrix[i,j]-y_inter_matrix[i,j-1]

            else:
                dx = x_inter_matrix[i,j]-x_inter_matrix[i-1,j]
                dy = y_inter_matrix[i,j]-y_inter_matrix[i,j-1]
  
            new_matrix[i,j]=max(matrix[i,j]+ dt*(dx+dy), 0)

### Normalize output matrix, just in case
    total = new_matrix.sum()       
    ### print(total)
    for i in range(dim1):
        for j in range(dim2):
            new_matrix[i,j] = new_matrix[i,j] / total
    return(new_matrix)


""""Takes as an input a probability matrix, initial matrix, and a step size and returns a sample from a distribution after taking one step of discretized SDE.
Also takes as inputs a regularization parameter lambda and a method - standard or entropic optimal transport.""" 
def general_SDE_step(matrix, init_matrix, dt, num, lam, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1):
        ### Initializing list of squares, current & original probabilities, initial sample, and dictionary for output sample.
    squares = []
    probabilities = []
    prob_og = []
    counts = {}
    new_counts = {}
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            probabilities.append(matrix[i,j])
            counts[(i,j)] = 0
            new_counts[(i,j)] = 0
            prob_og.append(init_matrix[i,j])

    ### Initializing cost matrix and solving optimal transport
    cost_matrix = np.ones([dim1*dim2,dim1*dim2])
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    if method == "standard":
        T = ot.emd(probabilities,prob_og,cost_matrix)
    if method == "entropic":
        T = ot.sinkhorn(probabilities,prob_og,cost_matrix, sinkhorn_reg)
    if method == "unbalanced":
        T = ot.sinkhorn_unbalanced(probabilities, prob_og, cost_matrix, reg = sinkhorn_reg, reg_m = (reg_sample,reg_og))

    ### Initializing sample from current prob. distribution
    sample = np.random.choice(len(squares), num, p=probabilities)
    for index, val in enumerate(sample):
        counts[squares[val]] += 1
    
    ### Moving each particle 
    for index, square in enumerate(squares):
        for _ in range(counts[square]):
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
            new_x = x_0 - lam*dt*(2*x_0-2*x_sum)+np.sqrt(2)*B_x
            new_y = y_0 - lam*dt*(2*y_0-2*y_sum)+np.sqrt(2)*B_y
            if new_x < 0:
                new_x = 0
            elif new_x >= dim1-0.5:
                new_x = dim1-1
            else:
                new_x = np.round(new_x,0)
            if new_y < 0:
                new_y = 0
            elif new_y >= dim2-0.5:
                new_y = dim2-1
            else:
                new_y = np.round(new_y,0)
            new_counts[(new_x,new_y)]+=1
    return new_counts  



"""Performs one step of the SDE given a matrix of particle locations (dictionary) instead of a probability matrix. """
def general_SDE_from_sample(sample, init_matrix, dt, lam, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = None):
    ### Initializing list of squares, deriving probability matrix from sample. 
    ### Initializing current and initial probability distributions, and new counts dictionary.
    vec_field = np.zeros((dim1, dim2, 2))

    squares = []
    curr_prob = []
    prob_og = []
    new_counts = {}
    total = sum(sample.values())
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            prob_og.append(init_matrix[i,j])
            new_counts[(i,j)] = 0
    for index, val in enumerate(squares):
        curr_prob.append(sample[val] / total)

    ### Initializing cost matrix and solving optimal transport
    cost_matrix = np.ones([dim1*dim2,dim1*dim2])
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    if method == "entropic":
        T = ot.sinkhorn(np.array(curr_prob),np.array(prob_og),cost_matrix, sinkhorn_reg)
    if method == "standard":
        T = ot.emd(curr_prob,prob_og,cost_matrix)
    if method == "unbalanced":
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
            else:
                new_x = np.round(new_x,0)
            if new_y < 0:
                new_y = 0
            elif new_y >= dim2-0.5:
                new_y = dim2-1
            else:
                new_y = np.round(new_y,0)
            new_counts[(new_x,new_y)]+=1
            x_coords += - lam*dt*(2*x_0-2*x_sum)+np.sqrt(2)*B_x
            y_coords += - lam*dt*(2*y_0-2*y_sum)+np.sqrt(2)*B_y
        (i,j) = square
        if sample[square] > 0:
            vec_field[i,j,0] = x_coords/sample[square]
            vec_field[i,j,1] = y_coords/sample[square]
        else:
            vec_field[i,j,0] = 0
            vec_field[i,j,1] = 0
    return (new_counts, vec_field)


"""Like SDE_step, but with no drift term (i.e. no optimal transport). 
Only moves particles according to Brownian motion. """
def BM_step(matrix, init_matrix, dt, num):
    ### Initializing list of squares, current & original probabilities, initial sample, and dictionary for output sample.
    squares = []
    probabilities = []
    prob_og = []
    counts = {}
    new_counts = {}
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            probabilities.append(matrix[i,j])
            counts[(i,j)] = 0
            new_counts[(i,j)] = 0
            prob_og.append(init_matrix[i,j])

    ### Initializing sample from probability matrix
    sample = np.random.choice(len(squares), num, p=probabilities)
    for index, val in enumerate(sample):
        counts[squares[val]] += 1
    
    ### Moving each particle according to Brownian motion
    for index, square in enumerate(squares):
        for _ in range(counts[square]):
            x_0 = square[0]
            y_0 = square[1]
            B_x = np.random.normal(0,dt)
            B_y = np.random.normal(0,dt)
            new_x = x_0 + np.sqrt(2)*B_x
            new_y = y_0 + np.sqrt(2)*B_y
            if new_x < 0:
                new_x = 0
            elif new_x >= dim1-0.5:
                new_x = dim1-1
            else:
                new_x = np.round(new_x,0)
            if new_y < 0:
                new_y = 0
            elif new_y >= dim2-0.5:
                new_y = dim2-1
            else:
                new_y = np.round(new_y,0)
            new_counts[(new_x,new_y)]+=1

    return new_counts



def BM_from_sample(sample, dt):
    ### Initializing list of squares and new_counts dictionary. 
    squares = []
    new_counts = {}
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            new_counts[(i,j)] = 0
    
    ### Moving each particle according to Brownian motion
    for index, square in enumerate(squares):
        for _ in range(sample[square]):
            x_0 = square[0]
            y_0 = square[1]
            B_x = np.random.normal(0,dt)
            B_y = np.random.normal(0,dt)
            new_x = x_0 + np.sqrt(2)*B_x
            new_y = y_0 + np.sqrt(2)*B_y
            if new_x < 0:
                new_x = 0
            elif new_x >= dim1-0.5:
                new_x = dim1-1
            else:
                new_x = np.round(new_x,0)
            if new_y < 0:
                new_y = 0
            elif new_y > dim2-0.5:
                new_y = dim2-1
            else:
                new_y = np.round(new_y,0)
            new_counts[(new_x,new_y)]+=1

    return new_counts
    
    
"""Given to matrices, want to visualize the coupling by seeing where the probability mass at a given index (i,j) is sent. 
The distribution is visualized by a sample of size num. """    
### Good example for spiral: i = 35, j = 38. Generally, look for something in the support of [matrix].
def see_coupling(matrix, init_matrix, i_coord, j_coord, num):
    new_matrix = np.ones([dim1,dim2])
    squares = []
    prob_curr = []
    prob_og = []
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            prob_curr.append(matrix[i,j])
            prob_og.append(init_matrix[i,j])
    cost_matrix = np.ones([dim1*dim2,dim1*dim2])
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    T = ot.emd(prob_curr,prob_og,cost_matrix)
    index = squares.index((i_coord,j_coord))
    print(index)
    row = T[index]
    x_sum = 0
    y_sum = 0
    if row.sum() > 0: 
        for index, (i,j) in enumerate(squares):
            new_matrix[i,j] = row[index]/row.sum()
        show_sample(new_matrix, num)
    elif row.sum()==0:
        print('Choose a different index')


### Given an initial matrix representing a sample from a distribution, init_matrix,
### a step size dt, a weighting parameter lam, a number of steps, and a factor of N
### for scaling up the sample, takes steps according to SDE_from_sample and returns the matrix of particles.
### All sample matrices represented as dictionaries.

def general_simulate_SDE(prob_matrix, init_sample, dt, lam, steps, N, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = False):
    vec_fields = []
    init_matrix_dict = {}
    for i in range(dim1):
        for j in range(dim2):
            init_matrix_dict[(i,j)] =init_sample[i,j] * N
    new_matrix_dict = init_matrix_dict
    for _ in range(steps):
        updated_matrix, new_vec_field = general_SDE_from_sample(new_matrix_dict, prob_matrix, dt, lam, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample=reg_sample)
        new_matrix_dict = updated_matrix 
        vec_fields.append(new_vec_field)
    return new_matrix_dict, vec_fields



### Given a probability matrix, takes a sample of size m and performs simulate_SDE.
### Then shows final result and prints OT distance between prob_matrix and 
### empirical sample, and between prob_matrix and output empirical distribution.

def general_test_SDE(prob_matrix, dt, lam, steps, N, m, method, sinkhorn_reg=1, reg_og = 1, reg_sample = 1, warm_start = False):
    ### Take sample from prob_matrix
    squares = []
    probabilities = []
    counts = {}
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            probabilities.append(prob_matrix[i,j])
            counts[(i,j)] = 0
    sample = np.random.choice(len(squares), m, p=probabilities)
    for index, val in enumerate(sample):
        counts[squares[val]] += 1
    
    
    output_matrix, vec_fields = general_simulate_SDE(prob_matrix, counts, dt, lam, steps, N, method = method, sinkhorn_reg=sinkhorn_reg, reg_og=reg_og, reg_sample = reg_sample)
    
    ### Initialize cost matrix
    cost_matrix = np.ones([dim1*dim2,dim1*dim2])
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    
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

### Given a sample
def test_SDE_from_sample(sample, prob_matrix, dt, lam, steps, N, method, sinkhorn_reg = 1, reg_og = 1, reg_sample = 1, warm_start = False):
    ### Initializing
    squares = []
    probabilities = []
    counts = sample
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
    init_matrix = np.ones([dim1,dim2])
    for i in range(dim1):
        for j in range(dim2):
            init_matrix[i,j] = counts[(i,j)]

    ### Simulation
    output_matrix, vec_fields = general_simulate_SDE(prob_matrix, counts, dt, lam, steps, N, method = method, sinkhorn_reg = sinkhorn_reg, reg_og = reg_og, reg_sample = reg_sample)
    
    ### Initialize cost matrix
    cost_matrix = np.ones([dim1*dim2,dim1*dim2])
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    
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
    



#### CREATING TOY DISTRIBUTIONS

### Generate Gaussian
mean = [35, 40]
var = [35,25]
gaussian = np.ones([dim1,dim2])
for i in range(dim1):
    for j in range(dim2):
        exponent = -0.5 * (((i-mean[0])**2 / var[0])+((j-mean[1])**2 / var[1]))
        gaussian[i,j] = math.exp(exponent)
total = gaussian.sum()
for i in range(dim1):
    for j in range(dim2):
        gaussian[i,j] = gaussian[i,j] / total


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



### TESTING BELOW


### Testing
test_output, vec_fields = general_test_SDE(circular, dt=1, lam=1, steps=3, N=2, m=300, method = "entropic")
new_matrix = np.ones([dim1,dim2])

for i in range(dim1):
    for j in range(dim2):
        if test_output[(i,j)] > 0:
            new_matrix[i,j] = 0
        else:
            new_matrix[i,j] = 255
cv2.imshow("photo", new_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Visualizing vector fields
field = vec_fields[1]
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
plt.xlim(-45, 45) 
plt.ylim(-45, 45) 
plt.grid() 
plt.show() 


### For testing/comparing methods while varying one parameter
"""
squares = []
probabilities = []
counts = {}
for i in range(dim1):
    for j in range(dim2):
        squares.append((i,j))
        probabilities.append(mixed[i,j])
        counts[(i,j)] = 0

x = [1,2,3]
y = []
y2 = []

for nval in x:

    new_counts = {}
    for i in range(dim1):
        for j in range(dim2):
            new_counts[(i,j)] = 0
    sample = np.random.choice(len(squares), 300, p=probabilities)
    for index, val in enumerate(sample):
        new_counts[squares[val]] += 1
    
    y.append(test_SDE_from_sample(new_counts, mixed, dt =1, lam = 100, steps = 3, N = nval))
    print("Original SDE: ")
    print(y)
    
    y2.append(test_entropic_SDE_from_sample(new_counts, mixed, dt=1, lam = 100, steps = 3, N = nval))
    print("Entropic SDE: ")
    print(y2)

plt.plot(x,y, label = "Original")
plt.plot(x, y2, label = "Entropic")
plt.ylabel("percentage decrease in OT distance")
plt.legend()
plt.show()

data = []
for i in range(len(x)):
    new_row = [x[i],y[i],y2[i]]
    data.append(new_row)
datarray = np.array(data)

df = pd.DataFrame(data, columns=["N val","Original SDE OT % Decrease","Entropic SDE OT % Decrease"])
df.to_csv("MixedExperiment2b.csv",index = False)"""



#### QUESTIONS:
#### - do we want to use regularization with Negative entropy or KL divergence in unbalanced method??
###  - Did I mess up initial placement of factor of lambda? (initially on the optimal transport term and not Brownian motion)

### NOTES: 
### Can we use a neural network to infer Kantorovich potential of true p_0 to true p_t from optimal transport matrix
### Probability ODE?? (see most recent diffusions paper)  

### Current Tasks:
###     - STORE AND PLOT VECTOR FIELDS/transport maps 
###     - WRITE CODE TO REVERSE ENGINEER u and v from T for warmstart
###     - do testing with new additions

###     - Compare our results with usual diffusion algorithms, using OT metric
###     - Read: Optimal Flow Matching, Input Convex Neural Networks, Maximally Monotone Operators 
###     - Think about how to learn Kantorovich potentials using neural network
### Other Goals:
###     - Architecture for moving program to images
###     - Given this, how do we sample?? 
###     - we get a noisy transport map that should be a gradient of a quadratic function 
###         - can do linear regression to figure out actual map 

