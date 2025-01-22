import numpy as np
import cv2
import ot


"""Initializing the image."""
img_path = "/Users/rohanbuluswar/Desktop/CS_Research/Spiral.png"
img = cv2.imread(img_path, 0)
dim1 = 75
dim2 = 75
img = cv2.resize(img, (dim1,dim2))


"""Given an initial image, properly formats it as probability distribution matrix with lighter region corresponding to lower density. """
reverse_img = np.matrix(np.ones([dim1,dim2]))
for x in range(dim1):
    for y in range(dim2):
        reverse_img[x,y] = (255-img[x,y])

tot = np.sum(reverse_img)
init_img = np.matrix(np.ones([dim1,dim2]))
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
    new_matrix = np.matrix(np.ones([dim1,dim2]))
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
    new_matrix = np.matrix(np.ones([dim1,dim2]))
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
    new_matrix = np.matrix(np.ones([dim1,dim2]))

### Construct list of indices representing squares in our grid, and initial and current probability distributions 
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            prob_curr.append(matrix[i,j])
            prob_og.append(init_matrix[i,j])

### Construct cost matrix representing transport cost from each square in our grid to each other square

    cost_matrix = np.matrix((np.ones([dim1*dim2,dim1*dim2])))
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)

### Solve optimal transport to obtain coupling matrix, define intermediate step matrix (of which we will take the divergence)
    T = ot.emd(prob_curr,prob_og,cost_matrix)
    x_inter_matrix = np.matrix(np.ones([dim1,dim2]))
    y_inter_matrix = np.matrix(np.ones([dim1,dim2]))

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

""""Takes as an input a probability matrix, initial matrix, and a step size and returns a sample from a distribution after taking one step of discretized SDE.""" 
def SDE_step(matrix, init_matrix, dt, num, lam):
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
    cost_matrix = np.matrix((np.ones([dim1*dim2,dim1*dim2])))
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    T = ot.emd(probabilities,prob_og,cost_matrix)

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
def SDE_from_sample(sample, init_matrix, dt, lam):
    ### Initializing list of squares, deriving probability matrix from sample. 
    ### Initializing current and initial probability distributions, and new counts dictionary.
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
    cost_matrix = np.matrix((np.ones([dim1*dim2,dim1*dim2])))
    for (index_1, (i,j)) in enumerate(squares):
        for (index_2, (a,b)) in enumerate(squares):
            cost_matrix[index_1,index_2] = np.sqrt((a-i)**2+(b-j)**2)
    T = ot.emd(curr_prob,prob_og,cost_matrix)

    ### Moving each particle 
    for index, square in enumerate(squares):
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
    new_matrix = np.matrix(np.ones([dim1,dim2]))
    squares = []
    prob_curr = []
    prob_og = []
    for i in range(dim1):
        for j in range(dim2):
            squares.append((i,j))
            prob_curr.append(matrix[i,j])
            prob_og.append(init_matrix[i,j])
    cost_matrix = np.matrix((np.ones([dim1*dim2,dim1*dim2])))
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
def simulate_SDE(prob_matrix, init_sample, dt, lam, steps, N):
    init_matrix_dict = {}
    for i in range(dim1):
        for j in range(dim2):
            init_matrix_dict[(i,j)] =init_sample[i,j] * N
    new_matrix_dict = init_matrix_dict
    for _ in range(steps):
        updated_matrix = SDE_from_sample(new_matrix_dict, prob_matrix, dt, lam)
        new_matrix_dict = updated_matrix 
    return new_matrix_dict



### Given a probability matrix, takes a sample of size m and performs simulate_SDE.
### Then shows final result and prints OT distance between prob_matrix and 
### empirical sample, and between prob_matrix and output empirical distribution.

def test_SDE(prob_matrix, dt, lam, steps, N, m):
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
    
    init_matrix = np.matrix((np.ones([dim1,dim2])))
    for i in range(dim1):
        for j in range(dim2):
            init_matrix[i,j] = counts[(i,j)]
    
    output_matrix = simulate_SDE(prob_matrix, counts, dt, lam, steps, N)
    
    ### Initialize cost matrix
    cost_matrix = np.matrix((np.ones([dim1*dim2,dim1*dim2])))
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
    return output_matrix









