#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt 
from queue import PriorityQueue
import pickle

def generate_gridworld(length, width, probability,start,end):
    solvable = False
    while not solvable:
        grid = np.random.choice([0,1], length * width, p = [1 - probability, probability]).reshape(length, width)
        grid[0][0] = 0
        grid[-1][-1] = 0
        curr_knowledge = generate_knowledge(grid,start,end,revealed = True)
        solvable = A_star(curr_knowledge, start, end)
    return grid

# Calculates h(x) using one of a range of heuristics
def hureisticValue(point1, point2):
    x1,y1=point1
    x2,y2=point2
    return abs(x1 - x2) + abs(y1 - y2)

class Cell:
    def __init__(self, row, col, dim):
        self.row=row
        self.col=col
        self.dim=dim
        self.neighbors=[]
        self.visited=False
        self.blocked=2
        self.c=9999
        self.b=9999
        self.e=9999
        self.h=9999
        self.n=len(self.neighbors)

    def getPos(self):
        return self.col, self.row
    #return a list of neighbors of the current cell
    def findneighbors(self):
        dim = self.dim
        y, x = self.getPos()
        neighbors = [(y2, x2) for x2 in range(x-1, x+2)                     
                              for y2 in range(y-1, y+2)
                              if (-1 < x < dim and
                                  -1 < y < dim and
                                  (x != x2 or y != y2) and
                                  (0 <= x2 < dim) and
                                  (0 <= y2 < dim))]
        return neighbors
    #sense the neigbhors, update current cell's info
    def sensing(self, knowledge,grid):
        neighbors = knowledge[self.row][self.col].findneighbors()
        c = 0
        b = 0
        e = 0
        h = 0
        for cell in neighbors:
            x, y = cell
            if grid[y][x] == 1:
                c += 1
            if knowledge[y][x].blocked == 1:
                b += 1
            if knowledge[y][x].blocked == 0:
                e += 1
            if knowledge[y][x].blocked == 2:
                h += 1
        self.c = c
        self.b = b
        self.e = e
        self.h = h
        self.n = len(neighbors)
                         
    def __lt__(self, other):
        return False

#generate knowledge embeded with cell class
def generate_knowledge(grid,start,end,revealed = False):
    cell_list = []
    rows = len(grid)
    cols = len(grid[0])
    #calculate initial_prob of each cell
    g_index = np.where(grid !=1)
    initial_prob = 1/len(g_index[0])
    for i in range(rows):
        cell_list.append([])
        for j in range(cols):
            cellOBJ = Cell(i,j,rows)
            if revealed:
                cellOBJ.blocked = grid[i][j] 
            cellOBJ.prob= initial_prob
            cell_list[i].append(cellOBJ)
    cell_list[start[1]][start[0]].blocked =0
    cell_list[end[1]][end[0]].blocked =0
    return cell_list


#Add (x,y) neighbors which has been visited and h != 0
def add_visited_neighbors(y,x,knowledge,queue):
    for neigbor in knowledge[y][x].findneighbors():
        x2, y2 = neigbor
        if knowledge[y2][x2].h != 0 and knowledge[y2][x2].visited == True and neigbor not in queue:
            queue.append(neigbor)
    return queue

#inferencing info of current cell, if updated then propagate to its visited neighbors and so on
def infering(y,x,knowledge,grid):
    queue = [(x,y)]
    queue = add_visited_neighbors(y,x,knowledge,queue)
    while len(queue) != 0:
        for item in queue:
            queue.remove(item)
            x1, y1 = item
            neighbors = knowledge[y1][x1].findneighbors()
            knowledge[y1][x1].sensing(knowledge,grid)
            #see if the information contradicts the current knowledge
            if knowledge[y1][x1].n < (knowledge[y1][x1].c + knowledge[y1][x1].e):
                return False
            elif knowledge[y1][x1].b > knowledge[y1][x1].c:
                return False
            if knowledge[y1][x1].c == knowledge[y1][x1].b:
                knowledge[y1][x1].h == 0
                for neighbor in neighbors:
                    x2, y2 = neighbor
                    if knowledge[y2][x2].blocked == 2:
                        knowledge[y2][x2].blocked = 0
                        #if a neighbor was updated, then add its visited neigbhors to queue
                        queue = add_visited_neighbors(y2,x2,knowledge,queue)
            elif knowledge[y1][x1].n - knowledge[y1][x1].c == knowledge[y1][x1].e:
                knowledge[y1][x1].h == 0
                for neighbor in neighbors:
                    x2, y2 = neighbor
                    if knowledge[y2][x2].blocked == 2:
                        knowledge[y2][x2].blocked = 1
                        #if a neighbor was updated, then add its visited neigbhors to queue
                        queue = add_visited_neighbors(y2,x2,knowledge,queue)
    #if knowledge[y1][x1].c == 1 and knowledge[y1][x1].h != 0:
    return knowledge

def A_star(curr_knowledge, start, end):
    # Initializes the g(x), f(x), and h(x) values for all squares
    g = {(x, y):float("inf") for y, eachRow in enumerate(curr_knowledge) for x, eachcolumn in enumerate(eachRow)}
    g[start] = 0
    f = {(x, y):float("inf") for y, eachRow in enumerate(curr_knowledge) for x, eachcolumn in enumerate(eachRow)}
    f[start] = hureisticValue(start, end)
    h = {(x, y): hureisticValue((x, y), end) for y, eachRow in enumerate(curr_knowledge) for x, eachcolumn in enumerate(eachRow)}
    parent = {}
    visited={start} # it is a set which provide the uniqueness, means it is ensure that not a single cell visit more than onece.
    tiebreaker = 0
	# Creates a priority queue using a Python set, adding start cell and its distance information
    pq = PriorityQueue()
    pq.put((f[start], tiebreaker, start))
    #count cell being processed
    cell_count = 0
    # A* algorithm, based on assignment instructions
    while not pq.empty():
		# Remove the node in the priority queue with the smallest f value
        n = pq.get()
        cell_count += 1
        successors = []
		# curr_pos is a tuple (x, y) where x represents the column the square is in, and y represents the row
        curr_pos = n[2]
        visited.remove(curr_pos)
		# if goal node removed from priority queue, shortest path found
        if curr_pos == end:
            shortest_path = []
            path_pointer = end
            while path_pointer != start:
                shortest_path.append(path_pointer)
                path_pointer = parent[path_pointer]
            shortest_path.append(start)
            shortest_path = shortest_path[::-1]
            return [shortest_path, cell_count]	
		# Determine which neighbors are valid successors
        #add unblocked or unknown cells
        if curr_pos[0] > 0 and curr_knowledge[curr_pos[1]][curr_pos[0] - 1].blocked != 1: # the current node has a neighbor to its left which is unblocked
            left_neighbor = (curr_pos[0] - 1, curr_pos[1])
            if g[left_neighbor] > g[curr_pos] + 1: # if neighbor is undiscovered
                successors.append(left_neighbor)
				
        if curr_pos[0] < len(curr_knowledge[0])  - 1 and curr_knowledge[curr_pos[1]][curr_pos[0] + 1].blocked != 1: # the current node has a neighbor to its right which is unblocked
            right_neighbor = (curr_pos[0] + 1, curr_pos[1])
            if g[right_neighbor] > g[curr_pos] + 1:
                #if neighbor is undiscovered
                successors.append(right_neighbor)
   
        if curr_pos[1] > 0 and curr_knowledge[curr_pos[1] - 1][curr_pos[0]].blocked != 1: # the current node has a neighbor to its top which is unblocked
            top_neighbor = (curr_pos[0], curr_pos[1] - 1)
            if g[top_neighbor] > g[curr_pos] + 1: # if neighbor is undiscovered
                successors.append(top_neighbor)
                
        if curr_pos[1] < len(curr_knowledge) - 1 and curr_knowledge[curr_pos[1] + 1][curr_pos[0]].blocked != 1: # the current node has a neighbor to its bottom which is unblocked
            bottom_neighbor = (curr_pos[0], curr_pos[1] + 1)
            if g[bottom_neighbor] > g[curr_pos] + 1: # if neighbor is undiscovered
                successors.append(bottom_neighbor)
			
        # Update shortest paths and parents for each valid successor and add to priority queue, per assignment instructions
        for successor in successors:
            g[successor] = g[curr_pos] + 1
            parent[successor] = curr_pos
            if successor not in visited:
                tiebreaker += 1
                pq.put((g[successor] + h[successor], -tiebreaker, successor))
                visited.add(successor)
		# if priority queue is empty at any point, then unsolvable
        if pq.empty():
            return False

# Handles processing of Repeated A*, restarting that algorithm if a blocked square is found in the determined shortest path
def algorithmA(grid, start, end, data_x, data_y, has_four_way_vision):
    # The assumed state of the gridworld at any point in time. For some questions, the current knowledge is unknown at the start
    curr_knowledge = generate_knowledge(grid,start,end)
    # If the grid is considered known to the robot, operate on that known grid
	# Else, the robot assumes a completely unblocked gridworld and will have to discover it as it moves
    complete_path = [(0,0)]
    prev_sq = (0,0)
	# Run A* once on grid as known, returning False if unsolvable
    shortest_path = A_star(curr_knowledge, start, end)
    if not shortest_path:
        return False
    is_broken = False
    cell_count = shortest_path[1]
    while True:
		# Move pointer square by square along path
        for sq in shortest_path[0]:
            x = sq[0]
            y = sq[1]
            if sq != prev_sq:
                data_x = add_data_x(curr_knowledge,data_x, prev_sq)
			# If blocked, rerun A* and restart loop
            if grid[y][x] == 1:
                # If the robot can only see squares in its direction of movement, update its current knowledge of the grid to include this blocked square
                if not has_four_way_vision:
                    curr_knowledge[y][x].blocked = 1
                shortest_path = A_star(curr_knowledge, prev_sq, end)   
                if not shortest_path:
                    return False
                is_broken = True
                cell_count += shortest_path[1]
                data_y = add_data_y(prev_sq,sq,data_y)
                break
			# If new square unblocked, update curr_knowledge. Loop will restart and move to next square on presumed shortest path
            else:
                if sq != complete_path[-1]:
                    complete_path.append(sq)
                curr_knowledge[y][x].blocked = 0
                # If the robot can see in all compass directions, update squares adjacent to its current position
                if has_four_way_vision:
                    if x != 0:
                        curr_knowledge[y][x - 1].blocked = grid[y][x - 1]
                    if x < len(curr_knowledge[0]) - 1:
                        curr_knowledge[y][x + 1].blocked = grid[y][x + 1]
                    if y != 0:
                        curr_knowledge[y - 1][x].blocked = grid[y - 1][x]
                    if y < len(curr_knowledge) - 1:
                        curr_knowledge[y + 1][x].blocked = grid[y + 1][x]
                    if is_path_blocked(curr_knowledge, shortest_path):
                         shortest_path = A_star(curr_knowledge, sq, end)
                         data_y = add_data_y(prev_sq,sq,data_y)
                         prev_sq = sq
                         is_broken = True
                         break
                data_y = add_data_y(prev_sq,sq,data_y)
            prev_sq = sq
        if not is_broken:
            break
        is_broken = False
    return [complete_path, cell_count,data_x,data_y]

def inference(grid, start, end, data_x, data_y):
    #generate initial shortest path
    curr_knowledge = generate_knowledge(grid, start, end)
    complete_path = [(0,0)]
    prev_sq = (0,0)
    shortest_path = A_star(curr_knowledge, start, end)
    if not shortest_path:
        return False
    is_broken = False
    cell_count = shortest_path[1]
    while True:
		# Move pointer square by square along path
        for sq in shortest_path[0]:
            x = sq[0]
            y = sq[1]
            if sq != prev_sq:
                data_x = add_data_x_inference(curr_knowledge,data_x,prev_sq)
			# If blocked, rerun A* and restart loop            
            if grid[y][x] == 1:
                curr_knowledge[y][x].blocked = 1
                x1, y1 = prev_sq
                curr_knowledge = infering(y1,x1,curr_knowledge,grid)
                shortest_path = A_star(curr_knowledge, prev_sq, end)
                #print(shortest_path)
                if not shortest_path:
                    return False
                is_broken = True
                cell_count += shortest_path[1]
                data_y = add_data_y(prev_sq,sq,data_y)
                break
			# If new square unblocked, update curr_knowledge. Loop will restart and move to next square on presumed shortest path
            else:
                if sq != complete_path[-1]:
                    complete_path.append(sq)
                curr_knowledge[y][x].blocked = 0
                curr_knowledge[y][x].visited = True
                if curr_knowledge[y][x].h != 0:
                    curr_knowledge = infering(y,x,curr_knowledge,grid)
                    if is_path_blocked(curr_knowledge, shortest_path):
                        shortest_path = A_star(curr_knowledge, sq, end)
                        data_y = add_data_y(prev_sq,sq,data_y)
                        prev_sq = sq
                        is_broken = True
                        break
                data_y = add_data_y(prev_sq,sq,data_y)
            prev_sq = sq
        if not is_broken:
            break
        is_broken = False
    return [complete_path, cell_count,data_x,data_y]    

def is_path_blocked(curr_knowledge, shortest_path):
    for cell in shortest_path[0]:
        x1, y1 = cell
        if curr_knowledge[y1][x1].blocked == 1:
            return True
    return False    
       
#generate dataset_x for currrent_knowledge, {0:unblocked, 1:blocked, 3:unknown}
def add_data_x(curr_knowledge,data_x,prev_sq):
    dim = len(curr_knowledge)
    curr_x = []
    for y in range(dim):
        for x in range(dim):
            curr_x.append(curr_knowledge[y][x].blocked)
    #mark the agent's location, 9 represents where the agent is
    curr_x[prev_sq[1]*dim + prev_sq[0]] = 9
    data_x.append(curr_x)
    return data_x

#generate dataset_x for current_knowledge, with 5 digits represent c,b,e,h,block respectively
#for example, 53220 means c=5,b=3,e=2,h=2,unblocked; if the last digit is 9, it means the agent is there
def add_data_x_inference(curr_knowledge,data_x,prev_sq):
    dim = len(curr_knowledge)
    curr_x = [[], [], [], [], []]
    agent_sq = prev_sq[1]*dim + prev_sq[0]
	# Store the appropriate data for each square in the grid
    for y in range(dim):
        for x in range(dim):
            curr_x[0].append(curr_knowledge[y][x].c)
            curr_x[1].append(curr_knowledge[y][x].b)
            curr_x[2].append(curr_knowledge[y][x].e)
            curr_x[3].append(curr_knowledge[y][x].h)
            curr_x[4].append(curr_knowledge[y][x].blocked)
    # Note the agent's current location
    curr_x[4][agent_sq] = 9
	# One-hot encode all of the data using the to_categorical() function from Tensorflow
    # The first parameter indicates the data to be encoded, and the second the number of categories
    # The values must be integers from 0 to the number of categories
    encoded_data = []
    encoded_data.append(list(map(lambda x: 9 if x == 9999 else x, curr_x[0])))
    encoded_data.append(list(map(lambda x: 9 if x == 9999 else x, curr_x[1])))
    encoded_data.append(list(map(lambda x: 9 if x == 9999 else x, curr_x[2])))
    encoded_data.append(list(map(lambda x: 9 if x == 9999 else x, curr_x[3])))
    encoded_data.append(list(map(lambda x: 3 if x == 9 else x, curr_x[4])))
    data_x.append(encoded_data)
    return data_x

def add_data_y(prev_sq,sq,data_y):
    curr_y = "stay"
    if sq[0] - prev_sq[0] == 1:
        curr_y = "right"
    elif sq[0] - prev_sq[0] == -1:
        curr_y = "left"
    elif sq[1] - prev_sq[1] == 1:
        curr_y = "down"
    elif sq[1] - prev_sq[1] == -1:
        curr_y = "up"
    else:
        return data_y
    data_y.append(curr_y)
    return data_y
#generate training dataset for ML_agent
def generate_dataset():
    dataset_x = []
    dataset_y = []
    start = (0,0)
    end = (30,30)
    trial = 0
    while trial < 1000:
        print("running trial {}".format(trial))
        grid = generate_gridworld(31, 31, 0.3,start,end)
        agent1 = algorithmA(grid, start, end, dataset_x, dataset_y, has_four_way_vision = True)
        trial +=1
    return dataset_x, dataset_y

#generate training dataset for ML_agent_inference
def generate_dataset_inference():
    dataset_x = []
    dataset_y = []
    start = (0,0)
    end = (30,30)
    trial = 0
    while trial < 1000:
        print("running trial {}".format(trial))
        grid = generate_gridworld(31, 31, 0.3,start,end)
        agent3 = inference(grid, start, end, dataset_x, dataset_y)
        trial +=1
    return dataset_x, dataset_y

# Based on provided example notebook (https://colab.research.google.com/drive/11qqoQfeUiPtYF0feAKxdUHZCEuAnL_4H?usp=sharing)		 
def generate_dense_NN(dim, layers, num_datapoints):
    # Create an input layer with a number of neurons equal to the number of squares in the gridworld
    input_layer = tf.keras.layers.Input(shape=((dim**2) * num_datapoints))
    # Generate layers hidden layers, each with a preset number of neurons and the rectified linear unit activation function
	# Each layer is bound to the previous layer
    dense_layer = tf.keras.layers.Dense(units=layers[0], activation=tf.nn.relu)(input_layer)
    for i in range(len(layers) - 1):
        dense_layer = tf.keras.layers.Dense(units=layers[i + 1], activation=tf.nn.relu)(dense_layer)
	# Create the output layer, with four neurons representing the four directions our network cbooses between,
	# and bind it to the previous layers
    output_layer = tf.keras.layers.Dense(units=4, activation=None)(dense_layer)
	# Create the neural network
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)
	
# Based on provided example notebook (https://colab.research.google.com/drive/1psKxc1vhK4jvc-ego9dZTs6qXsnnJulG?usp=sharing)		 
def generate_conv_NN(dim, layers, size, filter, stride):
    # Create an input layer with a number of neurons equal to the number of squares in the gridworld, arranged in a rectangle
    input_layer = tf.keras.layers.Input(shape=(dim, dim, 1))
	# Add one convolutional hidden layer, with a window of size size, moving stride squares each iteration, and with a filter value of filter
    layer = tf.keras.layers.Conv2D(filters=filter, kernel_size=size, strides=stride, activation=tf.nn.relu)(input_layer)
    layer = tf.keras.layers.Flatten()(layer)
	# Generate layers hidden layers, each with a preset number of neurons and the rectified linear unit activation function
	# Each layer is bound to the previous layer
    for i in range(len(layers)):
        layer = tf.keras.layers.Dense(units=layers[i], activation=tf.nn.relu)(layer)
	# Create the output layer, with four neurons representing the four directions our network cbooses between,
	# and bind it to the previous layers
    output_layer = tf.keras.layers.Dense(units=4, activation=None)(layer)
	# Create the neural network
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Extra Credit architecture - Max Pool Architecture
#def Max_Pool_NN(dim, layers,stride,pool_size):
    # Create an input layer with a number of neurons equal to the number of squares in the gridworld
#   input_layer = tf.keras.layers.Input(shape=dim**2)
    # Generate layers hidden layers, each with a preset number of neurons and the rectified linear unit activation function
	# Each layer is bound to the previous layer
#    pool_layer =  tf.keras.layers.MaxPool2D(
#    pool_size=(pool_size, pool_size), strides=stride, padding='valid', data_format=None)
#    for i in range(len(layers) - 1):
#        player = tf.keras.layers.Dense(units=layers[i], activation=tf.nn.relu)(dense_layer)
	# Create the output layer, with four neurons representing the four directions our network cbooses between,
	# and bind it to the previous layers
#    output_layer = tf.keras.layers.Dense(units=4, activation=None)(player)
	# Create the neural network
#    return tf.keras.Model(inputs=input_layer, outputs=output_layer)
	
def test_model(NN, input, output):
    num_correct = 0
    for i, test_inp in enumerate(input):
        print(output[i])
		# Get the expected prediction using our neural network
        res = NN.predict([input[i]])
        print(res)
        max_res = max(res[0])
        for j, _ in enumerate(res):
            if res[0][j] == max_res:
                break
        for k, _ in enumerate(output[i]):
            if output[i][k] > 0:
                break
        if j == k:
            num_correct += 1
    return num_correct

#if has_four_way_vision = True, mimic agent1; if has_four_way_vision = False, mimic agent3
def run_ML_agent(NN, grid, start, end, has_four_way_vision = False):
    # The assumed state of the gridworld at any point in time. For some questions, the current knowledge is unknown at the start
    curr_knowledge = generate_knowledge(grid,start,end)
    # If the grid is considered known to the robot, operate on that known grid
	# Else, the robot assumes a completely unblocked gridworld and will have to discover it as it moves
    complete_path = [start]
    prev_sq = start
    sq = start
    cell_count = 0
    while sq != end:
        cell_count += 1
		#Use predict to generate the probability of 4 directions, the agent will move toward the direction with higest probability
        next_move = predict_direction(prev_sq, sq, curr_knowledge, has_four_way_vision)
        x, y = next_move
			# If blocked, rerun A* and restart loop
        if grid[y][x] == 1:
            # if 
            if not has_four_way_vision:
               curr_knowledge[y][x].blocked = 1
               x1, y1 = prev_sq
               curr_knowledge = infering(y,x,curr_knowledge,grid)
            continue
			# If new square unblocked, update curr_knowledge. Loop will restart and move to next square on presumed shortest path
        else:
            if sq != complete_path[-1]:
                complete_path.append(sq)
            curr_knowledge[y][x].blocked = 0
            curr_knowledge[y][x].visted = True
            # If the robot can see in all compass directions, update squares adjacent to its current position
            if has_four_way_vision:
               if x != 0:
                   curr_knowledge[y][x - 1].blocked = grid[y][x - 1]
               if x < len(curr_knowledge[0]) - 1:
                   curr_knowledge[y][x + 1].blocked = grid[y][x + 1]
               if y != 0:
                   curr_knowledge[y - 1][x].blocked = grid[y - 1][x]
               if y < len(curr_knowledge) - 1:
                   curr_knowledge[y + 1][x].blocked = grid[y + 1][x]
            elif curr_knowledge[y][x].h != 0:
                curr_knowledge = infering(y,x,curr_knowledge,grid)
        prev_sq = sq
        sq = next_move
    return [complete_path, cell_count]

def predict_direction(NN, prev_sq, sq, curr_knowledge,has_four_way_vision = False):
    #the predict return probabilities for direction down, up, left, right
    if has_four_way_vision:
        input_x = add_data_x(curr_knowledge,[],prev_sq)
    else:
        input_x = add_data_x_inference(curr_knowledge,[],prev_sq)
    #use trained model NN to generate probility for 4 directions given the current_knowledge
    possible_move = NN.predict(input_x)
    x = sq[0]
    y = sq[1]
    neighbors = curr_knowledge.find_neighbors()
    best_direction = 0
    candidates = {(x,y+1):possible_move[0],(x,y-1):possible_move[1],(x-1,y):possible_move[2],(x+1,y):possible_move[3]}
    max_prob = 0
    for direction in candidates.keys():
        x1,y1 = direction
        #remove invalid direction when it bumps into a known block, backward, or across the border
        if curr_knowledge[y1][x1].blocked == 1 or direction not in neighbors or direction == prev_sq:
            candidates.pop(direction, None)
    #after droping invalid directions, find the one with highest probability
    for direction, probability in candidates.items():
        if probability >= max_prob:
            max_prob = probability
            best_direction = direction
    return best_direction

#compute average cell_count for ML_agent in 100 new grids
def test_ML_agent(NN,has_four_way_vision,original_agent = False):
    start = (0,0)
    end = (30,30)
    trial = 0
    avg_count_original = 0
    while trial < 100:
        print("running trial {}".format(trial))
        grid = generate_gridworld(31, 31, 0.3,start,end)
        test = run_ML_agent(NN, grid, start, end)
        if original_agent:
            if has_four_way_vision:
                original = algorithmA(grid, start, end, [], [], has_four_way_vision)
            else:
                original = inference(grid, start, end, [], [])
            avg_count_original = original/100
    avg_count_ML = test[1]/100
    return [avg_count_ML, avg_count_original]
    
#test
outs = {'down': 0, 'up': 1, 'left': 2, 'right': 3}
conv_outs = list(map(lambda x: outs[x], data_agent_1[1]))
# Convert output data to one-hot encoded form
one_hot = tf.keras.utils.to_categorical(conv_outs, 4).astype(int)
one_hot = [[int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])] for arr in one_hot]
# Train the network with our generated data
for arr in data_agent_1[0]:
    for i, val in enumerate(arr):
        arr[i] = int(arr[i])
input = data_agent_1[0][:30000]
output = one_hot[:30000]

conv_input = []
conv_test_input = []
for i in range(30000):
    curr = np.array(input[i])
    reshaped = curr.reshape(31, 31)
    inp = []
    for row in reshaped:
        r = []
        for val in row:
            r.append(int(val))
        inp.append(r)
    conv_input.append(inp)

for i in range(30000, len(data_agent_1[0])):
    curr = np.array(data_agent_1[0][i])
    reshaped = curr.reshape(31, 31)
    inp = []
    for row in reshaped:
        r = []
        for val in row:
            r.append(int(val))
        inp.append(r)
    conv_test_input.append(inp)
test_data = []
for i in range(1, 4):
    NN = generate_conv_NN(31, [3], (2, 2), 1, (1, 1))
	# Compile the neural network with the parameters given in the example notebook
    NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    NN.fit(conv_input, output, epochs=i)
    res = test_model(NN, conv_test_input, one_hot[30000:])
    test_data.append(res)
print(test_data)

