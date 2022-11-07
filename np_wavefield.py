"""
This is attempt to manually implement a propability wave field using numpy.

Cells are one dimension of the field array, each entry representing a state
and it's value the probability that the cell is in that state.

In each time step the new probabilites are calculated by:
    For each cell
        For each state
            calculate the probability by
                summing up relevant rules and for each rule
                    multiplying the probabilites of related cell states

Not sure how the rules are being learned yet.
To test this one simple rules is handcoded:
    Two states.
    trying to imitate waves
"""


import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt


def make_cells_probabilites(a):
    row_sums = a.sum(axis=2)
    new_matrix = a / row_sums[:, :, np.newaxis]
    return new_matrix


def cell_colvolution(array, kernels):
    w = int(kernels[0].shape[0]/2)
    padded_array = np.pad(array, ((1, 1), (1, 1), (0, 0)), "reflect")
    new_array = np.zeros_like(array)
    for x in range(0, array.shape[0]):
        for y in range(0, array.shape[1]):
            for s in range(len(kernels)):
                mult = np.multiply(kernels[s], padded_array[x:x+w*2+1, y:y+w*2+1, :])
                #print(np.sum(mult))
                #foo = np.sum(mult)
                new_array[x, y, s] = np.sum(mult)
    return new_array


def step(f):    
    #fixed: this also convolves into the 3rd channel, meaning it also convolves along the state axis, which I don't want
    #convolved = sps.fftconvolve(f, kernels[0][0], mode='same')
    #convolved += sps.fftconvolve(f, kernels[0][1], mode='same')
    
    convolved = cell_colvolution(f, kernels[0])
    #print("conv", convolved)
    cliped = np.clip(convolved,0,1)#+np.random.rand(10,10,3)*0.5
    #convolved = cell_colvolution(f, kernels[0][1])
    prob = make_cells_probabilites(cliped)
    #print("prob",prob)
    return prob
    return convolved


# Rules expressed as kernels
kernels = [
    [
    # 1 if the surrounding states are in 1, 2 if they are in 2
        # Kernel for state 1
        np.array([[  # upper
            [-.5,1,0],
            [-.5,1,0],
            [-.5,1,0],
        ],[         # middle
            [0,.5,0],
            [0,0,0],
            [0,.5,0]
        ],[         # lower
            [-.5,1,0],
            [-.5,1,0],
            [-.5,1,0],
        ]]),
        # Kernel for state 2
        np.array([[  # upper
            [1,-.5,0],
            [1,-.5,0],
            [1,-.5,0],
        ],[         # middle
            [.5,0,0],
            [0,0,0],
            [.5,0,0],
        ],[         # lower
            [1,-.5,0],
            [1,-.5,0],
            [1,-.5,0],
        ]])
    ]
]


np.random.seed(2)
field = np.random.rand(10,10,3)
field[:,:, 2] = 0
field = make_cells_probabilites(field)

print(field)


# prewarming
for i in range(0):
    field = step(field)
# recording
steps = []
steps.append(field)
for i in range(24):
    steps.append(step(steps[-1]))

fig, ax = plt.subplots(nrows=int(len(steps)/5), ncols=5)
for i, s in enumerate(steps):
    ax[i%5][int(i/5)].imshow(s)
for i, s in enumerate(steps):
    ax[i%5][int(i/5)].imshow(s)
plt.show()

print(steps[-1])
