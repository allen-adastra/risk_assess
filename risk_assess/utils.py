import math
import numpy as np

def change_frame(position_vector, translation, rotation_matrix):
    position_vector = position_vector.reshape(position_vector.shape[0], 1)
    position_vector += - translation
    position_vector = np.matmul(rotation_matrix, position_vector)
    return position_vector.flatten()

def rotation_matrix(theta):
    C = math.cos(theta)
    S = math.sin(theta)
    return np.array([[C, -S], [S, C]])