
from matplotlib.pyplot import axis
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.expand_dims(a, axis = 2)
print(b, b.shape)
c = np.repeat(b, 3, axis=2)
print(c, c.shape)