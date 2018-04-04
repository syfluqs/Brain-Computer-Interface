import matplotlib.pyplot as plt
import numpy as np

npz = np.load('data/sample_1.npz')
plt.plot(npz['data0'])
plt.plot(npz['data1'])
plt.show()