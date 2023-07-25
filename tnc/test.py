import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Create two sequences
# Trajectory A
A = np.array([1, 3, 4, 9, 8, 2, 1, 5, 7, 3], dtype=float)
# Trajectory B
B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

# Reshape the arrays for use with fastdtw package
A = np.reshape(A, (len(A),1))
B = np.reshape(B, (len(B),1))

# Apply DTW
distance, path = fastdtw(A, B, dist=euclidean)

print(f"DTW distance is {distance}")
print(f"DTW path is {path}")

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create an index array (X-Axis)
idx = np.array(range(max(len(A), len(B))))

# Append zeros if trajectories lengths are different
if len(A) < len(idx):
    A = np.concatenate([A, np.zeros((len(idx) - len(A), 1))])
if len(B) < len(idx):
    B = np.concatenate([B, np.zeros((len(idx) - len(B), 1))])

# Plot the trajectories
ax.plot(idx, A, zs=0, zdir='z', label='A')
ax.plot(idx, B, zs=1, zdir='z', label='B')

# Plot the path - this connects points matched in both trajectories
for p in path:
    ax.plot([p[0], p[1]], [A[p[0]], B[p[1]]], zs=[0, 1], zdir='z', color='r')

ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_zlabel('Sequence')
plt.legend()

plt.show()