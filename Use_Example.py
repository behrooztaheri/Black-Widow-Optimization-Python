import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MOBWO import MOBWO

# Viennet function (3 objectives)
def viennet_obj1(x):
    return x[0]**2 + (x[1] - 1)**2

def viennet_obj2(x):
    return x[0]**2 + (x[1] + 1)**2 + 1

def viennet_obj3(x):
    return (x[0] - 1)**2 + x[1]**2 + 2

# Setup
dim = 2
lb = np.array([-3, -3])
ub = np.array([3, 3])

# Run MOBWO
solutions, fitness, history = MOBWO(
    objective_funcs=[viennet_obj1, viennet_obj2, viennet_obj3],
    dim=dim,
    lb=lb,
    ub=ub,
    pop_size=150,
    max_iter=400,
    verbose=True
)

# Plot 3D Pareto front
fitness = np.array(fitness)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fitness[:, 0], fitness[:, 1], fitness[:, 2], 
           c='red', marker='o', alpha=0.6, s=20)
ax.set_xlabel('Objective 1', fontsize=10)
ax.set_ylabel('Objective 2', fontsize=10)
ax.set_zlabel('Objective 3', fontsize=10)
ax.set_title('MOBWO - 3D Pareto Front', fontsize=14)
plt.show()

print(f"\n3D Pareto solutions found: {len(solutions)}")