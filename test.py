import numpy as np
c=5000 * np.random.dirichlet(
                np.array(10 * [0.9]))
print(c)