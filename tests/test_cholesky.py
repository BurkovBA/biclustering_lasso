import numpy as np
import biclustering_lasso


matrix = np.arange(4).reshape((2, 2))
print(matrix)
print(biclustering_lasso.cholesky(matrix))
