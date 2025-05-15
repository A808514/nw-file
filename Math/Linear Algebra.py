# 1- Vectors

# ( Creating a Vector )

# import numpy as np

# list1 = [10, 20, 30]
# list2 = [[10], [20], [30]]

# # Making horizontal vector
# vector1 = np.array(list1)
# # Making vertical vector
# vector2 = np.array(list2)

# print(f"Horizontal Vector: {vector1}")
# print(f"Vertical Vector:\n {vector2}")

# ------------------------------------------------------------------------------

# ( Basic Arithmetic operation )

# import numpy as np

# list1 = [10, 20, 30]
# list2 = [50, 80, 100]

# vector1 = np.array(list1)
# vector2 = np.array(list2)

# # Addition
# print(f"Addition: {vector1 + vector2}")
# # Subtraction
# print(f"Subtraction: {vector1 - vector2}")
# # Multiplying
# print(f"Multiplying: {vector1 * vector2}")
# # Division
# print(f"Division: {vector1 / vector2}")

# ------------------------------------------------------------------------------

# ( Vector Dot Product  )

# import numpy as np

# list1 = [30, 40, 50]
# list2 = [20, 70, 90]

# vector1 = np.array(list1)
# vector2 = np.array(list2)

# # getting dot product of both the vectors
# # a . b = (a1 * b1 + a2 * b2 + a3 * b3)
# # a . b = (a1b1 + a2b2 + a3b3)
# dot_product = np.dot(vector1, vector2)
# print (f"Dot Product: {dot_product}")

# ------------------------------------------------------------------------------

# ( Vector-Scalar Multiplication )

# import numpy as np

# list1 = [10, 20, 30]
# vector1 = np.array(list1)

# scalar = 4

# # getting scalar multiplication value
# # s * v = (s * v1, s * v2, s * v3)
# scalar_mul = vector1 * scalar
# print(f"Scalar multiplication: {scalar_mul}")

# ------------------------------------------------------------------------------

# ( numpy.identity method )
# This method returns Unit Matrix

# import numpy as geek

# a = geek.identity(2, dtype=int)
# print(f"Matrix a: \n {a}")

# b = geek.identity(4, dtype=int)
# print(f"Matrix b: \n {b}")

# ------------------------------------------------------------------------------