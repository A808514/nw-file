# ((((((((((((((( Prerequisites for Data Analysis )))))))))))))))

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> Creating NumPy Arrays )

# import numpy as np

# # Creating 1D array
# array1 = np.array([1, 2, 3])
#
# # Creating 2D array
# array2 = np.array([[3, 5, 2], [9, 1, 7]])
#
# # Creating 3D array
# array3 = np.array([[[1, 2, 3], [8, 1, 7]], [[5, 2, 6], [9, 2, 4]]])
#
# print(array1)
# print(array2)
# print(array3)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Array Indexing --> Basic Indexing )

# import numpy as np

# array_1d = np.array([10, 20, 30, 40, 50, 60, 70])
# print(f"Single element access: {array_1d[2]}")
# print(f"Negative indexing: {array_1d[-3]}")

# array_2d = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
# print(f"Multidimensional array access: {array_2d[1, 3]}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Array Indexing --> Slicing )

# import numpy as np

# array1 = np.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])
# print(f"Range of elements: {array1[1 : 4]}")

# # all rows, socend column
# print(f"Multidimensional Slicing: {array1[:, 1]}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Array Indexing --> Advanced Indexing )

# import numpy as np

# array1 = np.array([1, 2, 3, 4, 5, 6, 7])

# # Integer array indexing
# indices = np.array([1, 3, 5])
# print(f"Integer array indexing: {array1[indices]}")

# # Boolean array indexing
# condition = (array1 % 2 != 0)
# print(f"Boolean array indexing: {array1[condition]}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Basic Operations --> Element-wise Operation )

# import numpy as np

# arr1 = np.array([1, 2, 3, 4])
# arr2 = np.array([5, 6, 7, 8])

# # Addition
# print(arr1 + arr2)

# # Subtraction
# print(arr1 - arr2)

# # Multiplication
# print(arr1 * arr2)

# # devision
# print(arr1 / arr2)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Basic Operations --> Unary Operation )

# import numpy as np

# arr = np.array([-4, -1, 0, 1, 4])

# # Applying a unary operation: absolute value
# result = np.abs(arr)
# print(f"Absolute value: {result}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Basic Operations --> Binary Operators  )

# import numpy as np

# arr1 = np.array([1, 2, 3, 4])
# arr2 = np.array([5, 6, 7, 8])

# # Applying a binary operation: addition
# addition = np.add(arr1, arr2)
# print(f"Addition Result: {addition}")

# # Applying a binary operation: multiplication
# multiplication = np.dot(arr1, arr2) # x1*y1 + x2*y2 + x3*y3 + ...
# print(f"Multiplication Result: {multiplication}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Basic Operations --> NumPy ufuncs  )

# import numpy as np

# arr1 = np.array([np.pi, np.pi/2, np.pi/4])
# print(f"Sin values of array elements: {np.sin(arr1)}")

# # exponential values
# arr2 = np.array([1, 2, 3, 4])
# print(f"Exponential values of array elements: {np.exp(arr2)})")

# # square root of array values
# print(f"Square root of array values: {np.sqrt(arr2)}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Introduction to NumPy -> NumPy Basic Operations --> NumPy Sorting Arrays  )

# import numpy as np

# dtypes = [("name", "S10"), ("grad_year", int), ("cgpa", float)]

# values = [
#     ("Abdallah", 2028, 10),
#     ("Ahmed", 2030, 13),
#     ("Sayed", 2022, 4),
#     ("Nader", 2025, 8),
# ]

# arr1 = np.array(values, dtype= dtypes)
# print(f"Array sorted by names: {np.sort(arr1, order='name')}")
# print(f"Array sorted by grad_year and then cgpa: {np.sort(arr1, order= ['grad_year', 'cgpa'])}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> One Dimensional Array )

# import numpy as np

# lst = [1, 2, 3, 4, 5, 6]

# # creating 1 dimensional array
# arr = np.array(lst)

# print(f"list in python: {lst}")
# print(f"Numpy array in python: {arr}")

# print(type(lst))
# print(type(arr))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Multi-Dimensional Array )

# import numpy as np

# list1 = np.array([1, 2, 3, 4, 5])
# list2 = np.array([10, 11, 12, 13, 14])
# list3 = np.array([50, 40, 30, 20, 10])

# # creating multi-dimensional array
# arr = np.array([list1, list2, list3])
# print(f"Numpy multi-dimensional array in python:\n{arr}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Shape )

# import numpy as np

# list1 = np.array([1, 2, 3, 4, 5])
# list2 = np.array([10, 11, 12, 13, 14])
# list3 = np.array([50, 40, 30, 20, 10])

# arr = np.array([list1, list2, list3])
# print(arr.shape)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Data type objects (dtype) )

# import numpy as np

# arr1 = np.array([1, 2, 3, 4, 5])
# arr2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# print(arr1.dtype)
# print(arr2.dtype)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Some different way of creating Numpy Array --> numpy.fromiter() )
# The fromiter() function create a new one-dimensional array from an iterable object.

# # ( Example 1 )
# import numpy as np

# iterable = [(i ** 2) for i in range(10)]

# arr = np.fromiter(iterable)
# print(f"fromiter() array : {arr}")

# # ( Example 2 )
# import numpy as np

# word = "Abdallah"
# arr = np.fromiter(word, dtype="U2")

# print(f"formiter() array : {arr}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Some different way of creating Numpy Array --> numpy.arange() )
# This is an inbuilt NumPy function that returns evenly spaced values within a given interval.

# import numpy as np

# arr = np.arange(1, 20, 2, dtype="int32")
# print(arr)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Some different way of creating Numpy Array --> numpy.linspace() )
# This function returns evenly spaced numbers over a specified between two limits.
# زي متتالية حسابية من اول الرقم االي حد نهاية ارقم اللي انت عاوزه بتحط عدد الارقام اللي انت عاوزه

# import numpy as np

# arr1 = np.linspace(5 , 15, 10) # this will print defult (floats)
# arr2 = np.linspace(5 , 15, 10, dtype=int) # this will print integer
# print(arr1)
# print(arr2)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Some different way of creating Numpy Array --> numpy.empty() )
# This function create a new array of given shape and type, without initializing value.

# import numpy as np

# arr1 = np.empty([4, 3], dtype=int, order="F")
# arr2 = np.empty([4, 3], order="F")
# print(arr1)
# print(arr2)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Some different way of creating Numpy Array --> numpy.ones() )
# This function is used to get a new array of given shape and type, filled with ones(1).

# import numpy as np

# arr = np.ones([4, 3], dtype=int)
# print(arr)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Basics of NumPy Arrays --> Some different way of creating Numpy Array --> numpy.zeros() )

# import numpy as np

# arr = np.zeros([4, 3], dtype=int)
# print(arr)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Data types and type casting --> Checking the Data Type of NumPy Array )

# import numpy as np

# arr = np.array([10, 20, 30, 40, 50])
# data_type = arr.dtype
# print(data_type)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Data types and type casting --> Create Arrays With a Defined Data Type )

# import numpy as np

# arr1 = np.array([1, 2, 3, 4, 5], dtype=float)
# arr2 = np.ones([3, 3], dtype=np.complex128)
# arr3 = np.zeros([2, 2], dtype=int)
# arr4 = np.zeros([4, ], dtype=bool)
# print(arr1)
# print(arr2)
# print(arr3)
# print(arr4)
# print(arr1.dtype)
# print(arr2.dtype)
# print(arr3.dtype)
# print(arr4.dtype)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Data types and type casting --> Convert Data Type of NumPy Arrays )
# We can convert data type of an arrays from one type to another

# import numpy as np

# arr1 = np.array([1.2, 2.5, 3.7])

# # convert array to another data type
# arr2 = arr1.astype(int)
# print(arr2)
# print(arr2.dtype)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Accessing 1D Array )

# import numpy as np

# arr = np.array([10, 20, 30, 40, 50])
# print(arr[2])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Accessing 2D Array )

# import numpy as np

# arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
# print(arr[1, 2])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Accessing 3D Array )

# import numpy as np

# arr = np.array([[[10, 20, 30], [40, 50, 60]], [[30, 50, 90], [70, 80, 90]]])
# print(arr[1, 1, 0])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Slicing Arrays )

# ( Example 1: 1 dimensional )
# import numpy as np

# arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# print(arr[1: 6: 2])

# ( Example 2: multi-dimensional )
# import numpy as np

# arr = np.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100], [110, 120, 130, 140, 150]])
# # from index 1 to index 2 print from index 0 to index 2
# print(arr[1 : 3, 0 : 3])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Boolean Indexing )

# ( Example 1 )
# import numpy as np

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# condition = arr % 2 == 0

# print(arr[condition])

# ( Example 2 )
# import numpy as np

# arr = np.array([1, 2, 3, 4, 5, 6 , 7, 8, 9, 10])

# # you must put the condition if more than one condition in tuple to can run the code
# # in this type of condition should use (& instead of and) (| instead of or)
# condition = (arr > 2) & (arr < 9)
# print(arr[condition])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Ellipsis (...) in Indexing )
# The ellipsis (...) is a shorthand for selecting all dimensions not explicitly mentioned

# import numpy as np

# arr = np.random.rand(3, 3, 5)
# print(arr[..., 2])
# print(arr[Ellipsis, 2])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Using np.newaxis )

# import numpy as np

# arr = np.array([1, 2, 3, 4, 5])

# # Add a new axis to convert the 1D array into a 2D column vector
# # arr[:, np.newaxis] inserts a new axis along the second dimension
# print(arr[:, np.newaxis])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Accessing and Modifying Data - Indexing and slicing --> Modifying Array Elements  )

# import numpy as np

# arr = np.array([1, 2, 3, 4, 5])
# arr[1:4] = 10
# print(arr)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Broadcasting Array in Single Value and 1D Addition )

# import numpy as np

# arr = np.array([1, 2, 3, 4, 5])
# result = arr + 2
# print(result)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Broadcasting Array in 1D and 2D Addition )

# import numpy as np

# arr1 = np.array([1, 2, 3])
# arr2 = np.array([[4, 5, 6], [9, 10, 11]])
# result = arr1 + arr2
# print(result)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Using Broadcasting for Matrix Multiplication )

# import numpy as np

# arr1 = np.array([10, 11, 12])
# arr2 = np.array([[13, 14, 15], [16, 17, 18]])
# result = arr1 * arr2
# print(result)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Scaling Data with Broadcasting )

# import numpy as np

# food_data = np.array(
#     [[0.8, 1.5, 0.4], [2.4, 1.8, 6.4], [7.2, 4.3, 3.9], [10.7, 5.5, 7.4]]
# )
# caloric_values = np.array([3, 6, 4])
# calorie_brakdown = food_data * caloric_values
# print(calorie_brakdown)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Adjusting Temperature Data Across Multiple Locations )

# import numpy as np

# temperatures = np.array([[10, 15, 20, 25, 30], [35, 40, 45, 50, 55], [60, 65, 70, 75, 80]])
# corrections = np.array([1.5, -0.5, 2])

# adjusted_temperatures = temperatures + corrections[:, np.newaxis]
# print(adjusted_temperatures)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Normalizing Image Datas )

# import numpy as np

# image = np.array([[10, 15, 20], [35, 40, 45], [60, 65, 70]])
# mean = image.mean(axis=0) # Mean per column
# std = image.std(axis=0) # Standard deviation per column

# normalized_image = (image - mean) / std
# print(normalized_image)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Broadcasting --> Centering Data in Machine Learning )

# import numpy as np

# data = np.array([[10, 15, 20], [25, 30, 35], [40, 45, 50]])
# mean = data.mean(axis=0, dtype=int)
# centerd_data = data - mean
# print(centerd_data)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Saving and loading NumPy Arrays -->  Saving a 3D Numpy Array as a Text File )

# import numpy as np

# arr = np.random.rand(5, 4, 3)
# arr_reshape = arr.reshape(arr.shape[0], -1)

# np.savetxt("main.txt", arr_reshape)
# loaded_arr = np.loadtxt("main.txt")

# load_original_arr = loaded_arr.reshape(
#     loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2]
# )
# print(f"shape of arr: {arr.shape}")
# print(f"shape of load_original_arr: {load_original_arr.shape}")

# if (arr == load_original_arr).all():
#     print("Yes, both the arrays are the same")
# else:
#     print("No, both arrays are different")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> NumPy for Numerical Operations --> Saving and loading NumPy Arrays -->  Saving and loading the 3D arrays(reshaped) into CSV files )

# import numpy as np

# arr = np.random.rand(5, 4, 3)
# reshape_arr = arr.reshape(arr.shape[0], -1)

# # Save the 2D array to a CSV file
# np.savetxt("main.csv", reshape_arr, delimiter=",")

# # Load the 2D array from the CSV file
# loaded_arr = np.loadtxt("main.csv", delimiter=",")

# load_original_arr = loaded_arr.reshape((arr.shape[0], arr.shape[1], arr.shape[2]))

# if np.array_equal(load_original_arr, arr):
#     print("Yes, both the arrays are the same")
# else:
#     print("No, both arrays are not the same")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Pandas Read CSV in Python )

# import pandas as pd

# # read csv file
# df = pd.read_csv("data.csv")
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Read specific columns using read_csv )

# import pandas as pd

# # read specific columns using read_csv
# df = pd.read_csv("data.csv", usecols=["Pulse", "Calories"])
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Setting an Index Column )
# The index_col parameter sets one or more columns as the DataFrame index, making the specified column(s) act as row labels for easier data referencing.

# import pandas as pd

# df = pd.read_csv("data.csv", index_col=["Pulse", "Calories"])
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Handling Missing Values Using read_csv  )
# The na_values parameter replaces specified strings (e.g., "N/A", "Unknown") with NaN

# import pandas as pd

# df = pd.read_csv("data.csv", na_values = ["N/A", "Unknown"])
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Reading CSV Files with Different Delimiters  )

# import pandas as pd

# data = """totalbill_tip, sex:smoker, day_time, size
# 16.99, 1.01:Female|No, Sun, Dinner, 2
# 10.34, 1.66, Male, No|Sun:Dinner, 3
# 21.01:3.5_Male, No:Sun, Dinner, 3
# 23.68, 3.31, Male|No, Sun_Dinner, 2
# 24.59:3.61, Female_No, Sun, Dinner, 4
# 25.29, 4.71|Male, No:Sun, Dinner, 4"""

# with open("main.csv", "w") as file:
#     file.write(data)

# df = pd.read_csv("main.csv", sep= "[:, |_]", engine="python")
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Using nrows in read_csv() )

# import pandas as pd

# df = pd.read_csv("data.csv", nrows=3)
# print(df)
# print(df.head(3))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Using skiprows in read_csv() )

# import pandas as pd

# df = pd.read_csv("data.csv")
# print(f"Data File: {df}")

# skiprows_data = pd.read_csv("data.csv", skiprows=[1, 3, 5, 7, 9])
# print(f"Dataset after skipping rows: {skiprows_data}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Parsing Dates (parse_dates) )
# The parse_dates parameter converts date columns into datetime objects, simplifying operations like filtering, sorting, or time-based analysis.

# import pandas as pd

# df = pd.read_csv("data.csv", parse_dates=["Duration"])
# print(df.info())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Read Dataset with Pandas --> Loading a CSV Data from a URL )

# import pandas as pd

# url = "https://media.geeksforgeeks.org/wp-content/uploads/20241121154629307916/people_data.csv"
# df = pd.read_csv(url)
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Save DataFrame as CSV file for further use --> create a sample data frame & export this DataFrame as a CSV file )

# import pandas as pd

# data = {"Name": ["Abdallah", "Ahmed", "Mohammed", "Tolba"], "Numbers": [90, 95, 20, 30]}

# df = pd.DataFrame(data)
# df.to_csv("data1.csv")

# df_csv = pd.read_csv("data1.csv")
# print(df_csv)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Save DataFrame as CSV file for further use --> Include index number )

# import pandas as pd

# data = {"Name": ["Abdallah", "Ahmed", "Mohammed", "Tolba"], "Numbers": [90, 95, 20, 30]}

# df = pd.DataFrame(data)
# df.to_csv("data1.csv", index=False)

# df_csv = pd.read_csv("data1.csv")
# print(df_csv)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Save DataFrame as CSV file for further use --> Export only selected columns )

# import pandas as pd

# data = {"Name": ["Abdallah", "Ahmed", "Mohammed", "Tolba"], "Numbers": [90, 95, 20, 30]}

# df = pd.DataFrame(data)
# df.to_csv("data1.csv", index=False, columns=["Name"])

# df_csv = pd.read_csv("data1.csv")
# print(df_csv)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Save DataFrame as CSV file for further use --> Export header )

# import pandas as pd

# data = {"Name": ["Abdallah", "Ahmed", "Mohammed", "Tolba"], "Numbers": [90, 95, 20, 30]}

# df = pd.DataFrame(data)
# df.to_csv("data1.csv", index=False, header=False)

# df_csv = pd.read_csv("data1.csv")
# print(df_csv)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Save DataFrame as CSV file for further use --> Handle NaN )

# import pandas as pd

# data = {"Name": ["Abdallah", "Ahmed", "Mohammed", ""], "Numbers": [90, "", 20, 30]}

# df = pd.DataFrame(data)
# df.to_csv("data1.csv", index=False, na_rep= "nothing")

# df_csv = pd.read_csv("data1.csv")
# print(df_csv)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Save DataFrame as CSV file for further use --> Handle NaN )

# import pandas as pd

# data = {"Name": ["Abdallah", "Ahmed", "Mohammed", "Tolba"], "Numbers": [90, 95, 20, 30]}

# df = pd.DataFrame(data)
# df.to_csv("data1.csv", index=False, sep="@")

# df_csv = pd.read_csv("data1.csv")
# print(df_csv)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Parsing in Pandas JSON Dataset Example  )

# import pandas as pd

# df = pd.DataFrame([[1, 2], [4, 5]],
#                 index=["row1", "row2"],
#                 columns=["col1", "col2"])

# print(df.to_json(orient="columns"))
# print(df.to_json(orient="index"))
# print(df.to_json(orient="records"))
# print(df.to_json(orient="split"))
# print(df.to_json(orient="table"))
# print(df.to_json(orient="values"))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Read the JSON File directly from Dataset )

# import pandas as pd

# df = pd.read_json('http://api.population.io/1.0/population/India/today-and-tomorrow/?format = json')
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Nested JSON Parsing with Pandas )

# import pandas as pd
# import requests

# url = 'https://raw.githubusercontent.com/a9k00r/python-test/master/raw_nyc_phil.json'
# response = requests.get(url)
# d = response.json()  # Load JSON from URL

# # Use the updated import method
# nycphil = pd.json_normalize(d['programs'])

# print(nycphil.head(3))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> JSON Normalization For Works Data )

# import pandas as pd
# import requests

# url = "https://raw.githubusercontent.com/a9k00r/python-test/master/raw_nyc_phil.json"
# response = requests.get(url)
# d = response.json()  # Load JSON from URL

# works_data = pd.json_normalize(
#     data=d["programs"],
#     record_path="works",
#     meta=["id", "orchestra", "programID", "season"],
# )
# print(works_data.head(3))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> JSON Normalization For Soloists Data )

# import pandas as pd
# import requests

# url = "https://raw.githubusercontent.com/a9k00r/python-test/master/raw_nyc_phil.json"
# response = requests.get(url)
# d = response.json()  # Load JSON from URL

# soloist_data = pd.json_normalize(
#     data=d["programs"], record_path=["works", "soloists"], meta=["id"]
# )

# print(soloist_data.head(3))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Working with Excel files )

# import pandas as pd

# df = pd.read_excel("Book 1.xlsx")
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Loading multiple sheets using Concat() method )

# import pandas as pd

# file_name = "Book 1.xlsx"

# sheet1 = pd.read_csv(file_name, sheet_name=0, index_col=0)
# sheet2 = pd.read_csv(file_name, sheet_name=1, index_col=0)

# new_data = pd.concat([sheet1, sheet2])
# print(new_data)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Head() and Tail() methods in Pandas )

# import pandas as pd

# file = pd.read_excel("Book 1.xlsx")
# print(file.head())
# print(file.tail())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Shape() method )

# import pandas as pd

# df = pd.read_excel("Book 1.xlsx")
# print(df.shape)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Sort_values() method in Pandas )
# If any column contains numerical data, we can sort that column using the sort_values() method in pandas

# import pandas as pd

# df = pd.read_excel("Book 1.xlsx")
# sorted_data = df.sort_values(["Duration"], ascending=False) # Descending order
# sorted_data = df.sort_values(["Duration"]) # Ascending order
# print(sorted_data)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Reading from JSON files into Pandas DataFrame --> Pandas Describe() method )

# import pandas as pd

# df = pd.read_excel("Book 1.xlsx")
# print(df.describe())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Find missing values in the dataset )

# import pandas as pd

# df = pd.read_csv("data.csv")
# print(df.isnull())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Find the number of missing values in the dataset )

# import pandas as pd

# df = pd.read_csv("data.csv")
# print(df.isnull().sum())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Removing missing values )
# The data_frame.dropna( ) function removes columns or rows which contains atleast one missing values.

# import pandas as pd

# df = pd.read_csv("data.csv")
# df_drop = df.dropna()
# print(df_drop)
# print(df_drop.isnull().sum())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Removing missing values )
# We can fill null values using data_frame.fillna( ) function.

# import pandas as pd

# df = pd.read_csv("data.csv")
# df = df.fillna(100)
# print(df)

# # to fill a specific column
# df["Calories"] = df["Calories"].fillna(100)
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Removing rows )
# By using the drop(index) function we can drop the row at a particular index. If we want to replace the data_frame with the row removed then add inplace = True in the drop function.

# import pandas as pd

# df = pd.read_csv("data.csv")
# df = df.drop(3)
# print(df.head())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Renaming rows )

# import pandas as pd

# df = pd.read_csv("data.csv")
# df = df.rename({0 : "Zero", 1 : "First", 2 : "Second", 3 : "Third", 4 : "Fourth"})
# print(df.head())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Adding new columns )

# import pandas as pd

# df = pd.read_csv("data.csv")
# df["New Column"] = 1
# print(df.head())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Sort by multiple columns )

# import pandas as pd

# df = pd.read_csv("data.csv")
# df = df.sort_values(["Duration", "Pulse", "Calories"])
# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> Merge Data Frames )

# import pandas as pd

# df1 = pd.DataFrame(
#     {
#         "Name": ["Jeevan", "Raavan", "Geeta", "Bheem"],
#         "Age": [25, 24, 52, 40],
#         "Qualification": ["Msc", "MA", "MCA", "Phd"],
#     }
# )

# df2 = pd.DataFrame(
#     {
#         "Name": ["Jeevan", "Raavan", "Geeta", "Bheem"],
#         "Salary": [100000, 50000, 20000, 40000],
#     }
# )

# data_merge = pd.merge(df2, df1)
# print(data_merge)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Processing with Pandas --> By defining a function beforehand )

# import pandas as pd

# def fun(value):
#     return("Yes" if value > 250 else "No")

# df = pd.read_csv("data.csv")
# df["Good calories"] = df["Calories"].apply(fun)
# print(df.head(10))

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Slicing rows with pandas Indexing --> Slicing Using iloc )

# import pandas as pd

# data = [
#     ["M.S.Dhoni", 36, 75, 5428000],
#     ["A.B.D Villers", 38, 74, 3428000],
#     ["V.Kohli", 31, 70, 8428000],
#     ["S.Smith", 34, 80, 4428000],
#     ["C.Gayle", 40, 100, 4528000],
#     ["J.Root", 33, 72, 7028000],
#     ["K.Peterson", 42, 85, 2528000],
# ]

# df = pd.DataFrame(data, columns=["Name", "Age", "Weight", "Salary"])

# # A. slicing rows in dataframe
# df1 = df.iloc[0:4]
# print(df1)

# # B. slicing columns in dataframe
# df2 = df.iloc[:, 0:2]
# print(df2)

# # C. Selecting a Specific Cell  in Dataframe in Python
# df3 = df.iloc[4, 2]  # row 4, column 2
# print(f"Specific cell value: {df3}")

# # D. Using Boolean Conditions in Dataframe in Python
# condition = df["Age"] > 35
# df4 = df[condition].iloc[:, :]
# print(f"Filtered data based on age:\n{df4}")
# print(df[condition])

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Slicing rows with pandas Indexing --> Slicing Using loc[] )

# import pandas as pd

# data = [
#     ["M.S.Dhoni", 36, 75, 5428000],
#     ["A.B.D Villers", 38, 74, 3428000],
#     ["V.Kohli", 31, 70, 8428000],
#     ["S.Smith", 34, 80, 4428000],
#     ["C.Gayle", 40, 100, 4528000],
#     ["J.Root", 33, 72, 7028000],
#     ["K.Peterson", 42, 85, 2528000],
# ]

# df = pd.DataFrame(data, columns=["Name", "Age", "Weight", "Salary"])

# # set index as labels manually
# df_custom = df.set_index("Name")
# print(df_custom)

# # A. Slicing Rows in Dataframe in Python
# sliced_row_custom = df_custom.loc["V.Kohli":"J.Root"]
# print(sliced_row_custom)

# # B. Selecting Specified cell in Dataframe in Python
# specific_cell_value = df_custom.loc["S.Smith", "Salary"]
# print(f"Value of specific cell: {specific_cell_value}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Aggregation and Grouping --> Summarize )

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "id": [7058, 4511, 7014, 7033],
#         "name": ["sravan", "manoj", "aditya", "bhanu"],
#         "Maths_marks": [99, 97, 88, 97],
#         "Chemistry_marks": [89, 99, 99, 90],
#         "telugu_marks": [99, 97, 88, 80],
#         "hindi_marks": [99, 97, 56, 67],
#         "social_marks": [79, 97, 78, 90]
#     }
# )

# # describing the data frame --> dataframe_name.describe()
# print(df.describe())

# # finding unique values --> dataframe[column_name].unique()
# print(df["Maths_marks"].unique())

# # counting unique values --> dataframe_name['column_name].nunique()
# print(df["Maths_marks"].nunique())

# # display the columns in the data frame --> dataframe.info()
# print(df.columns)

# # information about dataframe --> dataframe.columns
# print(df.info())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Aggregation and Grouping --> Aggregation )

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "id": [7058, 4511, 7014, 7033],
#         # "name": ["sravan", "manoj", "aditya", "bhanu"],
#         "Maths_marks": [80, 97, 88, 90],
#         "Chemistry_marks": [89, 99, 99, 90],
#         "telugu_marks": [90, 97, 88, 80],
#         "hindi_marks": [98, 97, 56, 67],
#         "social_marks": [79, 97, 78, 90]
#     }
# )

# # computing sum
# print(df.sum())

# # computing minimum values
# print(df.min())

# # computing maximum values
# print(df.max())

# # computing mean
# print(df.mean()) # code doesn't run because there is an object in data, if there isn't an object the code will run

# # finding count
# print(df.count())

# # computing standard deviation
# print(df.std()) # code doesn't run because there is an object in data, if there isn't an object the code will run

# # computing variance
# print(df.var()) # code doesn't run because there is an object in data, if there isn't an object the code will run

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Data Aggregation and Grouping --> Grouping )

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "id": [7058, 4511, 7014, 7033],
#         "name": ["sravan", "manoj", "aditya", "bhanu"],
#         "Maths_marks": [99, 97, 88, 90],
#         "Chemistry_marks": [89, 99, 99, 90],
#         "telugu_marks": [99, 97, 88, 80],
#         "hindi_marks": [99, 97, 56, 67],
#         "social_marks": [79, 97, 78, 90]
#     }
# )

# # group by name
# print(df.groupby("name").first())

# # group by name with social_marks sum
# print(df.groupby("name")["social_marks"].sum())
# print(df.groupby("name")["social_marks"].first())

# # group by name with maths_marks count
# print(df.groupby("name")["Maths_marks"].count())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Working with Date and Time )

# import pandas as pd

# df = pd.DataFrame()

# df["time"] = pd.date_range("2/18/2025", periods=6, freq="2h")

# df["Year"] = df["time"].dt.year
# df["Month"] = df["time"].dt.month
# df["Day"] = df["time"].dt.day
# df["hour"] = df["time"].dt.hour
# df["Minute"] = df["time"].dt.minute

# print(df)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Working with Date and Time --> Convert strings to Timestamps )

# import numpy as np
# import pandas as pd

# dt_strings = np.array(
#     ["04-03-2019 12:35 PM", "22-06-2017 11:01 AM", "05-09-2009 07:09 PM"]
# )

# timestamps = [
#     pd.to_datetime(date, format="%d-%m-%Y %I:%M %p", errors="coerce")
#     for date in dt_strings
# ]

# print(timestamps)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Working with Date and Time --> Change the pattern of Timestamps )

# import pandas as pd

# df = pd.DataFrame()

# df["time"] = pd.date_range("3/5/2019", periods=6, freq="2H")
# print(f"Old Pattern: {df['time']}")

# df["new_time"] = df["time"].dt.strftime("%d-%B")
# print(f"New Pattern: {df['new_time']}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Working with Date and Time --> Extract Days Of the Week from the given Date )

# import pandas as pd

# dates = pd.Series(pd.date_range("2/18/2025", periods=6, freq="ME"))
# print(dates)
# print(dates.dt.day_name())

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Pandas for Data Manipulation --> Working with Date and Time --> Extract Data in Date and Time Ranges )

# ( Method #1: If the dataset is not indexed with time. )
# import pandas as pd

# df = pd.DataFrame()
# df["date"] = pd.date_range("2/18/2025", periods=1000, freq="h")

# print(df.head())

# # Select observations between two datetimes
# x = df[(df["date"] > "2025-02-18 01:00:00") & (df["date"] < "2025-02-18 10:00:00")]
# print(x)

# ( Method #2: If the dataset is indexed with time )
# import pandas as pd

# df = pd.DataFrame()
# df["date"] = pd.date_range("2/18/2025", periods=1000, freq="h")

# df = df.set_index(df["date"])
# print(df.head())

# x = df.loc["2025-02-18 01:00:00" : "2025-02-18 10:00:00"]
# print(x)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Introduction to Matplotlib --> Matplotlib Pyplot )

# import matplotlib.pyplot as plt

# x = [0, 2, 4, 6, 8]
# y = [0, 4, 16, 36, 64]

# fig, ax = plt.subplots()
# ax.plot(x, y, marker='o', label="Data Points")
# ax.set_title("Basic components of matplotlib figure")
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Pyplot in Matplotlib --> Using plot() for Line Graphs )

# import matplotlib.pyplot as plt

# x = [0, 2, 4, 6, 8]
# y = [0, 4, 16, 36, 64]
# plt.plot(x, y)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Pyplot in Matplotlib --> Using pie() for Pie Charts )

# import matplotlib.pyplot as plt

# labels = ['Python', 'Java', 'C++', 'JavaScript']
# sizes = [40, 30, 20, 10]
# plt.pie(sizes, labels=labels, autopct="%1.1f%%")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Pyplot in Matplotlib --> UPlotting Bar Charts Using bar() )

# import matplotlib.pyplot as plt

# categories = ["A", "B", "C", "D"]
# values = [3, 7, 2, 5]
# plt.bar(categories, values)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Pyplot in Matplotlib --> Using scatter() for Scatter Plots )

# import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5]
# y = [5, 7, 9, 11, 13]
# plt.scatter(x, y)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Pyplot in Matplotlib --> Working with Histograms Using hist() )

# import matplotlib.pyplot as plt
# import numpy as np

# data = np.random.randn(1000)
# plt.hist(data, bins=30)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Matplotlib – Axes Class --> axes() function )
# axes([left , bottom, width, height])

# import matplotlib.pyplot as plt
# ax = plt.axes([0.1, 0.1, 0.8, 0.8])
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Matplotlib – Axes Class --> add_axes() function )
# add_axes([left, bottom, width, height])

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Matplotlib – Axes Class --> ax.legend() function )
# ax.legend(handles, labels, loc)

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes([0.1, 0.1, 0.8, 0.8])
# ax.legend(labels=("label1", "label2"), loc="upper left")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Matplotlib – Axes Class --> ax.plot() function )
# Syntax :  plt.plot(X, Y, ‘CLM’)

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-np.pi, np.pi, 20)
# s = np.sin(x)
# c = np.cos(x)

# ax = plt.axes([0.1, 0.1, 0.8, 0.8])
# ax1 = ax.plot(x, s, "bs:")
# ax2 = ax.plot(x, c, "ro-")

# ax.legend(labels=("Sin function", "Cos function"), loc="upper left")
# ax.set_title("Trigonometric function")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Matplotlib for 3D Plotting )

# ( Example 1 )
# import matplotlib.pyplot as plt
# import numpy as np

# z = np.random.randint(100, size=(50))
# x = np.random.randint(80, size=(50))
# y = np.random.randint(60, size=(50))

# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")

# ax.scatter3D(x, y, z, color = "green")
# plt.title("Simple 3D scatter plot")

# plt.show()

# ( Example 2 )
# import matplotlib.pyplot as plt
# import numpy as np

# z = 4 * np.tan(np.random.randint(10, size=500)) + np.random.randint(100, size=500)
# x = 4 * np.cos(z) + np.random.normal(size=500)
# y = 4 * np.sin(z) + 4 * np.random.normal(size=500)

# fig = plt.figure(figsize=(16, 9))
# ax = plt.axes(projection="3d")

# ax.grid(b=True, color="red", linestyle="-.", linewidth=0.3, aplha=0.2)

# # Creating color map
# cy_map = plt.get_cmap("hsv")

# scatter = plt.scatter(x, y, z, alpha=0.8, c=(x + y + z), cmap=cy_map, marker="^")
# ax.set_xlabel("X-axis", fontweight="bold")
# ax.set_ylabel("Y-axis", fontweight="bold")
# ax.set_zlabel("Z-axis", fontweight="bold")

# fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Matplotlib for Data Visualization --> Exploratory Data Analysis with matplotlib )

# # Step 1: Importing Required Libraries
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings as wr

# wr.filterwarnings("ignore")

# # Step 2: Reading Dataset
# df = pd.read_csv("WineQT.csv")
# print(df.head())

# # Step 3: Analyzing the Data
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.columns.to_list())

# # Step 4 : Checking Missing Values
# print(df.isnull().sum())

# # Step 5 : Checking for the duplicate values
# print(df.nunique())

# # Step 6: Univariate Analysis for (analyzing the distribution, central tendency, and spread of data effectively)
# # ( 1. Bar Plot for evaluating the count of the wine with its quality rate. )
# quality_counts = df["quality"].value_counts()
# plt.figure(figsize=(8, 6))
# plt.bar(quality_counts.index,quality_counts, color="darkblue")
# plt.title("Count plot of quality")
# plt.xlabel("Quality")
# plt.ylabel("Count")
# plt.show()

# # ( 2. Kernel density plot for understanding variance in the dataset )
# sns.set_style("darkgrid")
# numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# plt.figure(figsize=(14, len(numerical_columns) * 3))
# for idx, feature in enumerate(numerical_columns, 1):
#     plt.subplot(len(numerical_columns), 2, idx)
#     sns.histplot(df[feature], kde=True)
#     plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

# plt.tight_layout()
# plt.show()

# # ( 3. Swarm Plot for showing the outlier in the data )
# plt.figure(figsize=(10, 8))

# sns.swarmplot(x="quality", y="alcohol", data=df, palette="viridis")
# plt.title("Swarm Plot for Quality and Alcohol")
# plt.xlabel("Quality")
# plt.ylabel("Alcohol")
# plt.show()

# # Step 6: Bivariate Analysis for (understanding variable interactions and correlations effectively)
# # ( Pair Plot for showing the distribution of the individual variables )
# sns.set_palette("Pastel1")
# plt.figure(figsize=(10, 6))
# sns.pairplot(df)
# plt.suptitle("Pair Plot for DataFrame")
# plt.show()

# # ( Violin Plot for examining the relationship between alcohol and Quality )
# df["quality"] = df["quality"].astype("str")
# plt.figure(figsize=(10, 8))
# sns.violinplot(
#     x="quality",
#     y="alcohol",
#     data=df,
#     palette={
#         "3": "lightcoral",
#         "4": "lightblue",
#         "5": "lightgreen",
#         "6": "gold",
#         "7": "lightskyblue",
#         "8": "lightpink",
#     },
#     alpha=0.7
# )
# plt.title('Violin Plot for Quality and Alcohol')
# plt.xlabel('Quality')
# plt.ylabel('Alcohol')
# plt.show()

# # ( Box Plot for examining the relationship between alcohol and Quality )
# sns.boxplot(x="quality", y="alcohol", data=df)
# plt.show()

# # Step 7: Multivariate Analysis for (understanding complex relationships and patterns among multiple variables effectively)
# # ( Correlation Matrix for examining the correlation )
# plt.figure(figsize=(15, 10))
# sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Pastel2", linewidths=2)
# plt.title("Correlation Heatmap")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Introduction to Seaborn --> Some basic plots using seaborn )

# ( 1. Histplot )
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set_theme(style="dark")

# # Generate a random univariate dataset
# rs = np.random.RandomState(10)
# d = rs.normal(size=100)

# sns.histplot(d, kde=True, color="m")
# plt.show()

# ( 2. Lineplot )
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set_style(style="white")
# fmri = sns.load_dataset("fmri")

# sns.lineplot(x="timepoint", y="signal", hue="region", style="event", data=fmri)
# plt.show()

# ( 3. Lmplot )
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set_style(style="dark")
# df = sns.load_dataset("anscombe")
# sns.lmplot(x="timepoint", y="signal", data=df)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Types Of Seaborn Plots --> Relational Plots in Seaborn )

# ( 1. Scatter Plot )
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = sns.load_dataset("tips")
# sns.scatterplot(data=data, x="total_bill", y="tip")
# plt.show()

# ( 2. Line plot )
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = sns.load_dataset("tips")
# sns.lineplot(data=data, x="size", y="tip")
# plt.show()

# ( 3. Relational Plot (relplot) )
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = sns.load_dataset("tips")
# sns.relplot(data=data, x="total_bill", y="tip", hue="smoker")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Types Of Seaborn Plots --> Categorical Plots in Seaborn )

# ( 1. Bar Plot (barplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.barplot(data=data, x="day", y="total_bill")
# plt.show()

# ( 2. Count Plot (countplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.countplot(data=data, x="day")
# plt.show()

# ( 3. Box Plot (boxplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.boxplot(data=data, x="day", y="total_bill")
# plt.show()

# ( 4. Violin Plot (violinplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.violinplot(data=data, x="day", y="total_bill")
# plt.show()

# ( 6. Swarm Plot )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.swarmplot(data=data, x="day", y="total_bill")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Types Of Seaborn Plots --> Distribution Plots in Seaborn )

# ( 1. Histogram (histplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.histplot(data=data, x="total_bill")
# plt.show()

# ( 2. Kernel Density Estimate Plot (kdeplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.kdeplot(data=data, x="total_bill")
# plt.show()

# ( 3. Distribution Plot (displot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.displot(data=data, x="total_bill",kind="kde")
# plt.show()

# ( 4. Empirical Cumulative Distribution Function Plot (ecdfplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.ecdfplot(data=data, x="total_bill")
# plt.show()

# ( 5. Rug Plot (rugplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.rugplot(data=data, x="total_bill")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Types Of Seaborn Plots --> Matrix Plots in Seaborn )

# ( 1. Heatmap (heatmap) )
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = sns.load_dataset("flights")
# flights_pivot = data.pivot(index="month", columns="year", values="passengers")
# sns.heatmap(flights_pivot, annot=True, fmt="d", cmap="YlGnBu")
# plt.show()

# ( 2. Cluster Map (clustermap) )
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = sns.load_dataset("flights")
# flights_pivot = data.pivot(index="month", columns="year", values="passengers")
# sns.clustermap(flights_pivot, cmap="viridis", standard_scale=1, annot=True)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Types Of Seaborn Plots --> Pair Grid (PairGrid) in Seaborn )

# ( 1. Pair Plot (pairplot) )
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = sns.load_dataset("tips")
# sns.pairplot(data, hue="smoker", palette="coolwarm")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Pairplot function in seaborn )

# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df)
# plt.show()

# ( 1. Pairplot Seaborn: Plotting Selected Variables )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# selected_vars = ["total_bill", "tip"]
# sns.pairplot(df, vars=selected_vars)
# plt.show()

# ( 2. Pairplot Seaborn: Adding a Hue Color to a Seaborn Pairplot )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df, hue="size")
# plt.show()

# ( 3. Pairplot Seaborn: Modifying Color Palette )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df, hue="size", palette="husl")
# plt.show()

# ( 4. Pairplot Seaborn: Diagonal Kind of plots )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df, diag_kind="hist")
# plt.show()

# ( 5. Pairplot Seaborn:Adjusting Plot Kind )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df, kind="kde")
# plt.show()

# ( 6. Pairplot Seaborn:Controlling the Markers )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df, hue="sex", markers=["o", "s"])
# plt.show()

# ( 7. Pairplot Seaborn:Limiting the Variables )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# sns.pairplot(df, hue="sex", vars=["total_bill", "tip", "size"])
# plt.show()

# ( Advanced Customization With Seaborn Pairplot )
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = sns.load_dataset("tips")
# g = sns.pairplot(df, hue="day")
# g.figure.suptitle("Pairplot of Tips Dataset", y=1.02)
# g.set(xticks=[], yticks=[])
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> FacetGrid in Seaborn --> seaborn.FacetGrid() )

# ( Example 1 )
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = sns.load_dataset("tips")
# graph = sns.FacetGrid(df, col="sex", hue="day")

# # map the above form facetgrid with some attributes
# graph.map(plt.scatter, "total_bill", "tip", edgecolor="black").add_legend()
# plt.show()

# ( Example 2 )
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = sns.load_dataset("tips")
# graph = sns.FacetGrid(df, row="smoker", col="time")
# graph.map(plt.hist, "total_bill", bins=15, color="orange")
# plt.show()

# ( Example 3 )
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = sns.load_dataset("tips")
# graph = sns.FacetGrid(df, col="time", hue="smoker")
# graph.map(sns.regplot, "total_bill", "tip").add_legend()
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Time Series Visualization with Seaborn --> Single Line Plot )

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("data.csv")
# data = data.iloc[2:10, :]
# sns.lineplot(x="Pulse", y="Maxpulse", data=data)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Time Series Visualization with Seaborn --> Setting different styles )

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("WineQT.csv")
# data = data.iloc[2:10:]
# sns.lineplot(x="volatile acidity", y="citric acid", data=data, hue="chlorides")
# sns.set_style(style="darkgrid")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Time Series Visualization with Seaborn --> Multiple Line Plot )

# ( Example 1 ) To differentiate on the basis of color
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("WineQT.csv")
# data = data.iloc[2:10:]
# sns.lineplot(x="fixed acidity", y="alcohol", data=data, hue="free sulfur dioxide")
# plt.show()

# ( Example 2 ) To differentiate on the basis of line style
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("WineQT.csv")
# data = data.iloc[2:10:]
# sns.lineplot(x="fixed acidity", y="alcohol", data=data, style="free sulfur dioxide")
# plt.show()

# ( Example 3 ) To differentiate on the basis of size
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("WineQT.csv")
# data = data.iloc[2:10:]
# sns.lineplot(x="fixed acidity", y="alcohol", data=data, style="free sulfur dioxide", hue="free sulfur dioxide", size="free sulfur dioxide")
# sns.lineplot(x="fixed acidity", y="alcohol", data=data, size="free sulfur dioxide")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Time Series Visualization with Seaborn --> Error Bars in Line Plot )

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("WineQT.csv")
# data = data.iloc[2:10:]
# sns.lineplot(x="fixed acidity", y="alcohol", data=data, err_style="band")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Seaborn for Statistical Data Visualization --> Time Series Visualization with Seaborn --> Color Palette along the Line Plot )

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("WineQT.csv")
# data = data.iloc[2:10:]
# sns.lineplot(x="fixed acidity", y="alcohol", data=data, hue="residual sugar", palette="pastel")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Complete EDA Workflow Using NumPy, Pandas, and Seaborn )

# # ( Performing EDA with Numpy and Pandas - Set 1 )

# # Step 1: Setting Up Environment
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import trim_mean

# # Step 2: Loading and Inspecting the Dataset
# data = pd.read_csv("data.csv")
# print(f"Type of Dataset: {type(data)}")
# print(f"The first 10 rows of the dataset:\n{data.head(10)}")
# print(f"The last 10 rows of the dataset:\n{data.tail(10)}")

# # Step 3: Adding and Modifying Columns
# data["CaloriesThousands"] = data["Calories"] / 1000
# print(data.head())

# # Step 4: Describing the Data
# print(data.describe())
# print(data.info())

# # Step 5: Calculating Central Tendencies
# # ( Mean )
# pulse_mean = data.Pulse.mean()
# print(f"Pulse Mean: {pulse_mean}")
# calories_mean = data.Calories.mean()
# print(f"Calories Mean: {calories_mean}")

# # ( Trimmed Mean )
# pulse_tm = trim_mean(data.Pulse, 0.3)
# print(f"Pulse trimmed mean: {pulse_tm}")
# calories_tm = trim_mean(data.Calories, 0.3)
# print(f"Calories trimmed mean: {calories_tm}")

# # ( Weighted Mean )
# pulse_wm = np.average(data.Pulse, weights=data.Maxpulse)
# print(f"Weighted Pulse Mean: {pulse_wm}")

# # ( Median )
# maxpulse_median = data.Maxpulse.median()
# print(f"Maxpulse Median: {maxpulse_median}")
# pulse_median = data.Pulse.median()
# print(f"Pulse Median: {pulse_median}")

# # ( Visualizing with seaborn - Set 2 )
# # ( Visualizing Population per Million )
# fig, ax1 = plt.subplots()
# fig.set_size_inches(15, 9)

# ax1 = sns.barplot(x="Pulse", y="CaloriesThousands", data=data.sort_values("Calories"), palette="Set2")
# ax1.set(xlabel="Pulse", ylabel="CaloriesThousands")
# ax1.set_title("Calories in thousands by state", size=20)
# plt.xticks(rotation=-90)
# plt.show()

# # ( Visualizing Murder Rate per Lakh )
# fig, ax2 = plt.subplots()
# fig.set_size_inches(15, 9)

# ax2 = sns.barplot(x="Pulse", y="Calories", data=data.sort_values("CaloriesThousands", ascending=1), palette="husl")
# ax2.set(xlabel="Pulse", ylabel="Calories")
# ax2.set_title("Calories in thousands by state", size=20)
# plt.xticks(rotation=-90)
# plt.show()

# # ( Code #1 : Standard Deviation )
# pulse_std = data.Pulse.std()
# print(f"Standard deviation of Pulse: {pulse_std}")
# maxpulse_std = data.Maxpulse.std()
# print(f"Standard deviation of Maxpulse: {maxpulse_std}")

# # ( Code #2 : Variance )
# pulse_var = data.Pulse.var()
# print(f"Variance of Pulse: {pulse_var}")
# maxpulse_var = data.Maxpulse.var()
# print(f"Variance of Maxpulse: {maxpulse_var}")

# # ( Code #1 : Standard Deviation )
# pulse_quartile_range = data.Pulse.describe()["75%"] - data.Pulse.describe()["25%"]
# print(f"Pulse IQR: {pulse_quartile_range}")
# maxpulse_quartile_range = data.Maxpulse.describe()["75%"] - data.Maxpulse.describe()["25%"]
# print(f"Maxpulse IQR: {maxpulse_quartile_range}")

# # ( Code #4 : Median Absolute Deviation (MAD) )
# pulse_mad = np.median(np.abs(data["Pulse"] - np.median(data["Pulse"])))
# print(f"Pulse Mad: {pulse_mad}")
# maxpulse_mad = np.median(np.abs(data["Maxpulse"] - np.median(data["Maxpulse"])))
# print(f"Maxpulse Mad: {maxpulse_mad}")

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Complete EDA Workflow Using NumPy, Pandas, and Seaborn --> Titanic Data EDA using Seaborn )

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = pd.read_csv("Titanic-Dataset.csv")
# print(data.head())
# print(data.isnull().sum())

# sns.catplot(x="Sex", hue="Survived", kind="count", data=data)
# plt.show()

# # Group the dataset by Pclass and Survived and then unstack them
# group = data.groupby(["Pclass", "Survived"])
# pclass_survived = group.size().unstack()

# # Heatmap - Color encoded 2D representation of data.
# sns.heatmap(pclass_survived, annot=True, fmt="d")
# plt.show()

# # Violinplot Displays distribution of data across all levels of a category
# sns.violinplot(x="Sex", y="Age", hue="Survived", data=data, split=True)
# plt.show()

# # Adding a column Family_Size
# data["Family_Size"] = 0
# data["Family_Size"] = data["Parch"] + data["SibSp"]

# # Adding a column Alone
# data["Alone"] = 0
# data.loc[data["Family_Size"] == 0, "Alone"] = 1

# # Factorplot for Family_Size
# sns.catplot(x="Family_Size", y="Survived", data=data, kind="point")
# sns.catplot(x="Alone", y="Survived", data=data, kind="point")
# plt.show()

# # Divide Fare into 4 bins
# data["Fare_Range"] = pd.qcut(data["Fare"], 4)

# # Barplot - Shows approximate values based on the height of bars
# sns.barplot(x="Fare_Range", y="Survived", data=data)
# plt.show()

# # Countplot
# sns.catplot(x="Embarked", hue="Survived", kind="count", col="Pclass", data=data)
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Complete EDA Workflow Using NumPy, Pandas, and Seaborn --> Uber Rides Data Analysis )

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder

# data = pd.read_csv("UberDataset.csv")
# print(data.head())
# print(data.isnull().sum())
# print(data.shape)
# print(data.info())
# print(data.describe())

# # ( Data Preprocessing )
# data["PURPOSE"] = data["PURPOSE"].fillna(0)
# data["START_DATE"] = pd.to_datetime(data["START_DATE"], errors="coerce")
# data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce")
# data["date"] = pd.DatetimeIndex(data["START_DATE"]).date
# data["time"] = pd.DatetimeIndex(data["START_DATE"]).hour

# # changing into categories of day and night
# data["day-night"] = pd.cut(
#     x=data["time"],
#     bins=[0, 10, 15, 19, 24],
#     labels=["Morning", "Afternoon", "Evening", "Night"],
# )

# data.dropna(inplace=True)
# data.drop_duplicates(inplace=True)

# # ( Data Visualization )
# obj = data.dtypes == "object"
# object_col = list(obj[obj].index)

# unique_vlaues = {}
# for col in object_col:
#     unique_vlaues[col] = data[col].unique().size
# print(unique_vlaues)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# sns.countplot(data=data, x="CATEGORY", palette=["blue", "orange"], legend=False)
# plt.xlabel("CATEGORY")
# plt.ylabel("COUNT")
# plt.title("CATEGORY")
# plt.xticks(rotation=90)

# plt.subplot(1, 2, 2)
# sns.countplot(data=data, x="PURPOSE", palette="Set2", legend=False)
# plt.xlabel("PURPOSE")
# plt.ylabel("COUNT")
# plt.title("PURPOSE")
# plt.xticks(rotation=90)

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

# sns.countplot(data=data, x="day-night", palette="husl", legend=False)
# plt.show()

# plt.figure(figsize=(15, 5))
# sns.countplot(data=data, x="PURPOSE", hue="CATEGORY")
# plt.xticks(rotation=90)
# plt.show()

# object_col = ["PURPOSE", "CATEGORY"]
# data[object_col] = data[object_col].astype(str)
# OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# OH_col = pd.DataFrame(OH_encoder.fit_transform(data[object_col]))
# OH_col.index = data.index
# OH_col.columns = OH_encoder.get_feature_names_out()
# df_final = data.drop(object_col, axis=1)
# data = pd.concat([df_final, OH_col], axis=1)

# numeric_data = data.select_dtypes(include=["number"])
# sns.heatmap(numeric_data.corr(), cmap="BrBG", fmt=".2f", linewidths=2, annot=True)
# plt.show()

# data["MONTH"] = pd.DatetimeIndex(data["START_DATE"]).month
# month_label = {
#     1.0: "Jan",
#     2.0: "Feb",
#     3.0: "Mar",
#     4.0: "April",
#     5.0: "May",
#     6.0: "June",
#     7.0: "July",
#     8.0: "Aug",
#     9.0: "Sep",
#     10.0: "Oct",
#     11.0: "Nov",
#     12.0: "Dec",
# }
# data["MONTH"] = data.MONTH.map(month_label)
# mon = data.MONTH.value_counts(sort=False)
# df = pd.DataFrame(
#     {
#         "MONTHS": mon.values,
#         "VALUE COUNT": data.groupby("MONTH", sort=False)["MILES"].max(),
#     }
# )
# p = sns.lineplot(data=df)
# p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
# plt.show()

# data["DAY"] = data.START_DATE.dt.weekday
# day_label = {0: "Mon", 1: "Tues", 2: "Wed", 3: "Thus", 4: "Fri", 5: "Sat", 6: "Sun"}
# data["DAY"] = data["DAY"].map(day_label)
# day_label = data.DAY.value_counts()
# sns.barplot(x=day_label.index, y=day_label)
# plt.xlabel("DAY")
# plt.ylabel("COUNT")
# plt.show()

# sns.boxplot(data["MILES"])
# plt.show()

# sns.boxplot(data[data["MILES"]<100]["MILES"])
# plt.show()

# sns.displot(data[data["MILES"]<40]["MILES"])
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Complete EDA Workflow Using NumPy, Pandas, and Seaborn --> Zomato Data Analysis Using Python )

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# data = pd.read_csv("Zomato-data-.csv")
# print(data.head())
# print(data.shape)
# print(data.isnull().sum())
# print(data.info())
# print(data.describe())

# def handleRate(value):
#     value = str(value).split("/")
#     value = value[0]
#     return float(value)


# data["rate"] = data["rate"].apply(handleRate)
# print(data.head())

# sns.countplot(data=data, x="listed_in(type)", palette="Set1")
# plt.xlabel("Type of restaurant")
# plt.show()

# group_data = data.groupby("listed_in(type)")["votes"].sum()
# result = pd.DataFrame({"votes": group_data})
# plt.plot(result, c="green", marker="o")
# plt.xlabel("Type of resturant", c="r", size=20)
# plt.ylabel("Votes", c="r", size=20)
# plt.show()

# max_votes = data["votes"].max()
# resturant_with_max_votes = data.loc[data["votes"] == max_votes, "name"]
# print(f"Resturants with maximum votes: {resturant_with_max_votes}")

# sns.countplot(data=data, x="online_order", palette=["b", "r"])
# plt.show()

# plt.hist(data["rate"], bins=5)
# plt.title("Ratings Distribution")
# plt.show()

# couple_data = data["approx_cost(for two people)"]
# sns.countplot(x=couple_data)
# plt.show()

# plt.figure(figsize=(6, 6))
# sns.boxplot(x="online_order", y="rate", data=data, palette=["#3274a1", "#e1812c"])
# plt.show()

# pivot_table = data.pivot_table(index="listed_in(type)", columns="online_order", aggfunc="size", fill_value=0)
# sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt="d")
# plt.xlabel("Online order")
# plt.ylabel("Listed_in(type)")
# plt.title("Heatmap")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Complete EDA Workflow Using NumPy, Pandas, and Seaborn --> Global Covid-19 Data Analysis and Visualizations )

# ( Step 1: Importing Necessary Libraries )
# # Data analysis and Manipulation
# import pandas as pd
# import plotly.io as pio
# import plotly.express as px

# # Data Visualization
# from plotly.figure_factory import create_table
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # Importing Plotly
# import plotly.offline as py

# py.init_notebook_mode(connected=True)

# # Initializing Plotly
# pio.renderers.default = "colab"

# # ( Step 2: Importing the Datasets )
# data1 = pd.read_csv("covid.csv")
# data2 = pd.read_csv("covid_grouped.csv")
# data3 = pd.read_csv("coviddeath.csv")

# print(data1.head())
# print("--------------------------------")
# print(data2.head())
# print("--------------------------------")
# print(data3.head())

# print(list(data1.columns))
# print("--------------------------------")
# print(list(data2.columns))
# print("--------------------------------")
# print(list(data3.columns))

# print(data1.shape)
# print("--------------------------------")
# print(data2.shape)
# print("--------------------------------")
# print(data3.shape)

# print(data1.size)
# print("--------------------------------")
# print(data2.size)
# print("--------------------------------")
# print(data3.size)

# print(data1.isnull().sum())
# print("--------------------------------")
# print(data2.isnull().sum())
# print("--------------------------------")
# print(data3.isnull().sum())

# print(data1.info())
# print("--------------------------------")
# print(data2.info())
# print("--------------------------------")
# print(data3.info())

# print(data1.describe())
# print("--------------------------------")
# print(data2.describe())
# print("--------------------------------")
# print(data3.describe())

# # ( Step 3: Dataset cleaning )
# data1.drop(["NewCases", "NewDeaths", "NewRecovered"], axis=1, inplace=True)

# # Select random set of values from dataset1
# print(data1.sample(5))

# # Creating table using plotly express
# colorscale = [[0, '#4d004c'], [.5, '#f2e5ff'], [1, '#ffffff']]
# table = create_table(data1.head(15), colorscale=colorscale)
# py.plot(table, auto_open=True)

# # ( Step 4: Bar graphs- Comparisons between COVID infected countries in terms of total cases, total deaths, total recovered & total tests )
# fig1 = px.bar(
#     data1,
#     x="Country/Region",
#     y="TotalCases",
#     color="TotalCases",
#     height=500,
#     hover_data=["Country/Region", "Continent"],
# )
# py.plot(fig1, auto_open=True)

# fig2 = px.bar(
#     data1.head(15),
#     x="TotalTests",
#     y="Country/Region",
#     color="TotalTests",
#     orientation="h",
#     height=500,
#     hover_data=["Country/Region", "Continent"]
# )
# py.plot(fig2, auto_open=True)

# fig3 = px.bar(
#     data1.head(15),
#     x="TotalTests",
#     y="Continent",
#     color="TotalTests",
#     orientation="h",
#     height=500,
#     hover_data=["Country/Region", "Continent"]
# )
# py.plot(fig3, auto_open=True)

# # ( Step 5: Data Visualization through Bubble Charts-Continent Wise )
# fig4 = px.scatter(
#     data1,
#     x="Continent",
#     y="TotalCases",
#     color="TotalCases",
#     size="TotalCases",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"]
# )
# py.plot(fig4, auto_open=True)

# fig5 = px.scatter(
#     data1.head(60),
#     x="Continent",
#     y="TotalCases",
#     color="TotalCases",
#     size="TotalCases",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
#     log_y=True,
# )
# py.plot(fig5, auto_open=True)

# fig6 = px.scatter(
#     data1.head(55),
#     x="Continent",
#     y="TotalCases",
#     color="TotalCases",
#     size="TotalCases",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"]
# )
# py.plot(fig6, auto_open=True)

# fig7 = px.scatter(
#     data1.head(50),
#     x="Continent",
#     y="TotalCases",
#     color="TotalCases",
#     size="TotalCases",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
#     log_y=True
# )
# py.plot(fig7, auto_open=True)

# # ( Step 6: Data Visualization through Bubble Charts-Country Wise )
# fig8 = px.scatter(
#     data1.head(100),
#     x="Country/Region",
#     y="TotalCases",
#     color="TotalCases",
#     size="TotalCases",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
# )
# py.plot(fig8, auto_open=True)

# fig9 = px.scatter(
#     data1.head(30),
#     x="Country/Region",
#     y="TotalCases",
#     color="Country/Region",
#     size="TotalCases",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
#     log_y=True
# )
# py.plot(fig9, auto_open=True)

# fig10 = px.scatter(
#     data1.head(10),
#     x="Country/Region",
#     y="TotalDeaths",
#     color="Country/Region",
#     size="TotalDeaths",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
# )
# py.plot(fig10, auto_open=True)

# fig11 = px.scatter(
#     data1.head(30),
#     x="Country/Region",
#     y="Tests/1M pop",
#     color="Country/Region",
#     size="Tests/1M pop",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
# )
# py.plot(fig11, auto_open=True)

# fig12 = px.scatter(
#     data1.head(30),
#     x="Country/Region",
#     y="Tests/1M pop",
#     color="Tests/1M pop",
#     size="Tests/1M pop",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
# )
# py.plot(fig12, auto_open=True)

# fig13 = px.scatter(
#     data1.head(30),
#     x="TotalCases",
#     y="TotalDeaths",
#     color="TotalDeaths",
#     size="TotalDeaths",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
# )
# py.plot(fig13, auto_open=True)

# fig14 = px.scatter(
#     data1.head(30),
#     x="TotalCases",
#     y="TotalDeaths",
#     color="TotalDeaths",
#     size="TotalDeaths",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
#     log_x=True,
#     log_y=True
# )
# py.plot(fig14, auto_open=True)

# fig15 = px.scatter(
#     data1.head(30),
#     x="TotalCases",
#     y="TotalDeaths",
#     color="TotalTests",
#     size="TotalTests",
#     size_max=80,
#     hover_data=["Country/Region", "Continent"],
#     log_x=True,
#     log_y=True
# )
# py.plot(fig15, auto_open=True)

# # ( Step 7: Advanced Data Visualization- Bar graphs for All top infected Countries )
# fig16 = px.bar(
#     data2,
#     x="Date",
#     y="Confirmed",
#     color="Confirmed",
#     hover_data=["Confirmed", "Date", "Country/Region"],
#     height=400,
# )
# py.plot(fig16, auto_open=True)

# fig17 = px.bar(
#     data2,
#     x="Date",
#     y="Deaths",
#     color="Deaths",
#     hover_data=["Confirmed", "Date", "Country/Region"],
#     height=400
# )
# py.plot(fig17, auto_open=True)

# # ( Step 8: Countries Specific COVID Data Visualization: (United States) )
# df_US = data2.loc[data2["Country/Region"] == "US"]
# fig18 = px.bar(df_US, x="Date", y="Confirmed", color="Confirmed", height=400)
# py.plot(fig18, auto_open=True)

# fig19 = px.bar(df_US, x="Date", y="Recovered", color="Recovered", height=400)
# py.plot(fig19, auto_open=True)

# fig20 = px.line(df_US, x="Date", y="Recovered", height=400)
# py.plot(fig20, auto_open=True)

# fig21 = px.line(df_US, x="Date", y="Deaths", height=400)
# py.plot(fig21, auto_open=True)

# fig22 = px.line(df_US, x="Date", y="Confirmed", height=400)
# py.plot(fig22, auto_open=True)

# fig22 = px.line(df_US, x="Date", y="New cases", height=400)
# py.plot(fig22, auto_open=True)

# fig23 = px.bar(df_US, x="Date", y="New cases", height=400)
# py.plot(fig23, auto_open=True)

# fig24 = px.scatter(df_US, x="Confirmed", y="Deaths", height=400)
# py.plot(fig24, auto_open=True)

# # ( Step 9: Visualization of Data in terms of Maps )
# map1 = px.choropleth(
#     data2,
#     locations="iso_alpha",
#     color="Confirmed",
#     hover_name="Country/Region",
#     color_continuous_scale="Blues",
#     animation_frame="Date",
# )
# py.plot(map1, auto_open=True, auto_play=True)

# map2 = px.choropleth(
#     data2,
#     locations="iso_alpha",
#     color="Deaths",
#     hover_name="Country/Region",
#     color_continuous_scale="Viridis",
#     animation_frame="Date",
# )
# py.plot(map2, auto_open=True, auto_play=True)

# map3 = px.choropleth(
#     data2,
#     locations="iso_alpha",
#     color="Recovered",
#     hover_name="Country/Region",
#     color_continuous_scale="RdYlGn",
#     projection="natural earth",
#     animation_frame="Date",
# )
# py.plot(map3, auto_open=True, auto_play=True)

# map4 = px.bar(
#     data2,
#     x="WHO Region",
#     y="Confirmed",
#     color="WHO Region",
#     hover_name="Country/Region",
#     animation_frame="Date",
# )
# py.plot(map4, auto_open=True, auto_play=True)

# # ( Step 10: Visualize text using Word Cloud )
# sentences = data3["Condition"].tolist()
# sentences_str = " ".join(sentences)

# # Convert the string into WordCloud
# plt.figure(figsize=(20, 20))
# plt.imshow(WordCloud().generate(sentences_str))
# plt.show()

# colum2_list = data3["Condition Group"].tolist()
# column_str = " ".join(colum2_list)

# plt.figure(figsize=(20, 20))
# plt.imshow(WordCloud().generate(column_str))
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Complete EDA Workflow Using NumPy, Pandas, and Seaborn --> iPhone Sales Analysis )

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# data = pd.read_csv("Order_details-masked.csv")
# print(list(data.columns))
# print(data.shape)
# print(data.isnull().sum())
# print(data.info())
# print(data.describe())
# print(data.head(10))

# data["Time"] = pd.to_datetime(data["Transaction Date"])
# data["Hour"] = data["Time"].dt.hour

# timemost1 = data["Hour"].value_counts().index.tolist()[:24]
# timemost2 = data["Hour"].value_counts().values.tolist()[:24]

# tmost = np.column_stack((timemost1, timemost2))
# print("Hour Of Day \tCumulative Number of Purchases")
# print("\n".join("\t\t".join(map(str, row)) for row in tmost))

# timemost = data["Hour"].value_counts()
# timemost1 = []

# for i in range(0, 23):
#     timemost1.append(i)

# timemost2 = timemost.sort_index()
# timemost2.tolist()
# timemost2 = pd.DataFrame(timemost2)

# plt.figure(figsize=(20, 10))
# plt.title(
#     "Sales Happening Per Hour (Spread Throughout The Week)",
#     fontdict={"fontname": "Monospace", "fontsize": 30},
#     y=1.05,
# )
# plt.xlabel("Hour", fontsize=18, labelpad=20)
# plt.ylabel("Number Of Purchases Made", fontsize=18, labelpad=20)
# plt.plot(timemost1, timemost2, color="m")
# plt.grid()
# plt.show()

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Web Scraping For EDA --> How to Extract Weather Data from Google in Python? )

# import requests

# res = requests.get("https://ipinfo.io/")
# data = res.json()

# citydata = data["city"]
# print(f"Current Location: {citydata}")

# url = "https://wttr.in/{}".format(citydata)
# res = requests.get(url)

# print(res.text)

# ------------------------------------------------------------------------------

# ( Python For Data Analysis --> Web Scraping For EDA --> News Scraping and Analysis )

# import newspaper
# import feedparser

# def scrape_news_form_feed(feed_url):
#     articles = []
#     feed = feedparser.parse(feed_url)
#     for entry in feed.entries:
#         article = newspaper.Article(entry.link)
#         article.download()
#         article.parse()
#         articles.append(
#             {
#                 "title": article.title,
#                 "author": article.authors,
#                 "publish_date": article.publish_date,
#                 "content": article.text,
#             }
#         )
#     return articles

# feed_url = "http://feeds.bbci.co.uk/news/rss.xml"
# articles = scrape_news_form_feed(feed_url)

# for article in articles:
#     print(f"Title: {article['title']}")
#     print(f"Author: {article['author']}")
#     print(f"Publish_date: {article['publish_date']}")
#     print(f"Content: {article['content']}")

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Data Visualization with Matplotlib )

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([1, 2, 3, 4])
# y = x ** 2
# plt.plot(x, y)
# plt .show()

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Effective Data Visualization With Seaborn  )

# import seaborn as sns
# import matplotlib.pyplot as plt

# data = sns.load_dataset("tips")

# plt.figure(figsize=(6, 4))
# sns.scatterplot(x="total_bill", y="tip", data=data, hue="time", style="time")
# plt.title("Total Bill vs Tip")
# plt.xlabel("Total bill")
# plt.ylabel("Tip")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Data Visualization with Pandas )

# import pandas as pd
# import matplotlib.pyplot as plt

# data = {
#     "Category": ["A"] * 10 + ["B"] * 10,
#     "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
# }
# df = pd.DataFrame(data)
# df.boxplot(by="Category")
# plt.title("Box plot example")
# plt.xlabel("Category")
# plt.ylabel("Value")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Data Visualization with Pandas )

# import pandas as pd
# import matplotlib.pyplot as plt

# data = {
#     "Category": ["A"]*10 + ["B"]*10,
#     "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# }

# df = pd.DataFrame(data)
# df.boxplot(by="Category")
# plt.title("Box plot example")
# plt.xlabel("Category")
# plt.ylabel("Value")
# plt.show()

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Data Visualization with Plotly )

# import plotly.express as px

# tips = px.data.tips()
# fig = px.bar(tips, x="day", y="total_bill", title="Average total bill per day")
# fig.show()

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Data Visualization with Plotly )

# from plotnine import ggplot, aes, geom_line, labs, theme_minimal
# from plotnine.data import economics

# line_plot = (
#     ggplot(economics, aes(x="date", y="unemploy"))
#     + geom_line(color="blue")
#     + labs(title="Unemployment Rate Over Time", x="Date", y="Number of unemployed")
#     + theme_minimal()
# )

# line_plot.draw(True)

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Data Visualizations with Altair )

# import altair as alt
# from vega_datasets import data

# iris = data.iris()
# scatter_plot = alt.Chart(iris).mark_point().encode(
#     x="sepalLength",
#     y="petalLength",
#     color="species"
# )

# scatter_plot.save('scatter_plot.html')
# scatter_plot.show()

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Interactive Data Visualization with Bokeh )

# from bokeh.models import HoverTool
# from bokeh.plotting import figure, show
# from bokeh.io import output_file

# output_file("scatter_plot.html")
# p = figure(
#     title="Scatter plot with hover tool", x_axis_label="X-axis", y_axis_label="Y-axis"
# )
# p.scatter(x=[1, 2, 3, 4, 5], y=[6, 7, 2, 4, 5], size=10, color="green", alpha=0.5)
# hover = HoverTool()
# hover.tooltips = [("X", "@x"), ("Y", "@y")]
# p.add_tools(hover)
# show(p)

# ------------------------------------------------------------------------------

# ( Python Data Visulization --> Mastering Advanced Data Visualization with Pygal )

# import pygal
# from pygal.style import Style

# # Create a custom style
# custom_style = Style(
#     background="transparent",
#     plot_background="transparent",
#     foreground="#000000",
#     foreground_strong="#000000",
#     foreground_subtle="#6e6e6e",
#     opacity=".6",
#     opacity_hover=".9",
#     transition="400ms",
#     colors=("#E80080", "#404040"),
# )

# # Create a line chart
# line_chart = pygal.Line(
#     style=custom_style, show_legend=True, x_title="Months", y_title="Values"
# )
# line_chart.title = "Monthly Trends"
# line_chart.add("Series 1", [1, 3, 5, 7, 9])
# line_chart.add("Series 2", [2, 4, 6, 8, 10])

# line_chart.render_to_file("line_chart.svg")

# ------------------------------------------------------------------------------

# ((((((((((((((( Data Analysis Libraries )))))))))))))))

# ( Pandas Tutorial --> Pandas DataFrame --> Pandas Dataframe Index )

# import pandas as pd

# data = {
#     "Name": ["Abdallah", "Noor", "Reham", "John"],
#     "Age": [18, 20, 21, 16],
#     "Gender": ["Male", "Female", "Female", "Male"],
#     "Income": [1000000, 500000, 800000, 400000],
# }

# # ( Accessing and Modifying the Index )
# df = pd.DataFrame(data)
# print(df.index)

# # ( Setting a Custom Index )
# df_with_index = df.set_index("Name")
# print(df_with_index)

# # ( Resetting the Index )
# df_reset = df.reset_index()
# print(df_reset)

# # ( Indexing with loc )
# row = df.loc[1]
# print(row)

# # ( Changing the Index )
# df_with_new_index = df.set_index("Age")
# print(df_with_new_index)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Pandas Access DataFrame )

# import pandas as pd

# data = {
#     "Name": ["Abdallah", "Noor", "Reham", "John"],
#     "Age": [18, 20, 21, 16],
#     "Gender": ["Male", "Female", "Female", "Male"],
#     "Income": [1000000, 500000, 800000, 400000],
# }

# df = pd.DataFrame(data)

# # ( Accessing Columns From DataFrame )
# name_column = df["Name"]
# print(name_column)

# # ( Accessing Rows by Index )
# second_row = df.iloc[1]
# print(second_row)

# # ( Accessing Multiple Rows or Columns )
# subset = df.loc[0:2, ["Name", "Age"]]
# print(subset)

# # ( Accessing Rows Based on Conditions )
# filtered_data = df[df["Age"] > 17]
# print(filtered_data)

# # ( Accessing Specific Cells with at and iat )
# salary_at_index_2 = df.at[2, "Income"]
# print(salary_at_index_2)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Indexing and Selecting Data with Pandas )

# import pandas as pd

# # ( Selecting a Single Column )
# data = pd.read_csv("data.csv")
# first = data["Pulse"]
# print(first.head())

# # ( Selecting Multiple Columns )
# multiple = data[["Pulse", "Maxpulse", "Calories"]]
# print(multiple.head())

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Indexing a DataFrame using .loc[ ] )

# import pandas as pd

# data = pd.read_csv("data.csv", index_col="Calories")

# # ( Selecting multiple rows )
# first = data.loc[[340, 406]]
# print(first)

# # ( Selecting all of the rows and some columns )
# multiple = data.loc[:, ["Pulse", "Duration"]]
# print(multiple.head())

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Slicing Pandas Dataframe )

# import pandas as pd

# player_list = [
#     ["M.S.Dhoni", 36, 75, 5428000],
#     ["A.B.D Villers", 38, 74, 3428000],
#     ["V.Kohli", 31, 70, 8428000],
#     ["S.Smith", 34, 80, 4428000],
#     ["C.Gayle", 40, 100, 4528000],
#     ["J.Root", 33, 72, 7028000],
#     ["K.Peterson", 42, 85, 2528000],
# ]

# df = pd.DataFrame(player_list, columns=["Name", "Age", "Weight", "Salary"])

# # ( Using Boolean Conditions in a Pandas DataFrame )
# data = df[df["Age"] > 35]
# print(data)

# # ( Slicing Rows in Dataframe )
# df.set_index("Name", inplace=True)
# custom = df.loc["S.Smith":"J.Root"]
# print(custom)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Filter Pandas Dataframe with multiple conditions )

# import pandas as pd

# dataFrame = pd.DataFrame(
#     {
#         "Name": [
#             " RACHEL  ",
#             " MONICA  ",
#             " PHOEBE  ",
#             "  ROSS    ",
#             "CHANDLER",
#             " JOEY    ",
#         ],
#         "Age": [30, 35, 37, 33, 34, 30],
#         "Salary": [100000, 93000, 88000, 120000, 94000, 95000],
#         "JOB": ["DESIGNER", "CHEF", "MASUS", "PALENTOLOGY", "IT", "ARTIST"],
#     }
# )

# # ( Filter Pandas Dataframe with multiple conditions Using loc )
# data_loc = dataFrame.loc[
#     (dataFrame["Salary"] >= 100000)
#     & (dataFrame["Age"] < 40)
#     & (dataFrame["JOB"].str.startswith("D")),
#     ["Name", "JOB"],
# ]
# print(data_loc)

# # ( Filter Pandas Dataframe Using NumPy )
# import numpy as np

# data_loc2 = np.where(
#     (dataFrame["Salary"] >= 100000)
#     & (dataFrame["Age"] < 40)
#     & (dataFrame["JOB"].str.startswith("D"))
# )
# print(dataFrame.loc[data_loc2])

# # ( Filter Pandas Dataframe Using Query (eval and query works only with columns )
# data_loc3 = dataFrame.query("Salary <= 100000 & Age < 40 & JOB.str.startswith('C')")
# print(data_loc3)

# # ( Pandas Boolean indexing multiple conditions standard way (“Boolean indexing” works with values in a column only )
# data_loc4 = dataFrame[
#     (dataFrame["Salary"] >= 100000)
#     & (dataFrame["Age"] < 40)
#     & (dataFrame["JOB"].str.startswith("P"))
# ][["Name", "Age", "Salary"]]
# print(data_loc4)

# # ( Eval multiple conditions  (“eval” and “query” works only with columns )
# data_loc5 = dataFrame[dataFrame.eval("Salary <= 100000 & (Age < 40) & JOB.str.startswith('A')")]
# print(data_loc5)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Concatenating DataFrame using .concat() )

# import pandas as pd

# data1 = {
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Msc", "MA", "MCA", "Phd"],
# }

# data2 = {
#     "Name": ["Abhi", "Ayushi", "Dhiraj", "Hitesh"],
#     "Age": [17, 14, 12, 52],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Btech", "B.A", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1, index=[0, 1, 2, 3])
# df2 = pd.DataFrame(data2, index=[4, 5, 6, 7])

# frames = [df1, df2]
# res = pd.concat(frames)
# print(res)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Concatenating DataFrame by setting logic on axes )

# import pandas as pd

# data1 = {
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Msc", "MA", "MCA", "Phd"],
#     "Mobile No": [97, 91, 58, 76],
# }
# data2 = {
#     "Name": ["Gaurav", "Anuj", "Dhiraj", "Hitesh"],
#     "Age": [22, 32, 12, 52],
#     "Address": ["Allahabad", "Kannuaj", "Allahabad", "Kannuaj"],
#     "Qualification": ["MCA", "Phd", "Bcom", "B.hons"],
#     "Salary": [1000, 2000, 3000, 4000],
# }

# df1 = pd.DataFrame(data1, index=[0, 1, 2, 3])
# df2 = pd.DataFrame(data2, index=[2, 3, 6, 7])

# # ( set axes join = inner for intersection of dataframe )
# intersect = pd.concat([df1, df2], axis=1, join="inner")
# print(intersect)

# # ( set axes join = outer for union of dataframe )
# union = pd.concat([df1, df2], axis=1)
# print(union)

# # ( using a specific index, as passed to the join_axes argument )
# join = pd.concat([df1, df2], axis=1, join="inner")
# print(join)

# # ( Concatenating DataFrame using .append() )
# app = df1._append(df2)
# print(app)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Concatenating DataFrame by ignoring indexes )

# import pandas as pd

# data1 = {
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Msc", "MA", "MCA", "Phd"],
#     "Mobile No": [97, 91, 58, 76],
# }
# data2 = {
#     "Name": ["Gaurav", "Anuj", "Dhiraj", "Hitesh"],
#     "Age": [22, 32, 12, 52],
#     "Address": ["Allahabad", "Kannuaj", "Allahabad", "Kannuaj"],
#     "Qualification": ["MCA", "Phd", "Bcom", "B.hons"],
#     "Salary": [1000, 2000, 3000, 4000],
# }

# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# res = pd.concat([df1, df2], ignore_index=True)
# print (res)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Concatenating DataFrame with group keys )

# import pandas as pd

# data1 = {
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Msc", "MA", "MCA", "Phd"],
# }
# data2 = {
#     "Name": ["Abhi", "Ayushi", "Dhiraj", "Hitesh"],
#     "Age": [17, 14, 12, 52],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Btech", "B.A", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# res = pd.concat([df1, df2], keys=["x", "y"])
# print(res)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Concatenating with mixed ndims )

# import pandas as pd

# data1 = {
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Msc", "MA", "MCA", "Phd"],
# }

# df1 = pd.DataFrame(data1, index=[0, 1, 2, 3])
# s1 = pd.Series([1000, 2000, 3000, 4000], name="Salary")

# res = pd.concat([df1, s1], axis=1)
# print(res)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Merging a dataframe with one unique key combination )

# import pandas as pd

# data1 = {
#     "key": ["K0", "K1", "K2", "K3"],
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
# }
# data2 = {
#     "key": ["K0", "K1", "K2", "K3"],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Btech", "B.A", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# merge = pd.merge(df1, df2, on="key")
# print(merge)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Merging dataframe using multiple join keys )

# import pandas as pd

# data1 = {
#     "key": ["K0", "K1", "K2", "K3"],
#     "key1": ["K0", "K1", "K0", "K1"],
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
# }
# data2 = {
#     "key": ["K0", "K1", "K2", "K3"],
#     "key1": ["K0", "K0", "K0", "K0"],
#     "Address": ["Nagpur", "Kanpur", "Allahabad", "Kannuaj"],
#     "Qualification": ["Btech", "B.A", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# merge = pd.merge(df1, df2, on=["key", "key1"])
# print(merge)

# # using keys from left frame
# merge2 = pd.merge(df1, df2, how="left", on=["key", "key1"])
# print (merge2)

# # using keys from right frame
# merge3 = pd.merge(df1, df2, how="right", on=["key", "key1"])
# print (merge3)

# # Use union of keys from both frames
# merge4 = pd.merge(df1, df2, how="outer", on=["key", "key1"])
# print (merge4)

# # Use intersection of keys from both frames
# merge5 = pd.merge(df1, df2, how="inner", on=["key", "key1"])
# print (merge5)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Joining DataFrame )

# import pandas as pd

# data1 = {"Name": ["Jai", "Princi", "Gaurav", "Anuj"], "Age": [27, 24, 22, 32]}
# data2 = {
#     "Address": ["Allahabad", "Kannuaj", "Allahabad", "Kannuaj"],
#     "Qualification": ["MCA", "Phd", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# # ( useing .join() method in order to join dataframes )
# join = df1.join(df2)
# print (join)

# # ( useing how = 'outer' in order to get union )
# join_outer = df1.join(df2, how="outer")
# print(join_outer)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Joining dataframe using on in an argument )

# import pandas as pd

# data1 = {
#     "Name": ["Jai", "Princi", "Gaurav", "Anuj"],
#     "Age": [27, 24, 22, 32],
#     "Key": ["K0", "K1", "K2", "K3"],
# }
# data2 = {
#     "Address": ["Allahabad", "Kannuaj", "Allahabad", "Kannuaj"],
#     "Qualification": ["MCA", "Phd", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2, index=['K0', 'K2', 'K3', 'K4'])

# join_key = df1.join(df2, on="Key")
# print(join_key)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Merging, Joining, and Concatenating Dataframes --> Joining singly-indexed DataFrame with multi-indexed DataFrame )

# import pandas as pd

# data1 = {"Name": ["Jai", "Princi", "Gaurav"], "Age": [27, 24, 22]}
# data2 = {
#     "Address": ["Allahabad", "Kannuaj", "Allahabad", "Kanpur"],
#     "Qualification": ["MCA", "Phd", "Bcom", "B.hons"],
# }

# df1 = pd.DataFrame(data1, index=pd.Index(["K0", "K1", "K2"], name="key"))
# index = pd.MultiIndex.from_tuples(
#     [("K0", "Y0"), ("K1", "Y1"), ("K2", "Y2"), ("K2", "Y3")], names=["key", "Y"]
# )

# df2 = pd.DataFrame(data2, index=index)
# result = df1.join(df2, how="inner")
# print(result)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Sorting Pandas DataFrame --> Sort DataFrame by One Column Value )

# import pandas as pd

# data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#         'Age': [25, 30, 35, 40],
#         'Score': [85, 90, 95, 80]}

# df = pd.DataFrame(data)

# # Sorting by 'Age' in ascending order
# sorted_df = df.sort_values(by="Age")
# print(sorted_df)

# # Sorting by 'Age' in descending order
# sorted_df2 = df.sort_values(by="Age", ascending=False)
# print(sorted_df2)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Sorting Pandas DataFrame --> Sort DataFrame by Multiple Columns )

# import pandas as pd

# data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#         'Age': [25, 30, 35, 40],
#         'Score': [85, 90, 95, 80]}

# df = pd.DataFrame(data)
# sorted_df = df.sort_values(by=["Age", "Score"])
# print(sorted_df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Sorting Pandas DataFrame --> Sort DataFrame with Missing Values )

# import pandas as pd

# data_null = {"Name": ["Alice", "Bob", "Charlie", "David"],"Age": [28, 22, None, 22]}
# df_null = pd.DataFrame(data_null)

# sorted_df = df_null.sort_values(by="Age", na_position="first")
# print(sorted_df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Sorting Pandas DataFrame --> Choosing the Sorting Algorithm )

# import pandas as pd

# data = {
#     "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
#     "Age": [28, 22, 25, 22, 28],
#     "Score": [85, 90, 95, 80, 88],
# }
# df = pd.DataFrame(data)

# sorted_df1 = df.sort_values(by="Age", kind="mergesort")
# sorted_df2 = df.sort_values(by="Age", kind="heapsort")
# sorted_df3 = df.sort_values(by="Age", kind="quicksort")
# sorted_df4 = df.sort_values(by="Age", kind="stable")

# print(sorted_df1)
# print(sorted_df2)
# print(sorted_df3)
# print(sorted_df4)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Sorting Pandas DataFrame --> Custom Sorting with Key Functions )

# import pandas as pd

# data = {
#     "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
#     "Age": [28, 22, 25, 22, 28],
#     "Score": [85, 90, 95, 80, 88],
# }
# df = pd.DataFrame(data)
# sorted_df = df.sort_values(by="Name", key=lambda col: col.str.lower())
# print(sorted_df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas DataFrame --> Pivot Table in PandasCustom Sorting with Key Functions --> Create a Pivot Table in Pandas )

# import pandas as pd

# data = {
#     "Product": [
#         "Carrots",
#         "Broccoli",
#         "Banana",
#         "Banana",
#         "Beans",
#         "Orange",
#         "Broccoli",
#         "Banana",
#     ],
#     "Category": [
#         "Vegetable",
#         "Vegetable",
#         "Fruit",
#         "Fruit",
#         "Vegetable",
#         "Fruit",
#         "Vegetable",
#         "Fruit",
#     ],
#     "Quantity": [8, 5, 3, 4, 5, 9, 11, 8],
#     "Amount": [270, 239, 617, 384, 626, 610, 62, 90],
# }
# df = pd.DataFrame(data)

# # ( Get the Total Sales of Each Product )
# pivot1 = df.pivot_table(index=["Product"], values=["Amount"], aggfunc="sum")
# print(pivot1)

# # ( Get the Total Sales of Each Category )
# pivot2 = df.pivot_table(index=["Category"], values=["Amount"], aggfunc="sum")
# print(pivot2)

# # ( Get Total Sales by Category and Product Both )
# pivot3 = df.pivot_table(index=["Product", "Category"], values=["Amount"], aggfunc="sum")
# print(pivot3)

# # ( Get the Mean, Median, Minimum Sale by Category )
# pivot4 = df.pivot_table(index=["Category"], values=["Amount"], aggfunc={"median", "mean", "min"})
# print(pivot4)

# # ( Get the Mean, Median, Minimum Sale by Product )
# pivot5 = df.pivot_table(index=["Product"], values=["Amount"], aggfunc={"median", "mean", "min"})
# print(pivot5)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Creating a Series )

# import pandas as pd
# import numpy as np

# # ( Creating an Empty Pandas Series )
# ser = pd.Series()
# print(ser)

# # ( Creating a Series from a NumPy Array )
# data = np.array(['g', 'e', 'e', 'k', 's'])
# ser2 = pd.Series(data)
# print(ser2)

# # ( Creating a Series from a List )
# lst = ['g', 'e', 'e', 'k', 's']
# ser3 = pd.Series(lst)
# print(ser3)

# # ( Creating a Series from a Dictionary )
# data_dict = {'Geeks': 10, 'for': 20, 'geeks': 30}
# ser4 = pd.Series(data_dict)
# print(ser4)

# # ( Creating a Series Using NumPy Functions )
# ser5 = pd.Series(np.linspace(1, 10, 5))
# print(ser5)

# # ( Creating a Series Using range() )
# ser6 = pd.Series(range(5, 15))
# print(ser6)

# # ( Creating a Series Using List Comprehension )
# ser7 = pd.Series(range(1, 20, 3), index=[x for x in "abcdefg"])
# print(ser7)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Accessing elements of a Pandas Series --> Accessing Element from Series with Position )

# import pandas as pd
# import numpy as np

# data = np.array(["g", "e", "e", "k", "s", "f", "o", "r", "g", "e", "e", "k", "s"])
# ser = pd.Series(data)

# # ( Accessing the First Element of Series )
# print(ser[0])

# # ( Accessing First 5 Elements of Series )
# print(ser[:5])

# # ( Accessing Last 10 Elements of Series )
# print(ser[-10:])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Accessing elements of a Pandas Series --> Accessing First 5 Elements of Series in nba.csv File )

# import pandas as pd

# data = pd.read_csv('nba.csv')
# ser = pd.Series(data["Name"])

# # ( Accessing First 5 Elements of Series )
# print(ser.head())
# print(ser[:5])

# # ( Accessing a Multiple Element Using Index Label )
# print(ser[[0, 3, 6, 9]])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Accessing elements of a Pandas Series --> Access an Element in Pandas Using Label )

# import pandas as pd
# import numpy as np

# data = np.array(["g", "e", "e", "k", "s", "f", "o", "r", "g", "e", "e", "k", "s"])
# ser = pd.Series(data, index=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

# # ( Accessing a Single Element Using index Label )
# print(ser[16])

# # ( Accessing a Multiple Element Using index Label )
# print(ser[[10, 11, 12, 13, 14]])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Accessing elements of a Pandas Series --> Access Multiple Elements by Providing Label of Index )

# import pandas as pd
# import numpy as np

# ser = pd.Series(np.arange(3, 9), index=["a", "b", "c", "d", "e", "f"])
# print(ser[["a", "d"]])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Binary Operations on Series --> Binary Operations on Pandas Series )

# import pandas as pd

# s1 = pd.Series([10, 20, 30], index=["a", "b", "c"])
# s2 = pd.Series([1, 2, 3], index=["a", "b", "c"])

# # ( Arithmetic Operations on Series )
# add = s1 + s2
# print(add)

# # ( Comparison Operations on Series )
# equal = (s1 == s2)
# print(equal)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Binary Operations on Series --> Binary Operations on Pandas DataFrame )

# import pandas as pd

# df1 = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
# df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# # ( Arithmetic Operations on DataFrames )
# sub = df1 - df2
# print(sub)

# # ( Comparison Operations on DataFrames )
# equal = (df1 > df2)
# print(equal)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Binary Operations on Series --> Logical Operations on DataFrame and Series )

# import pandas as pd

# s1 = pd.Series([True, False, True])
# s2 = pd.Series([False, False, True])

# # ( Logical AND on Series )
# res = (s1 & s2)
# print(res)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Binary Operations on Series --> Handling Missing Data in Binary Operations )

# import pandas as pd

# df1 = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
# df2 = pd.DataFrame({'A': [1, None, 3], 'B': [None, 5, 6]})

# result = df1 + df2
# print(result)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Pandas Series Index() Methods --> Pandas Series Index() Methods )

# import pandas as pd

# ser = pd.Series(['New York', 'Chicago', 'Toronto', 'Lisbon'])

# ser.index = ['City 1', 'City 2', 'City 3', 'City 4']
# print(ser)

# # assigning duplicate or nonunique indexes in pandas
# ser.index = ['City 1', 'City 2', 'City 3', 'City 3']
# print(ser)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Pandas Series Index() Methods --> Find element’s index in pandas Series )

# import pandas as pd

# data = ['1/1/2018', '2/1/2018', '3/1/2018', '4/1/2018']
# index_name = ["Day 1", "Day 2", "Day 3", "Day 4"]
# ser = pd.Series(data=data, index=index_name)
# print(ser)
# print(ser.index)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Series --> Create a Pandas Series from Array )

# import pandas as pd
# import numpy as np

# # ( Creating a Pandas Series Without an Index )
# data = np.array(["a", "b", "c", "d", "e"])
# ser = pd.Series(data)
# print(ser)

# # ( Creating a Pandas Series With a Custom Index )
# data2 = np.array(["A", "B", "C", "D", "E"])
# ser2 = pd.Series(data2, index=[1000, 1001, 1002, 1003, 1004])
# print(ser2)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Read CSV Files with Pandas )

# import pandas as pd

# # ( Read specific columns using read_csv )
# df1 = pd.read_csv("people_data.csv", usecols=["First Name", "Email"])
# print(df1)

# # ( Setting an Index Column (index_col) )
# df2 = pd.read_csv("people_data.csv", index_col="First Name")
# print(df2)

# # ( Handling Missing Values Using read_csv )
# df3 = pd.read_csv("people_data.csv", na_values=["N/A", "Unknown"])
# print(df3)

# # ( Using nrows in read_csv() )
# df4 = pd.read_csv("people_data.csv", nrows=3)
# print(df4)

# # ( Using skiprows in read_csv() )
# df5 = pd.read_csv("people_data.csv", skiprows=[4, 5])
# print(df5)

# # ( Parsing Dates (parse_dates) )
# df6 = pd.read_csv("people_data.csv", parse_dates=["Date of birth"])
# print(df6.info())

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Reading CSV Files with Different Delimiters )

# import pandas as pd

# data = """totalbill_tip, sex:smoker, day_time, size, free
# 16.99, 1.01:Female|No, Sun, Dinner, 2
# 10.34, 1.66, Male, No|Sun:Dinner, 3
# 21.01:3.5_Male, No:Sun, Dinner, 3
# 23.68, 3.31, Male|No, Sun_Dinner, 2
# 24.59:3.61, Female_No, Sun, Dinner, 4
# 25.29, 4.71|Male, No:Sun, Dinner, 4"""

# with open("sample.csv", "w") as file:
#     file.write(data)

# df = pd.read_csv("sample.csv", sep="[:, |_]", engine="python")
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Loading a CSV Data from a URL )

# url = "https://media.geeksforgeeks.org/wp-content/uploads/20241121154629307916/people_data.csv"
# df = pd.read_csv(url)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Writing data to CSV Files --> Saving a Pandas Dataframe as a CSV )

# import pandas as pd

# name = ["aparna", "pankaj", "sudhir", "Geeku"]
# deg = ["MBA", "BCA", "M.Tech", "MBA"]
# scr = [90, 40, 80, 98]

# dict = {'name': name, 'degree': deg, 'score': scr}
# df = pd.DataFrame(dict)

# # ( Export CSV to a Working Directory )
# df.to_csv("file.csv")

# # ( Saving CSV Without Headers and Index )
# df.to_csv("file.csv", index=False, header=False)

# # ( Save the CSV file to a Specified Location )
# df.to_csv(r'D:\AI Engineer\file.csv')

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Writing data to CSV Files --> Write a DataFrame to CSV file using Tab Separator )

# import pandas as pd

# users = {"Name": ["Amit", "Cody", "Drew"], "Age": [20, 21, 25]}
# df = pd.DataFrame(users, columns=["Name", "Age"])

# df.to_csv("Users.csv", sep="\t", index=False)
# data = pd.read_csv("Users.csv")
# print(data)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Export Pandas dataframe to a CSV file )

# import pandas as pd

# scores = {"Name": ["a", "b", "c", "d"], "Score": [90, 80, 95, 20]}
# df = pd.DataFrame(scores)

# # ( Basic Export )
# df.to_csv("your_name.csv")

# # ( Remove Index Column )
# df.to_csv("your_name.csv", index=False)

# # ( Export only selected columns  )
# df.to_csv("your_name.csv", columns=["Name"], index=False)

# # ( Exclude Header Row )
# df.to_csv("your_name.csv", header=False)

# # ( Handling Missing Values  )
# df.to_csv("your_name.csv", na_rep="nothing")

# # ( Change Column Separator  )
# df.to_csv("your_name.csv", sep="\t", index=False)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Read JSON Files with Pandas )

# from io import StringIO
# import pandas as pd
# import json

# data = {
#     "One": {"0": 60, "1": 60, "2": 60, "3": 45, "4": 45, "5": 60},
#     "Two": {"0": 110, "1": 117, "2": 103, "3": 109, "4": 117, "5": 102},
# }

# # ( Read JSON Using Pandas pd.read_json() Method )
# df_json = pd.read_json(StringIO(json.dumps(data)), orient="index")
# print(df_json)

# # ( Using JSON module and pd.json_normalize() Method )
# json_data = json.dumps(data)
# df_normalize = pd.json_normalize(json.loads(json_data))
# print(df_normalize)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Parsing JSON Dataset --> Parsing in Pandas JSON Dataset Example )

# import pandas as pd

# df = pd.DataFrame(
#     [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
# )
# print(df.to_json(orient="split"))
# print(df.to_json(orient="index"))
# print(df.to_json(orient="records"))
# print(df.to_json(orient="table"))
# print(df.to_json(orient="values"))
# print(df.to_json(orient="columns"))

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Exporting Pandas DataFrame to JSON File --> Exporting a Simple DataFrame )

# import pandas as pd

# df = pd.DataFrame(
#     [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
#     index=["row 1", "row 2", "row 3"],
#     columns=["col 1", "col 2", "col 3"],
# )

# df.to_json("file.json", orient="split", compression="infer", index=True)
# df = pd.read_json("file.json", orient="split", compression="infer")
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Exporting Pandas DataFrame to JSON File --> Exporting a More Detailed DataFrame )

# import pandas as pd

# df = pd.DataFrame(
#     data=[
#         ["15135", "Alex", "25/4/2014"],
#         ["23515", "Bob", "26/8/2018"],
#         ["31313", "Martha", "18/1/2019"],
#         ["55665", "Alen", "5/5/2020"],
#         ["63513", "Maria", "9/12/2020"],
#     ],
#     columns=["ID", "NAME", "DATE OF JOINING"],
# )
# df.to_json("file.json", orient="split", compression="infer")
# df = pd.read_json("file.json", orient="split", compression="infer")
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Text File to CSV using Python Pandas --> Convert a Simple Text File to CSV )

# import pandas as pd

# df = pd.read_csv("main.txt")
# df.to_csv("main.csv")
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Text File to CSV using Python Pandas --> Handling Text Files Without Headers )

# import pandas as pd

# website = pd.read_csv("main.txt", header=None)
# website.columns = ["totalbill", "sex", "time", "male", "size"]
# website.to_csv("main.csv", index=None)
# print(website)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Input and Output (I/O) --> Text File to CSV using Python Pandas --> Handling Custom Delimiters )

# import pandas as pd

# account = pd.read_csv("main.txt", delimiter="/")
# account.to_csv("main.csv", index=None)
# print(account)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Checking for Missing Values Using isnull() )

# import pandas as pd
# import numpy as np

# data = {
#     "First Score": [100, 90, np.nan, 95],
#     "Second Score": [30, 45, 56, np.nan],
#     "Third Score": [np.nan, 40, 80, 98],
# }
# df = pd.DataFrame(data)
# missing_values = df.isnull()
# print(missing_values)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Filtering Data Based on Missing Values )

# import pandas as pd

# data = pd.read_csv("data.csv")
# bool_series = pd.isnull(data["Calories"])
# missing_data = data[bool_series]
# print(missing_data)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Detecting Non-Missing Values in a DataFrame )

# import pandas as pd
# import numpy as np

# data = {
#     "First Score": [100, 90, np.nan, 95],
#     "Second Score": [30, 45, 56, np.nan],
#     "Third Score": [np.nan, 40, 80, 98],
# }
# df = pd.DataFrame(data)
# non_missing_values = df.notnull()
# print(non_missing_values)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Filtering Data with Non-Missing Values )

# import pandas as pd

# data = pd.read_csv("data.csv")
# non_missing_values = pd.notnull(data["Calories"])
# non_missing_data = data[non_missing_values]
# print(non_missing_data)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Filling Missing Values with a Specific Value Using fillna() )

# import pandas as pd
# import numpy as np

# dic = {
#     "First Score": [100, 90, np.nan, 95],
#     "Second Score": [30, 45, 56, np.nan],
#     "Third Score": [np.nan, 40, 80, 98],
# }
# df = pd.DataFrame(dic)
# df.fillna(0, inplace=True)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Filling Missing Values with the Prev/Next Value Using fillna )

# import pandas as pd
# import numpy as np

# dic = {
#     "First Score": [100, 90, np.nan, 95],
#     "Second Score": [30, 45, 56, np.nan],
#     "Third Score": [np.nan, 40, 80, 98],
# }
# df = pd.DataFrame(dic)

# # ( Fill with Previous Value (Forward Fill) )
# df.ffill(inplace=True)
# print(df)

# # ( Fill with Next Value (Backward Fill) )
# df.bfill(inplace=True)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Fill NaN Values with ‘No Gender’ using fillna() )

# import pandas as pd

# data = pd.read_csv("employees.csv")
# data["Gender"].fillna("No Gender", inplace=True)
# print(data[10:25])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Replacing Missing Values Using replace() )

# import pandas as pd
# import numpy as np

# data = pd.read_csv("employees.csv")
# data.replace(to_replace=np.nan, value=-99, inplace=True)
# print(data[10:25])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Filling Missing Values Using interpolate() )

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "A": [12, 4, 5, None, 1],
#         "B": [None, 2, 54, 3, None],
#         "C": [20, 16, None, 3, 8],
#         "D": [14, 3, None, None, 6],
#     }
# )
# df.interpolate(method="linear", limit_direction="both", inplace=True)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Dropping Rows with At Least One Null Value )

# import pandas as pd
# import numpy as np

# dict = {
#     "First Score": [100, 90, np.nan, 95],
#     "Second Score": [30, np.nan, 45, 56],
#     "Third Score": [52, 40, 80, 98],
#     "Fourth Score": [np.nan, np.nan, np.nan, 65],
# }
# df = pd.DataFrame(dict)

# df_drop = df.dropna()
# print(df_drop)

# df.dropna(inplace=True)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Dropping Rows with All Null Values )

# import pandas as pd
# import numpy as np

# data = {
#     "First Score": [100, np.nan, np.nan, 95],
#     "Second Score": [30, np.nan, 45, 56],
#     "Third Score": [52, np.nan, 80, 98],
#     "Fourth Score": [np.nan, np.nan, np.nan, 65],
# }
# df = pd.DataFrame(data)
# df_drop = df.dropna(how="all")
# print(df_drop)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Dropping Columns with At Least One Null Value )

# import pandas as pd
# import numpy as np

# data = {
#     "First Score": [100, np.nan, np.nan, 95],
#     "Second Score": [30, np.nan, 45, 56],
#     "Third Score": [52, 40, 80, 98],
#     "Fourth Score": [np.nan, np.nan, np.nan, 65],
# }
# df = pd.DataFrame(data)
# df_drop = df.dropna(axis=1)
# print(df_drop)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Working with Missing Data in Pandas --> Dropping Rows with Missing Values in CSV Files )

# import pandas as pd

# data = pd.read_csv("employees.csv")
# new_data = data.dropna(axis=0, how="any")
# print(f"Data: {len(data)}")
# print(f"New Data: {len(new_data)}")
# print(f"Rows with at least one missing value: {len(data) - len(new_data)}")

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Removing Duplicates )

# import pandas as pd

# data = {
#     "Name": ["Alice", "Bob", "Alice", "David"],
#     "Age": [25, 30, 25, 40],
#     "City": ["NY", "LA", "NY", "Chicago"]
# }
# df = pd.DataFrame(data)

# # ( Dropping Duplicates )
# result = df.drop_duplicates()
# print(result)

# # ( Dropping Duplicates Based on Specific Columns )
# result2 = df.drop_duplicates(subset="Name")
# print(result2)

# # ( Keeping the Last Occurrence )
# result3 = df.drop_duplicates(keep="last")
# print(result3)

# # ( Dropping All Duplicates )
# result4 = df.drop_duplicates(keep=False)
# print(result4)

# # ( Modifying the Original DataFrame Directly )
# df.drop_duplicates(inplace=True)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Pandas Change Datatype )

# import pandas as pd

# data = {
#     "Name": ["John", "Alice", "Bob", "Eve", "Charlie"],
#     "Age": [25, 30, 22, 35, 28],
#     "Gender": ["Male", "Female", "Male", "Female", "Male"],
#     "Salary": [50000, 55000, 40000, 70000, 48000],
# }
# df = pd.DataFrame(data)

# # ( Using astype() method )
# df["Age"] = df["Age"].astype(float)
# print(df.dtypes)

# # ( Converting a Column to a DateTime Type )
# df["Join Date"] = ['2021-01-01', '2020-05-22', '2022-03-15', '2021-07-30', '2020-11-11']
# df["Join Date"] = pd.to_datetime(df["Join Date"])
# print(df.dtypes)

# # ( Changing Multiple Columns' Data Types )
# df = df.astype({'Age': 'float64', 'Salary': 'str'})
# print(df.dtypes)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> Drop Empty Columns in Pandas )

# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     {
#         "FirstName": ["Vipul", "Ashish", "Milan"],
#         "Gender": ["", "", ""],
#         "Age": [0, 0, 0],
#     }
# )
# df["Department"] = np.nan

# # ( Remove All Null Value Columns )
# df.dropna(how="all", axis=1, inplace=True)
# # print(df)

# # ( Replace Empty Strings with Null and Drop Null Columns )
# nan_value = float("NaN")
# df.replace("", nan_value, inplace=True)
# df.dropna(how="all", axis=1, inplace=True)
# print(df)

# # ( Replace Zeros with Null and Drop Null Columns )
# nan_value = float("NaN")
# df.replace(0, nan_value, inplace=True)
# df.dropna(how="all", axis=1, inplace=True)
# print(df)

# # ( Replace Both Zeros and Empty Strings with Null and Drop Null Columns )
# nan_value = float("NaN")
# df.replace(0, nan_value, inplace=True)
# df.replace("", nan_value, inplace=True)
# df.dropna(how="all", axis=1, inplace=True)
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> String manipulations in Pandas )

# import pandas as pd
# import numpy as np

# data = {
#     "Names": ["Gulshan", "Shashank", "Bablu", "Abhishek", "Anand", np.nan, "Pratap"],
#     "City": [
#         "Delhi",
#         "Mumbai",
#         "Kolkata",
#         "Delhi",
#         "Chennai",
#         "Bangalore",
#         "Hyderabad",
#     ],
# }
# df = pd.DataFrame(data)

# print(df["Names"].str.lower())
# print(df["Names"].str.upper())
# print(df["Names"].str.strip())
# df["Split Names"] = df["Names"].str.split("a")
# print(df[["Names", "Split Names"]])
# print(df["Names"].str.len())
# print(df["Names"].str.cat(sep=" --> "))
# print(df["City"].str.get_dummies())
# print(df["Names"].str.startswith("G"))
# print(df["Names"].str.endswith("h"))
# print(df["Names"].str.replace("Shashank", "Tom"))
# print(df["Names"].str.repeat(2))
# print(df["Names"].str.count("a"))
# print(df["Names"].str.find("a"))
# print(df["Names"].str.findall("a"))
# print(df["Names"].str.islower())
# print(df["Names"].str.isupper())
# print(df["Names"].str.isnumeric())
# print(df["Names"].str.swapcase())
# print(df["Names"].str.encode(encoding="utf-16"))
# print(df["Names"].str.encode(encoding="ascii"))

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Data Cleaning in Pandas --> String methods in Pandas )

# import pandas as pd

# data = pd.DataFrame(
#     [["tom", 10], ["nick", "15"], ["juli", 14.8]], columns=["Name", "Age"]
# )

# # ( How to identify mixed types in Pandas columns )
# for column in data.columns:
#     print(f"{column} : {pd.api.types.infer_dtype(data[column])}")

# # ( How to deal with mixed types in Pandas columns )
# data["Age"] = data["Age"].astype(int)

# for column in data.columns:
#     print(f"{column} : {pd.api.types.infer_dtype(data[column])}")

# # ( Using to_numeric() function )
# data["Age"] = data["Age"].apply(lambda x: pd.to_numeric(x, errors="ignore"))

# for column in data.columns:
#     print(f"{column} : {pd.api.types.infer_dtype(data[column])}")

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Operations --> Data Processing with Pandas )

# import pandas as pd

# data = pd.read_csv("data.csv")

# # ( Removing rows )
# data.drop(3, inplace=True)
# print(data.head())

# # ( Renaming rows )
# data.rename({0: "1st", 1:"2nd"}, inplace=True)
# print(data.head())

# # ( Sort by column )
# data.sort_values(by="Duration", inplace=True)
# print(data.head())

# # ( Sort by multiple columns )
# data.sort_values(by=["Duration", "Pulse"], inplace=True)
# print(data)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Operations --> Pandas dataframe.groupby() Method )

# import pandas as pd

# data = pd.read_csv("nba.csv")

# # ( Grouping by a Single Column )
# team = data.groupby("Team")
# print(team.first())

# # ( Grouping by Multiple Columns )
# grouping = data.groupby(["Team", "Position"])
# print(grouping.first())

# # ( Applying Aggregation with GroupBy )
# aggregat = data.groupby(["Team", "Position"]).agg(
#     Total_salary = ("Salary", "sum"),
#     Avg_salary = ("Salary", "mean"),
#     Player_count = ("Name", "count")
# )
# print(aggregat)

# # ( How to Apply Transformation Methods )
# data["Rank within Team"] = data.groupby("Team")["Salary"].transform(lambda x: x.rank(ascending=False))
# print(data.head())

# # ( Filtering Groups Using Filtration Methods )
# filtered_data = data.groupby("Team").filter(lambda x: x["Salary"].mean() >= 1000000)
# print(filtered_data)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Pandas Operations --> Different Types of Joins in Pandas )

# import pandas as pd

# data1 = {"id": [1, 2, 10, 12], "val1": ["a", "b", "c", "d"]}
# df1 = pd.DataFrame(data1)

# data2 = {"id": [1, 2, 9, 8], "val1": ["p", "q", "r", "s"]}
# df2 = pd.DataFrame(data2)

# # ( Pandas Inner Join )
# inner = pd.merge(df1, df2, on="id", how="inner")
# print(inner)

# # ( Pandas Left Join )
# left = pd.merge(df1, df2, on="id", how="left")
# print(left)

# # ( Pandas Right Join )
# right = pd.merge(df1, df2, on="id", how="right")
# print(right)

# # ( Pandas Full Outer Join )
# outer = pd.merge(df1, df2, on="id", how="outer")
# print(outer)

# # ( Pandas Index Join )
# index = pd.merge(df1, df2, left_index=True, right_index=True)
# print(index)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Advanced Pandas Operations --> Finding Correlation between Data )

# import pandas as pd

# df = pd.read_csv("data.csv")
# print(df.corr(method="pearson"))
# print(df.corr(method="kendall"))
# print(df.corr(method="spearman"))

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Advanced Pandas Operations --> Basic of Time Series Manipulation Using Pandas )

# import pandas as pd

# # ( Create DateTime Values with Pandas )
# range_date = pd.date_range(start="20/3/2025", end="15/4/2025", freq="Min")
# print(range_date)

# # ( Determine the Data Type of an Element in the DateTime Range )
# print(type(range_date))

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Advanced Pandas Operations --> Basic of Time Series Manipulation Using Pandas --> Create DataFrame with DateTime Index )

# import pandas as pd
# import numpy as np

# rd = pd.date_range(start="20/3/2025", end="15/4/2025", freq="Min")
# df = pd.DataFrame(rd, columns=["date"])
# df["date"] = np.random.randint(0, 100, size=len(rd))
# print(df)

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Advanced Pandas Operations --> Basic of Time Series Manipulation Using Pandas --> Convert DateTime elements to String format )

# import pandas as pd
# import numpy as np

# rd = pd.date_range(start="20/3/2025", end="15/4/2025", freq="Min")
# df = pd.DataFrame(rd, columns=["Date"])
# df["Date"] = np.random.randint(0, 100, size=len(rd))

# string = [str(x) for x in rd]
# print(string[1:11])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Advanced Pandas Operations --> Basic of Time Series Manipulation Using Pandas --> Accessing Specific DateTime Element )

# import pandas as pd
# import numpy as np

# rd = pd.date_range(start="20/3/2025", end="15/4/2025", freq="Min")
# df = pd.DataFrame(rd, columns=["datetime"])
# df["data"] = np.random.randint(0, 100, size=len(rd))

# df.set_index("datetime", inplace=True)
# filtered_df = df.loc["2025-04-05"]
# print(filtered_df.iloc[1:11])

# ------------------------------------------------------------------------------

# ( Pandas Tutorial --> Advanced Pandas Operations --> Time Series Analysis & Visualization in Python )

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import adfuller

# data = pd.read_csv("stock_data.csv")
# data.drop(columns="Unnamed: 0", inplace=True)

# # ( Plotting Line plot for Time Series data )
# sns.set_style(style="whitegrid")
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=data, x="Date", y="High", label="High Price", color="b")

# plt.xlabel("Date")
# plt.ylabel("High")
# plt.title("Share Highest Price Over Time")
# plt.show()

# ------------------------------------------------------------------------------

# ((((((((((((((( Data Preprocessing )))))))))))))))

# ( Data Formatting --> Formatting float column of the data frame in Pandas --> Round off the Column Values to Two Decimal Places )

# import pandas as pd

# data = {
#     "Month": ["January", "February", "March", "April"],
#     "Expense": [21525220.653, 31125840.875, 23135428.768, 56245263.942],
# }
# dataframe = pd.DataFrame(data, columns=["Month", "Expense"])
# print(f"Given Datafram:\n{dataframe}")

# # round to two decimal places in python pandas
# pd.options.display.float_format = "{:.2f}".format
# print(f"Result:\n{dataframe}")

# ------------------------------------------------------------------------------

# ( Data Formatting --> Formatting float column of the data frame in Pandas --> Pandas DataFrame Formatting with Commas and Decimal Precision )

# import pandas as pd

# data = {
#     "Product": ["Laptop", "Phone", "Tablet", "Desktop"],
#     "Price": [1200.50, 799.99, 349.75, 1500.25],
# }
# products_dataframe = pd.DataFrame(data, columns=["Product", "Price"])
# pd.options.display.float_format = "{:.2f}".format

# # Create a new DataFrame with formatted values
# formatted_products = products_dataframe.copy()
# formatted_products["Price"] = formatted_products["Price"].apply(lambda x: "{:.2f}".format(x))
# print(formatted_products)

# ------------------------------------------------------------------------------

# ( Data Formatting --> Formatting float column of the data frame in Pandas --> Formatting and Scaling Population Data in Pandas DataFrame )

# import pandas as pd

# data = {
#     "City": ["New York", "Los Angeles", "Chicago", "Houston"],
#     "Population": [8336817, 3980400, 2716000, 2328000],
# }
# city_dataframe = pd.DataFrame(data, columns=["City", "Population"])

# # Format with commas and display population in millions
# pd.options.display.float_format = "{:,.2f}".format
# city_dataframe["Population"] = city_dataframe["Population"] / 1000000
# print(city_dataframe)

# ------------------------------------------------------------------------------

# ( Data Formatting --> How to Check the Data Type in Pandas DataFrame )

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "Cust_No": [1, 2, 3],
#         "Cust_Name": ["Alex", "Bob", "Sophie"],
#         "Product_id": [12458, 48484, 11311],
#         "Product_cost": [65.25, 25.95, 100.99],
#         "Purchase_Date": [
#             pd.Timestamp("20180917"),
#             pd.Timestamp("20190910"),
#             pd.Timestamp("20200610"),
#         ],
#     }
# )
# # Print a list datatypes of all columns
# print(df.dtypes)

# # print datatype of particular column
# print(df.Cust_No.dtypes)

# # Checking the Data Type of a Particular Column
# print(df["Product_cost"].dtype)

# # ( Check the Data Type in Pandas using pandas.DataFrame.select_dtypes  )

# # Returns Two column of int64
# print(df.select_dtypes(include="int64"))

# # Returns columns excluding int64
# print(df.select_dtypes(exclude="int64"))

# ------------------------------------------------------------------------------

# ( Data Formatting --> How to change the Pandas datetime format in Python? )

# import pandas as pd

# date_sr = pd.Series(
#     pd.date_range("2024-12-31", periods=3, freq="ME", tz="Asia/Calcutta")
# )
# ind = ["Day 1", "Day 2", "Day 3"]
# date_sr.index = ind
# change_format = date_sr.dt.strftime("%d-%m-%Y")
# print(change_format)

# ------------------------------------------------------------------------------

# ( Data Formatting --> Convert the column type from string to datetime format in Pandas data frame --> Pandas Convert Column To DateTime using pd.to_datetime() )

# import pandas as pd

# df1 = pd.DataFrame(
#     {
#         "Date": ["11/8/2011", "04/23/2008", "10/2/2019"],
#         "Event": ["Music", "Poetry", "Theatre"],
#         "Cost": [10000, 5000, 15000],
#     }
# )
# print(f"Before Conversion:\n{df1}\n{df1.info()}")

# df1["Date"] = pd.to_datetime(df1["Date"])
# print(f"After Coversion:\n{df1}\n{df1.info()}")

# ------------------------------------------------------------------------------

# ( Data Formatting --> Convert the column type from string to datetime format in Pandas data frame --> Converting from ‘yymmdd’ Format )

# import pandas as pd

# player_list = [
#     ["200712", 50000],
#     ["200714", 51000],
#     ["200716", 51500],
#     ["200719", 53000],
#     ["200721", 54000],
#     ["200724", 55000],
#     ["200729", 57000],
# ]
# df1 = pd.DataFrame(player_list, columns=["Date", "Patinets"])
# print(f"Before Conversion:\n{df1}\n{df1.dtypes}")

# df1["Date"] = pd.to_datetime(df1["Date"], format="%y%m%d")
# print(f"After Conversion:\n{df1}\n{df1.dtypes}")

# ------------------------------------------------------------------------------

# ( Data Formatting --> Convert the column type from string to datetime format in Pandas data frame --> Converting from ‘yyyymmdd’ Format )

# import pandas as pd

# df = pd.DataFrame(
#     {
#         "Date": ["11/8/2011", "04/23/2008", "10/2/2019"],
#         "Event": ["Music", "Poetry", "Theatre"],
#         "Cost": [10000, 5000, 15000],
#     }
# )
# print(f"Before Coversion:\n{df}\n{df.info()}")

# df["Date"] = df["Date"].astype("datetime64[ns]")
# print(f"After Coversion:\n{df}\n{df.info()}")

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Missing values --> Drop rows from Pandas dataframe with missing values or NaN in columns --> Using notna() )

# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     {
#         "A": [100, np.nan, np.nan, 95],
#         "B": [30, np.nan, 45, 56],
#         "C": [52, np.nan, 80, 98],
#         "D": [np.nan, np.nan, np.nan, 65],
#     }
# )
# print(f"Original Data:\n{df}")
# print(f"Drop rows with at least 1 NaN:\n{df[df.notna().all(axis=1)]}")
# print(f"Drop rows where all values are NaN:\n{df[df.notna().any(axis=1)]}")
# print(f"Drop columns with at least 1 NaN:\n{df.loc[:, df.notna().all()]}")

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Missing values --> Drop rows from Pandas dataframe with missing values or NaN in columns --> Using query() )

# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     {
#         "A": [100, np.nan, np.nan, 95],
#         "B": [30, np.nan, 45, 56],
#         "C": [52, np.nan, 80, 98],
#         "D": [np.nan, np.nan, np.nan, 65],
#     }
# )
# print(f"Original Data:\n{df}")
# print(
#     f"Drop rows with at least 1 NaN:\n{df.query('A == A and B == B and C == C and D == D')}"
# )

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Missing values --> Drop rows from Pandas dataframe with missing values or NaN in columns --> Using isna() with mask() )

# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     {
#         "A": [100, np.nan, np.nan, 95],
#         "B": [30, np.nan, 45, 56],
#         "C": [52, np.nan, 80, 98],
#         "D": [np.nan, np.nan, np.nan, 65],
#     }
# )
# result = df.mask(df.isna())
# print(f"Drop rows with at least 1 NaN:\n{result.dropna()}")
# print(f"Drop columns with at least 1 NaN:\n{result.dropna(axis=1)}")

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Missing values --> Handling Missing Values --> Imputation Methods )

# import pandas as pd
# import numpy as np

# data = {
#     "School ID": [101, 102, 103, np.nan, 105, 106, 107, 108],
#     "Name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry"],
#     "Address": [
#         "123 Main St",
#         "456 Oak Ave",
#         "789 Pine Ln",
#         "101 Elm St",
#         np.nan,
#         "222 Maple Rd",
#         "444 Cedar Blvd",
#         "555 Birch Dr",
#     ],
#     "City": [
#         "Los Angeles",
#         "New York",
#         "Houston",
#         "Los Angeles",
#         "Miami",
#         np.nan,
#         "Houston",
#         "New York",
#     ],
#     "Subject": [
#         "Math",
#         "English",
#         "Science",
#         "Math",
#         "History",
#         "Math",
#         "Science",
#         "English",
#     ],
#     "Marks": [85, 92, 78, 89, np.nan, 95, 80, 88],
#     "Rank": [2, 1, 4, 3, 8, 1, 5, 3],
#     "Grade": ["B", "A", "C", "B", "D", "A", "C", "B"],
# }
# df = pd.DataFrame(data)

# # ( Mean, Median, and Mode Imputation )
# mean = df["Marks"].fillna(df["Marks"].mean())
# median = df["Marks"].fillna(df["Marks"].median())
# mode = df["Marks"].fillna(df["Marks"].mode())

# print(mean)
# print(median)
# print(mode)

# # ( Forward and Backward Fill )
# forward_fill = df["Marks"].ffill()
# backward_fill = df["Marks"].bfill()

# print(forward_fill)
# print(backward_fill)

# # ( Interpolation Techniques )
# linear_interpolation = df["Marks"].interpolate(method="linear")
# quadratic_interpolation = df["Marks"].interpolate(method="quadratic")

# print(linear_interpolation)
# print(quadratic_interpolation)

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Missing values --> Handle Missing Data with Simple Imputer )

# import numpy as np
# from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# data = [[12, np.nan, 34], [10, 32, np.nan], [np.nan, 11, 20]]
# print(f"Original Data:\n{data}")

# imputer_data = imputer.fit(data)
# data2 = imputer_data.transform(data)

# print(data2)

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Outliers Detection --> Detect and Remove the Outliers using Python --> Outlier Detection And Removal )

# import pandas as pd
# import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import load_diabetes

# diabates = load_diabetes()

# # ( Create the dataframe )
# column_name = diabates.feature_names
# df_diabates = pd.DataFrame(diabates.data)
# df_diabates.columns = column_name
# print(df_diabates.head())

# # ( visualizing the data )
# sns.boxplot(df_diabates["bmi"])
# plt.show()


# # ( Removing Outliers )
# def removal_box_plot(df, column, threshold):
#     sns.boxplot(df[column])
#     plt.title(f"Original Box Plot of {column}")
#     plt.show()

#     removed_outliers = df[df[column] <= threshold]

#     sns.boxplot(removed_outliers[column])
#     plt.title(f"Box Plot without Outliers of {column}")
#     plt.show()
#     return removed_outliers


# threshold_value = 0.12
# no_outliers = removal_box_plot(df_diabates, "bmi", threshold_value)

# # ( Visualizing Outliers Using Scatterplot )
# fig, ax_outlier = plt.subplots(figsize=(6, 4))
# ax_outlier.scatter(df_diabates["bmi"], df_diabates["bp"])
# ax_outlier.set_xlabel("(body mass index of people)")
# ax_outlier.set_ylabel("(bp of the people )")
# plt.show()

# # ( Removing Outliers Using Scatterplot )
# outlier_indices = np.where((df_diabates["bmi"] > 0.12) & (df_diabates["bp"] < 0.8))
# no_outlier = df_diabates.drop(outlier_indices[0])

# fig, ax_no_outlier = plt.subplots(figsize=(6, 4))
# ax_no_outlier.scatter(no_outlier["bmi"], no_outlier["bp"])
# ax_no_outlier.set_xlabel("(body mass index of people)")
# ax_no_outlier.set_ylabel("(bp of the people )")
# plt.show()

# # ( Z-score )
# z_score = np.abs(stats.zscore(df_diabates["age"]))
# print(z_score)

# # ( Removal of Outliers with Z-Score )
# threshold_z = 2
# outlier_indices = np.where(z_score > threshold_z)[0]
# no_outliers = df_diabates.drop(outlier_indices)
# print(f"Original DataFrame Shape: {df_diabates.shape}")
# print(f"DataFrame Shape after Removing Outliers: {no_outliers.shape}")

# # ( IQR (Inter Quartile Range) )
# q1 = np.percentile(df_diabates["bmi"], 25, method="midpoint")
# q3 = np.percentile(df_diabates["bmi"], 75, method="midpoint")
# iqr = q3 - q1
# print(iqr)

# upper = q3 + 1.5 * iqr
# lower = q1 - 1.5 * iqr
# upper_array = np.array(df_diabates["bmi"] >= upper)
# lower_array = np.array(df_diabates["bmi"] <= lower)

# print(f"Upper Bound: {upper}")
# print(upper_array.sum())
# print(f"Lower Bound: {lower}")
# print(lower_array.sum())

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Outliers Detection --> Detect and Remove the Outliers using Python --> Outlier Removal in Dataset using IQR )

# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_diabetes

# diabetes = load_diabetes()

# column_name = diabetes.feature_names
# df = pd.DataFrame(diabetes.data)
# df.columns = column_name
# print(f"Old shape: {df.shape}")

# # ( Calculate the upper and lower limits )
# q1 = df["bmi"].quantile(0.25)
# q3 = df["bmi"].quantile(0.75)
# iqr = q3 - q1
# lower = q1 - 1.5 * iqr
# upper = q3 + 1.5 * iqr

# upper_array = np.where(df["bmi"] >= upper)[0]
# lower_array = np.where(df["bmi"] <= lower)[0]

# df.drop(index=upper_array, inplace=True)
# df.drop(index=lower_array, inplace=True)
# print(f"New shape: {df.shape}")

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Outliers Detection --> Z-score for outlier Detection )

# import pandas as pd
# import numpy as np
# from scipy.stats import zscore
# import matplotlib.pyplot as plt

# data = [1, 2, 2, 2, 3, 1, 1, 15, 2, 2, 2, 3, 1, 1, 2]
# df = pd.DataFrame(data, columns=["Value"])

# # ( Calculate Z-Scores )
# df["Z-score"] = zscore(df["Value"])
# print(df)

# # ( Identify Outliers )
# outliers = df[df["Z-score"].abs() > 3]
# print(outliers)

# # ( Visualize the Data )
# plt.figure(figsize=(8, 6))
# plt.scatter(df["Value"], np.zeros_like(df["Value"]), color="b", label="Data Points")
# plt.scatter(
#     outliers["Value"], np.zeros_like(outliers["Value"]), color="r", label="Outliers"
# )
# plt.title("Outlier Detection using Z-Score")
# plt.xlabel("Value")
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Binning --> Binning method for data smoothing )

# import numpy as np
# from sklearn.datasets import load_iris

# # ( load iris data set )
# dataset = load_iris()
# data = dataset.data
# zero = np.zeros(150)

# # ( take 1st column among 4 column of data set )
# for i in range(150):
#     zero[i] = data[i, 1]

# zero = np.sort(zero)

# bin1 = np.zeros((30, 5))
# bin2 = np.zeros((30, 5))
# bin3 = np.zeros((30, 5))

# # ( Bin Mean )
# for i in range(0, 150, 5):
#     k = int(i / 5)
#     mean = (zero[i] + zero[i + 1] + zero[i + 2] + zero[i + 3] + zero[i + 4]) / 5
#     for j in range(5):
#         bin1[k, j] = mean
# print(f"Bin Mean:\n{bin1}")

# # ( Bin Boundaries )
# for i in range(0, 150, 5):
#     k = int(i / 5)
#     for j in range(5):
#         if (zero[i + j] - zero[i]) < (zero[i + 4] - zero[i + j]):
#             bin2[k, j] = zero[i]
#         else:
#             bin2[k, j] = zero[i + 4]
# print(f"Bin Boundaries:\n{bin2}")

# # ( Bin Median )
# for i in range(0, 150, 5):
#     k = int(i / 5)
#     for j in range(5):
#         bin3[k, j] = zero[i + 2]
# print(f"Bin Median:\n{bin3}")

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Isolation Forest for outlier detection --> Local Outlier Factor (LOF) )

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.neighbors import LocalOutlierFactor

# # ( Load the datasets )
# df = load_iris(as_frame=True).frame
# x = df[["sepal length (cm)", "sepal width (cm)"]]

# # ( Define the model and set the number of neighbors )
# lof = LocalOutlierFactor(n_neighbors=5)

# # ( Fit the model to the data )
# lof.fit(x)

# # ( Calculate the outlier scores for each point )
# scores = lof.negative_outlier_factor_

# # ( Identify the points with the highest outlier scores )
# outliers = np.argwhere(scores > np.percentile(scores, 95))

# for i in range(len(x)):
#     if i not in outliers:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="g")
#     else:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="r")

# plt.xlabel("sepal length (cm)", fontsize=13)
# plt.ylabel("sepal width (cm)", fontsize=13)
# plt.title("Anomly by Local Outlier Factor", fontsize=16)
# plt.show()

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Isolation Forest for outlier detection --> Isolation Forest )

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.ensemble import IsolationForest

# df = load_iris(as_frame=True).frame
# x = df[["sepal length (cm)", "sepal width (cm)"]]

# # ( Define the model and set the contamination level )
# model = IsolationForest(contamination=0.05)

# model.fit(x)
# scores = model.decision_function(x)
# outliers = np.argwhere(scores < np.percentile(scores, 5))

# for i in range(len(x)):
#     if i not in outliers:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="g")
#     else:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="r")

# plt.xlabel("sepal length (cm)", fontsize=13)
# plt.ylabel("sepal width (cm)", fontsize=13)
# plt.title("Anomly by Isolation Forest", fontsize=16)
# plt.show()

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Isolation Forest for outlier detection --> One-class Support Vector Machines (SVMs) )

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn import svm

# df = load_iris(as_frame=True).frame
# x = df[["sepal length (cm)", "sepal width (cm)"]]

# # ( Define the model and set the nu parameter )
# model = svm.OneClassSVM(nu=0.05)

# model.fit(x)
# scores = model.decision_function(x)
# outliers = np.argwhere(scores < np.percentile(scores, 5))

# for i in range(len(x)):
#     if i not in outliers:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="g")
#     else:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="r")

# plt.xlabel("sepal length (cm)", fontsize=13)
# plt.ylabel("sepal width (cm)", fontsize=13)
# plt.title("Anomly by One-class Support Vector Machines", fontsize=16)
# plt.show()

# ------------------------------------------------------------------------------

# ( Data Cleaning --> Isolation Forest for outlier detection --> Elliptic Envelope )

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.covariance import EllipticEnvelope

# df = load_iris(as_frame=True).frame
# x = df[["sepal length (cm)", "sepal width (cm)"]]

# # ( Define the model and set the contamination level )
# model = EllipticEnvelope(contamination=0.05)

# model.fit(x)
# scores = model.decision_function(x)
# outliers = np.argwhere(scores < np.percentile(scores, 5))

# for i in range(len(x)):
#     if i not in outliers:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="g")
#     else:
#         plt.scatter(x.iloc[i, 0], x.iloc[i, 1], color="r")

# plt.xlabel("sepal length (cm)", fontsize=13)
# plt.ylabel("sepal width (cm)", fontsize=13)
# plt.title("Anomly by Elliptic Envelope", fontsize=16)
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd

# data = pd.read_csv(
#     "data3.csv",
# )
# data.set_index("online_order", inplace=True)
# data.reset_index(inplace=True, drop=True)
# con = data[data["online_order"].(["Yes"])]
# con = data.pivot_table(index="name", aggfunc="count")
# con = data.loc(axis=1)[["name", "online_order"]]
# con = data.iloc(axis=1)[[0, 1]]

# data.set_index("name", inplace=True)
# data.sort_values("name", inplace=True, ignore_index=True)

# print(data.loc[0 : 2, "name" : "book_table"])
# print(data.iloc[0 : 3, 0 : 3])

# data.set_index(["name", "online_order", "book_table"], inplace=True)
# data.loc[("San Churro Cafe", "Yes") : ("Grand Village" , "No"), "book_table": "votes"]
# print(data.loc[("San Churro Cafe", "Yes") : ("Grand Village" , "No"), "book_table": "votes"])
# print(data.sort_index())

# con = data.pivot_table("votes", index="name", columns="online_order", fill_value=0)
# print(con.mean())

# data["new"] = data[["votes", "approx_cost(for two people)"]].agg("mean", axis=1).so

# ------------------------------------------------------------------------------

# import pandas as pd

# data1 = pd.read_csv("data3.csv")
# data2 = pd.read_csv("data4.csv")
# data3 = pd.read_csv("data5.csv")
# mer = data1.merge(data2, on = "name") \
# .merge(data3, on = "Last Login Time")

# mer1 = data1.merge(data2, on="name", indicator=True)
# semi = data1[data1["book_table"].isin(data2["book_table"])]
# print(mer1)

# m1 = pd.merge_asof(data1, data3, left_on="votes", right_on="Salary", direction="forward")
# m1 = pd.merge_ordered()
# print(m1)

# ------------------------------------------------------------------------------

# import pandas as pd

# data1 = pd.read_csv("data3.csv")
# data2 = pd.read_csv("data4.csv")
# data3 = pd.read_csv("data5.csv")

# con1 = pd.concat([data1, data2], ignore_index=True)
# con2 = pd.concat([data1, data2], keys=["table_1", "table_2"])
# con3 = pd.concat([data1, data2], keys=["table_1", "table_2"], sort=True)
# con4 = pd.concat([data1, data2], keys=["table_1", "table_2"], join="outer")
# con5 = pd.concat([data1, data2], keys=["Table_1", "Table_2"], join="inner", sort=True, verify_integrity=True)
# print(con1)

# ------------------------------------------------------------------------------

# import pandas as pd

# data1 = pd.read_csv("data3.csv")
# data2 = pd.read_csv("data4.csv")
# data3 = pd.read_csv("data5.csv")

# con = data1.query("votes > 80000 or approx_cost >= 800")
# print(con)

# ------------------------------------------------------------------------------

# import pandas as pd

# data1 = pd.read_csv("data3.csv")
# data2 = pd.read_csv("data4.csv")
# data3 = pd.read_csv("data5.csv")

# con = data1.melt(id_vars="name", value_vars=["votes"], var_name="price", value_name="dollars")
# print(con)

# ------------------------------------------------------------------------------

# import pandas as pd

# data1 = pd.read_csv("data3.csv")
# data2 = pd.read_csv("data4.csv")
# data3 = pd.read_csv("data5.csv")

# mer = data1.merge(data2, on = "name", indicator=True, how = "outer", suffixes= ("_table1", "_table2"), sort=True)
# print(mer)

# mer_ord = pd.merge_ordered(data1, data2, on="name", fill_method="ffill")
# print(mer_ord)

# mer_asof = pd.merge_asof(data2, data3, on="votes",direction="forward")
# print(mer_asof)

# con = pd.concat([data1, data2], ignore_index=True, join="inner")
# print(con)

# que = data1.query("votes >= 80000")
# print(que)

# melt = data1.melt(id_vars="name", value_vars=["votes"], var_name="price", value_name="dollars", ignore_index=True)
# print(melt)

# ------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from scipy.stats import iqr, uniform, binom, norm, poisson, expon, t
# import matplotlib.pyplot as plt
# import seaborn as sns

# data1 = pd.read_csv("data3.csv")
# data2 = pd.read_csv("data4.csv")
# data3 = pd.read_csv("data5.csv")

# print(data1["votes"].quantile([0.25, 0.50, 0.75]))
# print(np.quantile(data1["votes"], np.linspace(0, 1, 3)))
# print(iqr(data1["votes"]))
# print(round(data1["votes"].describe(), 2))

# np.random.seed(7)
# print(data1.sample(3, replace=True))
# print(len(data1))

# print(uniform.cdf(7, 0, 12))
# print(uniform.rvs(0, 5, 10))

# print(binom.rvs(1, 0.5, 10)) # binom.rvs(# of coins, probability of heads/success, size=# of trials)
# print(binom.pmf(1, 10, 0.5)) # # binom.pmf(num heads, num trials, prob of heads)

# print(1 - norm.cdf(154, 161, 7))
# print(norm.ppf(0.8413447460685429, 161, 7))

# norma = norm.rvs(161, 7, size=10**5)
# plt.hist(norma)
# sns.kdeplot(norma)
# plt.show()

# dice = pd.Series([1, 2, 3, 4, 5, 6])
# samp_list = []
# std_list = []
# for i in range(1000):
#     samp = dice.sample(5, replace=True)
#     samp_list.append(samp.mean())
#     std_list.append(samp.std())

# plt.hist(samp_list)
# sns.kdeplot(samp_list)

# plt.hist(std_list)
# sns.kdeplot(std_list)
# plt.show()

# print(poisson.pmf(5, 8))
# print(poisson.cdf(5, 8))
# print(1 - poisson.cdf(5, 8))
# print(1 - poisson.cdf(5, 10))

# poisson_list = poisson.rvs(8, size=10000)
# print(poisson_list)
# plt.hist(poisson_list)
# sns.kdeplot(poisson_list)
# plt.show()

# print(expon.cdf(1, scale=2))
# print(1 - expon.cdf(4, scale=2))
# print(expon.cdf(4, scale=2) - expon.cdf(1, scale=2))
# expon_list = expon.rvs(scale=2, size=100000)
# plt.hist(expon_list)
# sns.kdeplot(expon_list)
# plt.show()

# x = np.linspace(-5, 5, 100)
# degrees_of_freedom = [1, 2, 5, 10]
# for df in degrees_of_freedom:
#     y = t.pdf(x, df)
#     plt.plot(x, y, label=f"Degrees of freedom: {df}")
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt

# x = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
# y = [10, 12, 20, 30, 34, 25, 15]
# fig, ax = plt.subplots()
# ax.bar(x, y)
# ax.plot(x, y, marker="o", linestyle="None")
# ax.plot(x, y, marker="o", linestyle="dashdot", color="blue")
# ax.set_xlabel("Time (Days)")
# ax.set_ylabel("Temperature")
# ax.set_title("Temperature during the week")
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt

# x = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
# y = [10, 12, 20, 30, 34, 25, 15]
# fig, ax = plt.subplots(2, 1, sharey=True)
# ax[0].bar(x, y, color="b")
# ax[1].plot(x, y, marker="o", color="r")
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("data1.csv")
# data["date"] = pd.to_datetime(data["date"])
# data.set_index("date", inplace=True)

# fig, ax = plt.subplots()
# ax.plot(data.index, data["co2"], color="b")
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2 (ppm)", color="b")
# ax.tick_params("y", colors="b")

# ax2 = ax.twinx()
# ax2.plot(data.index, data["relative_temp"], color="r")
# ax2.set_ylabel("Relative Tempretures", color="r")
# ax2.tick_params("y", colors="r")

# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd

# def plot_timeseries(axes, x, y, color, xlabel, ylabel):
#     axes.plot(x, y, color=color)
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.tick_params("y", colors=color)


# data = pd.read_csv("data1.csv")
# data["date"] = pd.to_datetime(data["date"])
# data.set_index("date", inplace=True)

# fig, ax = plt.subplots()
# plot_timeseries(ax, data.index, data["co2"], "blue", "Time", "CO2 (ppm)")

# ax2 = ax.twinx()
# plot_timeseries(
#     ax2, data.index, data["relative_temp"], "red", "Time", "Relative Tempretures"
# )
# ax2.annotate(
#     "==1degree",
#     xy=(pd.Timestamp("1958-06-09"), 0.980),
#     xytext=(pd.Timestamp("1958-06-11"), 0.1),
#     arrowprops={"arrowstyle": "->", "color": "black"},
# )

# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("olympic_medals.csv")
# data.set_index("Country", inplace=True)

# fig, ax = plt.subplots()
# ax.bar(data.index, data["Gold"], label="Gold")
# ax.bar(data.index, data["Silver"], bottom=data["Gold"], label="Silver")
# ax.bar(data.index, data["Bronze"], bottom=data["Gold"] + data["Silver"], label="Bronze")
# ax.set_xticklabels(data.index, rotation=90)
# ax.set_ylabel("Number of Medals")
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("olympic_medals.csv")
# data.set_index("Country", inplace=True)

# fig, ax = plt.subplots()
# ax.hist(data["Gold"], bins=5, histtype="step", label="Gold")
# ax.hist(data["Silver"], bins=5, histtype="step", label="Silver")
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("olympic_medals.csv")
# data.set_index("Country", inplace=True)

# fig, ax = plt.subplots()
# ax.bar("Gold", data["Gold"].mean(), yerr=data["Gold"].std())
# ax.bar("Silver", data["Silver"].mean(), yerr=data["Silver"].std())
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd

# data = pd.read_csv("olympic_medals.csv")
# data.set_index("Country", inplace=True)

# fig, ax = plt.subplots()
# ax.errorbar(data["Gold"], data["Silver"], yerr=data["Silver"].std())
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# data = pd.read_csv("olympic_medals.csv")
# data.set_index("Country", inplace=True)

# plt.style.use("ggplot")
# fig, ax = plt.subplots()
# fig.set_size_inches([5, 3])
# ax.hist(data["Gold"])
# ax.hist(data["Silver"])
# ax.hist(data["Bronze"])
# fig.savefig("Figure3.png", dpi=300)
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import seaborn as sns

# height = [62, 64, 69, 75, 66, 68, 65, 71, 76, 73]
# weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187]

# sns.scatterplot(x=height, y=weight)
# sns.lineplot(x=height, y=weight)
# plt.show()

# ------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import seaborn as sns

# gender = [
#     "Female",
#     "Female",
#     "Female",
#     "Female",
#     "Male",
#     "Male",
#     "Male",
#     "Male",
#     "Male",
#     "Male",
# ]
# sns.countplot(x=gender)
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# df = pd.read_csv("trial.csv")
# sns.countplot(x="how_masculine",data=df)
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# hue_colors = {"usa" : "black", "japan" : "red", "europe" : "blue"}
# df = pd.read_csv("trial.csv")
# sns.scatterplot(data=df, x="mpg", y="horsepower", hue="origin", palette=hue_colors)
# sns.countplot(data=df, x="cylinders", hue="origin")
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# hue_colors = {"Yes": "black", "No": "red"}
# df = pd.read_csv("trial.csv")
# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     kind="scatter",
#     col="smoker",
#     hue="smoker",
#     col_wrap= 3,
#     palette=hue_colors,
# )
# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     kind="line",
#     col="smoker",
#     hue="smoker",
#     palette=hue_colors,
# )

# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     kind="scatter",
#     row="smoker",
#     hue="smoker",
#     palette=hue_colors,
# )
# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     kind="line",
#     row="smoker",
#     hue="smoker",
#     palette=hue_colors,
# )
# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     col="smoker",
#     row="time",
#     hue="smoker",
#     palette=hue_colors,
#     col_order=["Yes", "No"],
# )
# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     size="size",
#     hue="size"
# )
# sns.relplot(
#     data=df,
#     x="total_bill",
#     y="tip",
#     hue="smoker",
#     style="smoker",
#     alpha=0.5
# )

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("trial.csv")

# sns.relplot(
#     data=df,
#     x="mpg",
#     y="horsepower",
#     kind="line",
#     hue="origin",
#     style="origin",
#     markers=True,
#     dashes=False,
#     errorbar=None,
# )
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("trial.csv")
# sns.catplot(data=df, x="cylinders", hue="origin", kind="count", order=[6, 8, 3, 4])
# sns.catplot(data=df, x="cylinders", y="horsepower", hue="origin", kind="bar", ci=None)
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("trial.csv")
# sns.catplot(data=df, x="cylinders", y="mpg", kind="box", showfliers=False, whis=[0, 100])
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("trial.csv")

# sns.catplot(
#     data=df,
#     x="cylinders",
#     y="mpg",
#     kind="point",
#     hue="origin",
#     linestyle="none",
#     estimator="median",
#     capsize=0.2,
#     errorbar=None,
# )
# sns.catplot(data=df, x="cylinders", y="mpg", kind="bar", hue="origin")
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("trial.csv")

# sns.set_theme(style="dark", palette="Blues", context="notebook", font="Arial", font_scale=1.3)
# sns.set_palette("seismic") # ( RdBu, PuOr, BrBG, coolwarm, seismic, Spectral, PiYG, RdYlBu )
# sns.set_style("ticks")
# sns.set_context("notebook")
# sns
# sns.catplot(
#     data=df,
#     x="cylinders",
#     y="mpg",
#     kind="bar",
#     hue="origin",
#     linestyle="none",
#     errorbar=None,
# )
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("trial.csv")

# g = sns.catplot(data=df, x="mpg", y="horsepower", kind="bar", errorbar=None, col="origin", hue="origin")
# g.set_titles("This is {col_name}")
# g.set(xlabel="New Xlabe", ylabel="New Ylabel")

# g.set_xticklabels(rotation=90)

# g.figure.suptitle("New Title", y=0.8)
# plt.suptitle("New Title")

# plt.show()

# ------------------------------------------------------------------------------

# lst = [42, 30, 29, 12, 23]
# lam = map(lambda num : num ** 2, lst)
# print(list(lam))

# def sqrt(x):
#     if x < 0:
#         raise ValueError('x must be non-negative')
#     try:
#         return x ** 0.5
#     except TypeError:
#         print('x must be an int or float')

# ------------------------------------------------------------------------------

# word = "DataCamp"
# it = iter(word)
# print(next(it))
# print(next(it))
# print(next(it))
# print(*it)

# num = range(3)
# it = iter(num)
# print(*it)

# num = range(3)
# it = iter(num)
# print(next(it))
# print(next(it))
# print(next(it))

# ------------------------------------------------------------------------------

# lst1 = ["A", "B", "C", "D"]
# lst2 = [1, 2, 3, 4]

# enu = enumerate(lst1)
# print(*enu)
# for index, value in enumerate(lst1, start=10):
#     print(f"{index} : {value}")

# zp = zip(lst1, lst2)
# print(*zp)
# for index, value in zip(lst1, lst2):
#     print(f"{index} : {value}")

# ------------------------------------------------------------------------------

# import pandas as pd

# num = 0
# df = pd.read_csv("data.csv", chunksize=5)
# print(df.read())

# for i in df:
#     print(i)

# ------------------------------------------------------------------------------

# num = [10, 25, 14, 23]
# print([i for i in num if i % 2 == 0])
# print([i if i % 2 == 0 else i ** 2 for i in num])

# print((2 * num for num in range(10)))

# def seq(n):
#     i = 0
#     while i < n:
#         yield i
#         i += 1

# gen = list(seq(5))
# print(gen)

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")

# print(df["study_hours_per_day"].head())
# print(df.dtypes)
# df["study_hours_per_day"] = df["study_hours_per_day"].astype(int) # to change the column type
# print(df["study_hours_per_day"].head())

# print(df.select_dtypes("number").head()) # to extract a specific data type
# print(df.select_dtypes("object").head())

# print(df.groupby("gender").agg({"social_media_hours" : ["mean", "std"], "study_hours_per_day" : "median"}))
# print(df.groupby("gender").agg(
#     mean_rating = ("social_media_hours", "mean"),
#     median_rating = ("study_hours_per_day", "median")
# ))
# print(df.agg({"social_media_hours" : ["mean", "std"], "study_hours_per_day" : "median"}))

# print(df.columns[df.isna().sum() > 50])

# print(len(df) * 0.05)

# print(df.nunique()) # Count the number of categorical data in every column

# print(df.shape)
# oprint(len(df) * 0.05)

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("covid.csv")

# cols = df.columns[df.isna().sum() > 100]
# df.drop(columns=cols, inplace=True, axis=1)

# numeric_cols = df.select_dtypes("number").columns
# for i in numeric_cols:
#     df.fillna({i: df[i].mean()}, inplace=True) # this is the 1st method
#     df[i] = df[i].fillna(df[i].mean()) # this is the 2nd method


# object_cols = df.select_dtypes("object").columns
# for i in object_cols:
#     df[i] = df[i].fillna(df[i].mode)

# print(df.isna().sum())

# ------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np

# df = pd.read_csv("covid.csv")

# print(df.select_dtypes("object").head())
# print(df.select_dtypes("number").head())

# print(df["WHO Region"].value_counts())

# print(df["WHO Region"].nunique())
# print(df["WHO Region"].unique())

# x1 = df["Continent"].str.contains("South America", na=False)
# x2 = df["Continent"].str.contains("Europe|Asia", na=False)
# x3 = df["Continent"].str.contains("^N", na=False)

# conditions = [
#     (df["Continent"].str.contains("South America", na=False)),
#     (df["Continent"].str.contains("Europe|Asia", na=False)),
#     (df["Continent"].str.contains("^N", na=False))
# ]

# df["Country/Region"] = np.select(conditions, ["South", "Euro", "Asi"], default="Other")
# print(df["Country/Region"].head())

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")
# df["diet_quality"] = df["diet_quality"].str.replace("Fair", "Unfair")

# df["attendance_group"] = df.groupby("age")["attendance_percentage"].transform(lambda x : x.std())
# print(df.head())

# ------------------------------------------------------------------------------

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# df = pd.read_csv("student_habits_performance.csv")

# sns.boxplot(data=df, y="netflix_hours")
# plt.show()

# ( To remove outliers )
# per_25 = df["netflix_hours"].quantile(0.25)
# per_75 = df["netflix_hours"].quantile(0.75)
# iqr = per_75 - per_25
# lower = per_25 - 1.5 * iqr
# upper = per_75 + 1.5 * iqr
# outliers = df[(df["netflix_hours"] < lower) | (df["netflix_hours"] > upper)]["netflix_hours"].index
# df.drop(outliers, inplace=True)

# sns.boxplot(data=df, y="netflix_hours")
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd

# # ( This a the 1st way to convert a column to datatime )
# df = pd.read_csv("divorce.csv", parse_dates=["divorce_date", "dob_man", "marriage_date"])

# ( This a the 2nd way to convert a column to datatime )
# df = pd.read_csv("divorce.csv")

# df["divorce_date"] = pd.to_datetime(df["divorce_date"])
# df["dob_man"] = pd.to_datetime(df["dob_man"])
# df["marriage_date"] = pd.to_datetime(df["marriage_date"])

# df["marriage_day"] = df["marriage_date"].dt.day
# df["marriage_month"] = df["marriage_date"].dt.month
# df["marriage_year"] = df["marriage_date"].dt.year

# print(df.head())

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("student_habits_performance.csv")

# df.dropna(inplace=True)

# integer = df.select_dtypes("number")
# sns.heatmap(integer.corr(), annot=True)
# plt.show()

# print(df["study_hours_per_day"].corr(df["exam_score"]))

# sns.lineplot(data=df, x="exam_score" , y="study_hours_per_day", hue="gender", ci=False)
# sns.scatterplot(data=df, x="exam_score" , y="study_hours_per_day", hue="gender")
# plt.show()

# sns.pairplot(data=integer, vars=["age", "study_hours_per_day", "social_media_hours", "exam_score"])
# plt.show()

# sns.kdeplot(df, x="exam_score", hue="gender")
# sns.kdeplot(df, x="exam_score", hue="gender", cut=0)
# sns.kdeplot(df, x="exam_score", hue="gender", cumulative=True)
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# pd.Series()
# df = pd.read_csv("student_habits_performance.csv")

# df.dropna(inplace=True)

# print(df["age"].value_counts(normalize=True))
# print(df["exam_score"].describe())


# # ( This is a method we use to cut the columns into classes ot lables )
# twenty_fifth = df["exam_score"].quantile(0.25)
# seventy_fifth = df["exam_score"].quantile(0.75)
# maximum = df["exam_score"].max()

# labels = ["Good", "Very Good", "Excellent"]
# bins = [
#     0,
#     twenty_fifth,
#     seventy_fifth,
#     maximum
# ]
# df["exam_category"] = pd.cut(df["exam_score"], labels=labels, bins=bins)
# print(df["exam_category"].nunique())
# sns.countplot(data=df, x="exam_category", hue="gender")
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("divorce.csv")

# df.dropna(inplace=True)

# print(pd.crosstab(df["education_man"], df["education_woman"]))
# print(pd.crosstab(df["education_man"], df["education_woman"], values=df["income_man"], aggfunc="mean"))

# print(df.pivot_table(index="education_man", columns="education_woman", values="income_man"))

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")

# df.dropna(inplace=True)

# print(df["gender"].value_counts(normalize=True))

# obj = df.select_dtypes("object").columns
# print(df[obj].nunique())

# ( Convert data type into categorical type and this will be useful to decrease the capacity of the data if it is large )
# print(df["gender"].nbytes)
# print(df["parental_education_level"].nbytes)

# df["parental_education_level"] = df["parental_education_level"].astype("category")
# print(df["parental_education_level"].dtype)

# df["gender"] = pd.Categorical(df["gender"], categories=["Male", "Female", "Other"], ordered=True) # use this if thd data are ordinal

# print(df["gender"].nbytes)
# print(df["parental_education_level"].nbytes)

# ------------------------------------------------------------------------------

# import pandas as pd

# columns_types = {"gender" : "category"}
# df = pd.read_csv("student_habits_performance.csv", dtype=columns_types)

# df.dropna(inplace=True)

# print(df["gender"].dtype)

# ------------------------------------------------------------------------------

# import pandas as pd

# my_data = ["A", "A", "C", "B", "C", "A"]

# series = pd.Series(my_data, dtype="category")
# print(series)

# series_ordered = pd.Categorical(my_data, categories=["C", "B", "A"], ordered=True)
# print(series_ordered)

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("student_habits_performance.csv")

# df.dropna(inplace=True)

# intger_cols = df.select_dtypes("number").columns
# print(df.groupby("gender").mean())
# print(df.groupby("gender")[intger_cols].agg(["mean", "size"]))
# print(df.groupby(["gender", "parental_education_level"])[intger_cols].agg("mean"))

# print(df.groupby(["gender", "parental_education_level"])[intger_cols].agg("size"))

# sns.countplot(data=df, x="gender", hue="parental_education_level")
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")
# df.dropna(inplace=True)

# df["internet_quality"] = df["internet_quality"].astype("category")

# df["internet_quality"] = df["internet_quality"].cat.set_categories(new_categories=["Average", "Poor", "Good"], ordered=True) # this like 5966
# df["internet_quality"] = pd.Categorical(df["internet_quality"], categories=["Poor", "Average", "Good"], ordered=True) # this like 5965

# repce = {
#     "Good" : "Excellent",
#     "Poot" : "Weak",
#     "Average" : "Good"
# }
# df["internet_quality"] = df["internet_quality"].cat.rename_categories(new_categories=["Weak", "Good", "Excellent"])
# df["internet_quality"] = df["internet_quality"].cat.rename_categories(new_categories=repce)
# df["internet_quality"] = df["internet_quality"].cat.rename_categories(lambda x : x.upper())
# df["internet_quality"] = df["internet_quality"].replace(repce)


# df["internet_quality"] = df["internet_quality"].cat.remove_categories(removals="Weak")
# df["internet_quality"] = df["internet_quality"].cat.reorder_categories(new_categories=["Average", "Good", "Poor"], ordered=False)

# print(df["internet_quality"].value_counts())
# print(df["internet_quality"].head())

# print(df.isna().sum())

# print(df["internet_quality"].cat.categories)
# print(df["internet_quality"].value_counts())

# ------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("Students Social Media Addiction.csv")

# sns.catplot(data=df, x="Gender", y="Avg_Daily_Usage_Hours", hue="Most_Used_Platform", kind="bar")
# sns.catplot(data=df, x="Gender", y="Avg_Daily_Usage_Hours", kind="point", dodge=True, hue="Most_Used_Platform", join=False)
# sns.catplot(data=df, x="Gender", y="Avg_Daily_Usage_Hours", kind="point", join=False)

# ax = sns.catplot(data=df, x="Gender", kind="count", col="Most_Used_Platform", col_wrap=4, palette=sns.color_palette("Set1"))
# ax.figure.suptitle("Number of hours for setting by Gender on Most_Used_Platform")
# ax.set_axis_labels(x_var="Gender", y_var="Hours")
# plt.subplots_adjust(top=0.9)
# plt.show()

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")
# df.dropna(inplace=True)

# df["internet_quality"] = df["internet_quality"].astype("category")

# df["internet_quality"] = df["internet_quality"].cat.set_categories(new_categories=["Poor", "Average", "Good"], ordered=True)
# df["internet_quality"] = df["internet_quality"].cat.codes

# print(df["internet_quality"].head())

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")
# df.dropna(inplace=True)

# df["internet_quality"] = df["internet_quality"].astype("category")

# df["internet_quality"] = df["internet_quality"].cat.set_categories(new_categories=["Poor", "Average", "Good"], ordered=True)

# name = df["internet_quality"]
# code = df["internet_quality"].cat.codes

# # ( This to encode )
# df["name_map"] = list(zip(name, code))
# print(df["name_map"].head())

# # ( This to decode )
# name_map = dict(zip(name, code))
# print(df["internet_quality"].map(name_map))

# ------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np

# df = pd.read_csv("student_habits_performance.csv")
# df.dropna(inplace=True)

# df["internet_quality"] = np.where(
#     df["parental_education_level"].str.contains("Master", regex=False), 1, 0
# )
# print(df["internet_quality"].head())

# ------------------------------------------------------------------------------

# import pandas as pd

# df = pd.read_csv("student_habits_performance.csv")
# df.dropna(inplace=True)

# onehot = pd.get_dummies(data=df, columns=["gender"], prefix="Gend", dtype=int)
# print(onehot.head())
# print(onehot.shape)

# ------------------------------------------------------------------------------

# import pandas as pd
# from scipy.stats import uniform
# from scipy.stats import binom
# from scipy.stats import iqr
# import numpy as np

# print(uniform.rvs(0, 5, size=10))
# print(uniform.cdf(9, 4, 10))

# print(binom.rvs(5, 0.5, size=10))
# print(binom.pmf(7, 10, 0.5))
# print(binom.cdf(7, 10, 0.5))

# df = pd.read_csv("student_habits_performance.csv")
# print(df["age"].quantile(np.linspace(0, 1, 5)))
# iqr = iqr(df["age"])
# lower = df["age"].quantile(0.25) - 1.5 * iqr
# upper = df["age"].quantile(0.75) + 1.5 * iqr
# print(lower)
# print(upper)

# ------------------------------------------------------------------------------

# ( Introduction to Importing Data in Python )

import mysql.connector
import numpy as np
import pandas as pd

# data = np.loadtxt("file.txt", skiprows=1, usecols=[1, 4])
# data = np.loadtxt("file.txt", dtype=str)
# print(data)

# ( Pickle file )
# import pickle
# df = pd.read_csv("data1.csv")
# df_array = df.to_pickle("main.pkl")
# print(df_array)

# with open("main.pkl", "rb") as file:
#     df = pickle.load(file)
# print(df)

# ( Excel File )
# excel = pd.read_excel("Book 1.xlsx")
# excel = pd.ExcelFile("Book 1.xlsx")
# print(excel.sheet_names)

# ( SAS File )
# from sas7bdat import SAS7BDAT
# with SAS7BDAT("main.sas7bdat") as file:
#     df_sas = file.to_data_frame()
# print(df_sas)

# ( HDF5 file )
# import h5py
# with h5py.File("main.hdf5", "r") as file:
#     data = file["my_dataset"][:]
# print(data)

# ( Matlap file )
# import scipy.io
# filename = "main.mat"
# mat = scipy.io.loadmat(filename)
# print(mat)

# ( Creating a database engine )
# from sqlalchemy import create_engine, inspect
# engine = create_engine('sqlite:///example.db')
# inspector = inspect(engine)
# table_names = inspector.get_table_names()
# print(table_names)

# ( MYSQL )
# import mysql
# data = mysql.connector.connect(
#     host = "127.0.0.1",
#     user = "root",
#     password = "Abdallah149.",
#     database = "ai"
# )
# cursor = data.cursor()
# cursor.execute("SHOW DATABASES;")
# databases = cursor.fetchall()
# print(databases)
# for db in databases:
#     print(db[0])

# cursor = data.cursor()
# cursor.execute("SHOW TABLES;")
# table = cursor.fetchall()
# for db in table:
    # print(db[0])

# cursor.execute("SELECT * FROM newjoinee;")
# employee = pd.DataFrame(cursor.fetchall())
# print(employee)

# ( Creating a database engine )
# from sqlalchemy import create_engine
# engine = create_engine("mysql+mysqlconnector://root:Abdallah149.@127.0.0.1/ai")
# employee = pd.read_sql("SELECT * FROM employee;", engine)
# employee = pd.read_sql_query("SELECT * FROM employee;", engine)
# print(employee)

# ( Creating a database engine )
# from sqlalchemy import create_engine, text
# engine = create_engine("mysql+mysqlconnector://root:Abdallah149.@127.0.0.1/ai")
# con = engine.connect()
# rs = con.execute(text("SELECT * FROM employee"))
# df = pd.DataFrame(rs.fetchall(), columns=rs.keys())
# print(df)