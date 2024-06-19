import numpy as np
import pandas as pd

frame = pd.read_csv('data.csv')
array = frame.values
print(array.shape)
print(array)


