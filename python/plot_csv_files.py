#!/usr/bin/env python3
# Use this script to plot a csv file and all its columns. 

import argparse
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

# Instantiate the parse
parser = argparse.ArgumentParser(description='Plot ALL the columns of a csv-file')

parser.add_argument('filename', type=str,
                    help='The name of the file to plot. Should be comma-delimited. First column should be index or time.')

args = parser.parse_args()

print(args.filename)

# 
data = pd.read_csv(args.filename, delimiter=',')

# assume first column is time
timestamp = data.iloc[:, 0]
print(timestamp)

cols = data.shape[1]

plt.figure()
for i in range(1, cols):
    plt.plot(timestamp.to_numpy(), data.iloc[:, i], label=f"{data.columns[i]}")

plt.legend()

plt.show()
