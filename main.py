from math import exp
import pandas as pd
import numpy as np
from sys import argv
from NeuralNetworks import *

def main(file_name, num_hiddens, epochs, learning_rate):
	try:
		if argv[1] == 'xor.txt':
			file_name = argv[1]
	except TypeError:
		print("Please enter the available file name: 'xor.txt'\n")
	print(model_file(file_name,2,int(num_hiddens),1,int(epochs),float(learning_rate)))

if __name__ == '__main__':
    main(argv[1], argv[2], argv[3], argv[4])
