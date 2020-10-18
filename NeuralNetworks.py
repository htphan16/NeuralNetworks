from math import exp
import pandas as pd
import numpy as np

# Sigmoid function	
def sigmoid(x):
	return 1/(1 + exp(-x))

# Weights and Biases
def init_weights_biases(num_input_nodes, num_hidden_nodes, num_output_nodes):
	parameter_dictionary = {'hidden_biases': np.zeros((num_hidden_nodes, 1)), 
							'output_biases': np.zeros((num_output_nodes, 1)),
							'hidden_weights': np.random.randn(num_hidden_nodes, num_input_nodes),
							'output_weights': np.random.randn(num_output_nodes, num_hidden_nodes)
						   }
	return parameter_dictionary

# Reading our data
def read_file_to_array(file_name):
	file = open(file_name, 'r').readlines()
	infile = [line.strip('\n').split() for line in file]
	first_line = infile[0]
	values = np.array(infile[1:]).T
	feature_array = np.array(values[0:-1]).astype('float64') 
	label_array = np.array([values[-1]]).astype('float64') 
	header_array = np.array([[text] for text in first_line])
	return feature_array, label_array, header_array

# Forward Propagation
def forward_propagate(feature_array, parameter_dictionary, verbose=False):
	if verbose:
		print('Feature Array', feature_array)
		print('Hidden Weights', parameter_dictionary['hidden_weights'])
		print('Hidden Weights by Feature Array', np.dot(parameter_dictionary['hidden_weights'], feature_array))
	hidden_layer_values = np.dot(parameter_dictionary['hidden_weights'], feature_array) + parameter_dictionary['hidden_biases']
	if verbose:
		print('Hidden Layer Values', hidden_layer_values)
	hidden_layer_outputs = np.vectorize(sigmoid)(hidden_layer_values)
	output_layer_values = np.dot(parameter_dictionary['output_weights'], hidden_layer_outputs) + parameter_dictionary['output_biases']
	output_layer_outputs = np.vectorize(sigmoid)(output_layer_values)
	output_vals = {'hidden_layer_outputs': hidden_layer_outputs, 'output_layer_outputs': output_layer_outputs}
	if verbose:
		print('Hidden Layer Outputs', hidden_layer_outputs)
		print('Output Layer Outputs', output_layer_outputs)
	return output_vals

# Calculating Our Loss
def find_loss(output_layer_outputs, labels):
	num_examples = labels.shape[1]
	loss = (-1/num_examples) * np.sum(np.multiply(labels, np.log(output_layer_outputs)) + np.multiply(1-labels, np.log(1-output_layer_outputs)))
	return loss

# Backpropagation
def backprop(feature_array, labels, output_vals, weights_biases_dict, verbose=False):
	if verbose:
		print()
	num_examples = labels.shape[1]
	hidden_layer_outputs = output_vals['hidden_layer_outputs']
	output_layer_outputs = output_vals['output_layer_outputs']
	output_weights = weights_biases_dict['output_weights']
	raw_error = output_layer_outputs - labels
	if verbose:
		print('raw error', raw_error)
	output_weights_gradient = np.dot(raw_error, hidden_layer_outputs.T)/num_examples
	if verbose:
		print('output_weights_gradient', output_weights_gradient)
	output_bias_gradient = np.sum(raw_error, axis=1, keepdims=True)/num_examples
	if verbose:
		print('output_bias_gradient', output_bias_gradient)
	blame_array = np.dot(output_weights.T, raw_error)
	if verbose:
		print('blame_array', blame_array)
	hidden_outputs_squared = np.power(hidden_layer_outputs, 2)
	if verbose:
		print('hidden_layer_outputs', hidden_layer_outputs)
		print('hidden_outputs_squared', hidden_outputs_squared)
	propagated_error = np.multiply(blame_array, 1-hidden_outputs_squared)
	if verbose:
		print('propagated_error', propagated_error)
	hidden_weights_gradient = np.dot(propagated_error, feature_array.T)/num_examples
	hidden_bias_gradient = np.sum(propagated_error, axis=1, keepdims=True)/num_examples
	if verbose:
		print('hidden_weights_gradient', hidden_weights_gradient)
		print('hidden_bias_gradient', hidden_bias_gradient)
	gradients = {'hidden_weights_gradient': hidden_weights_gradient,
	             'hidden_bias_gradient': hidden_bias_gradient,
	             'output_weights_gradient': output_weights_gradient,
	             'output_bias_gradient': output_bias_gradient
	            }
	return gradients

# Updating our weights
def update_weights_biases(parameter_dictionary, gradients, learning_rate):
	new_hidden_weights = parameter_dictionary['hidden_weights'] - learning_rate*gradients['hidden_weights_gradient']
	new_hidden_biases = parameter_dictionary['hidden_biases'] - learning_rate*gradients['hidden_bias_gradient']
	new_output_weights = parameter_dictionary['output_weights'] - learning_rate*gradients['output_weights_gradient']
	new_output_biases = parameter_dictionary['output_biases'] - learning_rate*gradients['output_bias_gradient']
	updated_parameters = {'hidden_weights': new_hidden_weights,
	                      'hidden_biases': new_hidden_biases,
	                      'output_weights': new_output_weights,
	                      'output_biases': new_output_biases
						 }
	return updated_parameters

def model_file(file_name, num_inputs, num_hiddens, num_outputs, epochs, learning_rate):
	features, labels, headers = read_file_to_array(file_name)
	parameter_dictionary = init_weights_biases(num_inputs, num_hiddens, num_outputs)
	for i in range(epochs):
		output_vals = forward_propagate(features, parameter_dictionary, verbose=False)
		loss = find_loss(output_vals['output_layer_outputs'], labels)
		if i % 1000 == 0:
			print('Loss every 1000 epochs:', loss)
		gradients = backprop(features, labels, output_vals, parameter_dictionary, verbose=False)
		parameter_dictionary = update_weights_biases(parameter_dictionary, gradients, learning_rate)
	print('Final loss:', loss)
	print('Final weights and biases:')
	return parameter_dictionary

