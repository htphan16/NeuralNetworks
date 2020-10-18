# CS 365 Lab C: NeuralNetworks
# Huong Phan

Run model_file with different hyperparameters: number of hidden nodes, number of epochs, learning rate
python main.py model_file <file_name> <num_hiddens>  <epochs>  <learning_rate>
e.g. python main.py model_file xor.txt 2 1000 0.3
python main.py model_file xor.txt 2 50000 0.1
python main.py model_file xor.txt 3 100000 0.1

Returns Final loss and Final weights and biases.

Note: If there is overflow error, simply run the command again.

