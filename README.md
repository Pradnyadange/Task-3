In this task, we implement a neural network to detect symmetry in 6-bit binary numbers.
For this we need pytorch.
backpropagation
we intially implment the feedforward funtion 
We calculate the error using mean squared error.
then implement the backward pass for backpropagation.
finally train the network and evaluate its accuracy.


## Dataset
We generate 64 samples of 6-bit binary numbers (0-63) and label them as symmetric (1) if they read the same forwards and backwards, otherwise 0.

## Training Process
We train the network using backpropagation with mean squared error loss and gradient descent optimization.
 ## How to run the code
Run the script using:
py main.py
## RESULT
The neural network achieves high accuracy in detecting symmetry in 6-bit binary numbers after training.
