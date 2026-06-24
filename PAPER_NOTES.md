Learning representations 
by back-propagating errors

Introduction :  This is the ML research paper by Rumelhart, Hinton & Williams that introduces the new learning procedure named back-propagation.Back-propagation is an algorithm that trains neural networks by reducing prediction error. It works by propagating errors backward, computing gradients using the chain rule, and updating weights and biases to improve performance. 

Central Claim : Back-propagation works better than all previous methods for training multi-layer neural networks. The biggest advantage of using back-propagation over others is that it helps models to generalise well, improving performance on unseen data.
Back-propagation is the only method that can actually train the hidden layers. It generalises the pattern, not memorises it. It updates weights and biases efficiently to reduce error.One more advantage instead of using Δw = -ε · ∂E/∂w this equation it uses Δw(t) = -ε · ∂E/∂w(t) + α · Δw(t-1) in which alpha terms remembers the previous direction which help in converging of gradient faster than plain gradient descent

Core architecture or algorithm: 
Algorithm : 
Step 1 : Forward pass Work.
Step 2 : Compute the error
Step 3: Backward pass
Step 4: Weight update



Working : 
Step 1 : Forward Pass Work : 
In forward pass, input data moves through the network to generate an output. 
Input data is passed to the input layer and forwarded to hidden layers
Each neuron computes a weighted sum and adds a bias
Activation functions (like ReLU) are applied to produce outputs
Outputs from one layer become inputs to the next layer
Final layer uses functions like softmax to produce predictions (e.g., probabilities for classification)


Step 2 : Compute the error:
In back-propagation error is calculated using the formula E = ½ × sum of (actual_output - desired_output)²

Step 3 : Backward pass:
In this step, the error between predicted and actual output is propagated backward to update weights and biases. 


For output layer:
    δ = (actual - desired) × y×(1-y)
For hidden layers:
    δ = (sum of δ_next × weights) × y×(1-y)


Step 4 : Weight Update: 
Weight is updated by using improved version, i.e., Δw(t) = -ε × δ × y_input  +  α × Δw(t-1)

Dataset, evaluation metric, and the baseline:
Dataset : 
1. Symmetry Detection:
Input:All 64 possible 6-bit binary vectors
        e.g. [1, 0, 1, 1, 0, 1]
Label:1 if symmetric, 0 if not
Size:64 total examples
2.Family Tree:
Data:Two isomorphic family trees (English + Italian)
24 people × relationships = 104 possible triples
Form: (person1, relationship) → person2
e.g. (Colin, father) → Arthur
Train:100 of 104 triples
Test: 4 held-out triples
Evaluation metric: 
They never report accuracy % or loss curves explicitly — they just show the learned weights and say, "It worked."
They measured the network as "correct" on output unit; error was considered zero and learning was successful. 
Baseline they compare against : 
Perceptron / single layer networks — which literally cannot solve tasks requiring hidden layers.





















