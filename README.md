# NeuroCanvas: Neural Networks & NLP Models

A PyTorch-based machine learning project implementing several models of increasing complexity, from a simple perceptron to attention.

**Project Structure**

├── models.py        # Model definitions

├── train.py         # Training loops for each model

├── losses.py        # Loss functions

├── backend.py       # Dataset classes with live visualization

├── gpt_model.py     # Character-level GPT transformer

├── chargpt.py       # GPT training script

└── test.py          # Tests

**Models**
1. Perceptron (PerceptronModel): A binary linear classifier trained using the perceptron update rule (no gradient descent). Converges when all training points are correctly classified.

        Input: 3-dimensional feature vectors
        
        Output: Class prediction (+1 or -1)
        
        Training: Online updates until convergence, 100% training accuracy

2. Regression (RegressionModel): A feedforward neural network that approximates sin(x) over the interval [-2π, 2π].

        Architecture: Linear(1→128) → ReLU → Linear(128→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
        
        Loss: Mean Squared Error
        
        Convergence criterion: Average loss ≤ 0.02

3. Digit Classifier (DigitClassificationModel): A fully-connected network for handwritten digit classification on the MNIST dataset.

        Input: 784-dimensional flattened image (28×28 pixels)
        
        Architecture: Linear(784→128) → ReLU → Linear(128→128) → ReLU → Linear(128→10)
        
        Loss: Cross-Entropy
        
        Target accuracy: ≥ 95% validation accuracy

4. Convolutional Digit Classifier (DigitConvolutionalModel): An extension of the digit classifier that applies a learned 2D convolution before the fully-connected layers.

        Input: 28×28 grayscale images
        
        Architecture: Convolve(3×3) → flatten → Linear(676→100) → ReLU → Linear(100→100) → ReLU → Linear(100→10)
        
        Loss: Cross-Entropy
        
        Target accuracy: ≥ 81% validation accuracy

5. Language Identification (LanguageIDModel): A recurrent neural network that identifies the language of a word from a 5-class set: English, Spanish, Finnish, Dutch, and Polish.

        Input: Sequence of one-hot character vectors (47-character combined alphabet)
        
        Architecture: RNN with hidden layers of size 100
        
        Loss: Cross-Entropy
        
        Target accuracy: ≥ 90% validation accuracy

# To run:
        python/python3 test.py
