from torch import stack, tril
from torch.nn import Module
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim

class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        """
        super(PerceptronModel, self).__init__()
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (dimensions)
        Returns: a node containing a single number (the score)
        """
        return tensordot(self.w, x.view(-1), dims=1)
        
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        score = self(x)
        return 1 if score.item() >= 0 else -1


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize model parameters
        super().__init__()
        self.layer1 = Linear(1, 128)
        self.layer2 = Linear(128, 128)
        self.layer3 = Linear(128, 64)
        self.output = Linear(64, 1)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        out = relu(self.layer1(x))
        out = relu(self.layer2(out))
        out = relu(self.layer3(out))
        out = self.output(out)
        return out


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """

    def __init__(self):
        # Initialize model parameters
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        hidden = [128, 128]
        self.layer1 = Linear(input_size, hidden[0])
        self.layer2 = Linear(hidden[0], hidden[1])
        self.output = Linear(hidden[1], output_size)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        out = relu(self.layer1(x))
        out = relu(self.layer2(out))
        out = self.output(out)
        return out

class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique characters.
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        # Initialize model parameters
        hidden = [100, 100]
        self.Wx = Linear(self.num_chars, hidden[0])
        self.Wh = Linear(hidden[0], hidden[1])
        self.output = Linear(hidden[1], len(self.languages))

    def forward(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` = a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet.

        Use a Recurrent Neural Network to summarize the list `xs` into a single tensor.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits) - higher scores correspond to
        greater probability of the word originating from a particular language.
        """
        h = relu(self.Wx(xs[0])) # first character
        for x in xs[1:]:
            h = relu(self.Wx(x) + self.Wh(h))
        out = self.output(h)
        return out

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    """

    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    outY = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    outX = input_tensor_dimensions[1] - weight_dimensions[1] + 1
    Output_Tensor = zeros((outY, outX))
    for y in range(outY):
        for x in range(outX):
            section = input[y:y+weight_dimensions[0], x:x+weight_dimensions[1]]
            Output_Tensor[y, x] = tensordot(section, weight)
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.
    """

    def __init__(self):
        # Initialize model parameters
        super().__init__()
        output_size = 10
        self.convolution_weights = Parameter(ones((3, 3)))
        hidden = [100,100]
        input_dim = 26 * 26
        self.layer1 = Linear(input_dim, hidden[0])
        self.layer2 = Linear(hidden[0], hidden[1])
        self.output = Linear(hidden[1], output_size)

    def forward(self, x):
        # Treat x as a regular 1-dimensional datapoint
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        out = relu(self.layer1(x))
        out = relu(self.layer2(out))
        return self.output(out)

class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        B, T, C = input.size()
        Q = self.q_layer(input)
        K_T = self.k_layer(input).movedim(1,2)
        V = self.v_layer(input)
        M = matmul(Q, K_T) / self.layer_size ** 0.5
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]
        output = softmax(M, dim=-1)
        return matmul(output, V)