# Test runner for neural network models
import optparse
import sys
import traceback
import contextlib
import numpy as np

from torch import nn, Tensor
import torch
import backend

TESTS = []

def test(q):
    def deco(fn):
        TESTS.append((q, fn))
        return fn
    return deco


################################################################################
# CLI options
################################################################################

def parse_options(argv):
    parser = optparse.OptionParser(description='Run tests on neural network models')
    parser.add_option('--question', '-q',
                      dest='grade_question',
                      default=None,
                      help='Run only one question (e.g. `-q q1`)')
    parser.add_option('--no-graphics',
                      dest='no_graphics',
                      action='store_true',
                      default=False,
                      help='Do not display graphics.')
    parser.add_option('--check-dependencies',
                      dest='check_dependencies',
                      action='store_true',
                      default=False,
                      help='Check that numpy and matplotlib are installed.')
    (options, args) = parser.parse_args(argv)
    return options

################################################################################
# Main
################################################################################

def main():
    options = parse_options(sys.argv)

    if options.check_dependencies:
        check_dependencies()
        return

    if options.no_graphics:
        disable_graphics()

    questions = list(sorted(set(q for q, fn in TESTS)))

    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        questions = [options.grade_question]

    passed = []
    failed = []

    for q in questions:
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for testq, fn in TESTS:
            if testq != q:
                continue
            print("*** Running {} ...".format(fn.__name__))
            try:
                fn()
                print("*** PASS: {}".format(fn.__name__))
                passed.append(fn.__name__)
            except KeyboardInterrupt:
                print("\n\nCaught KeyboardInterrupt: aborting")
                sys.exit(1)
            except Exception:
                print("*** FAIL: {}".format(fn.__name__))
                print(traceback.format_exc())
                failed.append(fn.__name__)

    print('\n==================')
    print('Results')
    print('==================')
    print('Passed: {}'.format(len(passed)))
    print('Failed: {}'.format(len(failed)))
    if failed:
        print('Failed tests: {}'.format(', '.join(failed)))


################################################################################
# Helpers
################################################################################

def check_dependencies():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)
    for t in range(400):
        angle = t * 0.05
        x = np.sin(angle)
        y = np.cos(angle)
        line.set_data([x, -x], [y, -y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)

def disable_graphics():
    backend.use_graphics = False

@contextlib.contextmanager
def no_graphics():
    old = backend.use_graphics
    backend.use_graphics = False
    yield
    backend.use_graphics = old

def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, \
            "{} should return an instance of nn.Parameter, not None".format(method_name)
        assert isinstance(node, nn.Parameter), \
            "{} should return an instance of nn.Parameter, instead got type {!r}".format(
                method_name, type(node).__name__)
    elif expected_type == 'loss':
        assert node is not None, \
            "{} should return a loss node, not None".format(method_name)
        assert isinstance(node, nn.modules.loss._Loss), \
            "{} should return a loss node, instead got type {!r}".format(
                method_name, type(node).__name__)
    elif expected_type == 'tensor':
        assert node is not None, \
            "{} should return a Tensor, not None".format(method_name)
        assert isinstance(node, Tensor), \
            "{} should return a Tensor, instead got type {!r}".format(
                method_name, type(node).__name__)
    else:
        assert False, "Unexpected expected_type: {}".format(expected_type)

    if expected_type != 'loss':
        assert all(
            expected == '?' or actual == expected
            for actual, expected in zip(node.detach().numpy().shape, expected_shape)
        ), "{} should return an object with shape {}, got {}".format(
            method_name, expected_shape, node.shape)


################################################################################
# Tests
################################################################################

@test('q1')
def check_perceptron():
    import models, train

    print("Sanity checking perceptron...")
    np_random = np.random.RandomState(0)

    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        p.get_weights()
        number_of_parameters = 0
        for param in p.parameters():
            number_of_parameters += 1
            verify_node(param, 'parameter', (1, dimensions), 'PerceptronModel.parameters()')
        assert number_of_parameters == 1, 'Perceptron Model should only have 1 parameter'

    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        point = np_random.uniform(-10, 10, (1, dimensions))
        score = p(Tensor(point))
        verify_node(score, 'tensor', (1,), "PerceptronModel.forward()")
        calculated_score = score.item()
        for param in p.parameters():
            expected_score = float(np.dot(point.flatten(), param.detach().numpy().flatten()))
        assert np.isclose(calculated_score, expected_score), \
            "PerceptronModel.forward() score {:.4f} does not match expected {:.4f}".format(
                calculated_score, expected_score)

    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        random_point = np_random.uniform(-10, 10, (1, dimensions))
        for point in (random_point, np.zeros_like(random_point)):
            prediction = p.get_prediction(Tensor(point))
            assert prediction in (1, -1), \
                "get_prediction() should return 1 or -1, not {}".format(prediction)
            expected = np.where(np.dot(point, p.get_weights().data.T) >= 0, 1, -1).item()
            assert prediction == expected, \
                "get_prediction() returned {}; expected {}".format(prediction, expected)

    print("Sanity checking perceptron weight updates...")
    for multiplier in (-5, -2, 2, 5):
        p = models.PerceptronModel(2)
        orig_weights = p.get_weights().data.reshape((1, 2)).detach().numpy().copy()
        if np.abs(orig_weights).sum() == 0.0:
            continue
        point = multiplier * orig_weights
        sanity_dataset = backend.Custom_Dataset(
            x=np.tile(point, (500, 1)),
            y=np.ones((500, 1)) * -1.0)
        train.train_perceptron(p, sanity_dataset)
        new_weights = p.get_weights().data.reshape((1, 2)).detach().numpy()
        expected_weights = orig_weights if multiplier < 0 else orig_weights - point
        assert np.all(new_weights == expected_weights), \
            "Weight update check failed. Got {}, expected {}".format(new_weights, expected_weights)

    print("Sanity checking complete. Now training perceptron...")
    model = models.PerceptronModel(3)
    dataset = backend.PerceptronDataset(model)
    train.train_perceptron(model, dataset)
    backend.maybe_sleep_and_close(1)

    assert dataset.epoch != 0, "Perceptron never iterated over training data"
    accuracy = np.mean(
        np.where(np.dot(dataset.x, model.get_weights().data.T) >= 0.0, 1.0, -1.0) == dataset.y)
    print("Training accuracy: {:.2%}".format(accuracy))
    assert accuracy >= 1.0, \
        "Perceptron accuracy {:.2%} did not reach 100%".format(accuracy)


@test('q2')
def check_regression():
    import train, losses, models

    model = models.RegressionModel()
    dataset = backend.RegressionDataset(model=model)

    for batch_size in (1, 2, 4):
        inp_x = torch.tensor(dataset.x[:batch_size], dtype=torch.float, requires_grad=True)
        inp_y = torch.tensor(dataset.y[:batch_size], dtype=torch.float, requires_grad=True)
        loss = losses.regression_loss(model(inp_x), inp_y)
        verify_node(loss, 'tensor', (1,), "regression_loss()")
        grad_y = torch.autograd.grad(loss, inp_x, allow_unused=True, retain_graph=True)
        grad_x = torch.autograd.grad(loss, inp_y, allow_unused=True, retain_graph=True)
        assert grad_x[0] is not None, \
            "regression_loss() does not depend on input x — check RegressionModel.forward()"
        assert grad_y[0] is not None, \
            "regression_loss() does not depend on labels y — check regression_loss()"

    train.train_regression(model, dataset)
    backend.maybe_sleep_and_close(1)

    data_x = torch.tensor(dataset.x, dtype=torch.float32)
    labels = torch.tensor(dataset.y, dtype=torch.float32)
    train_loss = losses.regression_loss(model(data_x), labels)
    verify_node(train_loss, 'tensor', (1,), "regression_loss()")
    train_loss = train_loss.item()

    train_predicted = model(data_x)
    verify_node(train_predicted, 'tensor', (dataset.x.shape[0], 1), "RegressionModel()")
    error = labels - train_predicted
    sanity_loss = torch.mean((error.detach()) ** 2)
    assert np.isclose(train_loss, sanity_loss), \
        "regression_loss() returned {:.4f}, but recomputed loss is {:.4f}".format(
            train_loss, sanity_loss)

    print("Final loss: {:f}".format(train_loss))
    assert train_loss <= 0.02, \
        "Final loss {:.4f} must be ≤ 0.02".format(train_loss)
    

@test('q3')
def check_digit_classification():
    import models, train, losses

    model = models.DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model)

    for batch_size in (1, 2, 4):
        inp_x = torch.tensor(dataset.x[:batch_size], dtype=torch.float, requires_grad=True)
        inp_y = torch.tensor(dataset.y[:batch_size], dtype=torch.float, requires_grad=True)
        loss = losses.digitclassifier_loss(model(inp_x), inp_y)
        verify_node(loss, 'tensor', (1,), "digitclassifier_loss()")
        grad_y = torch.autograd.grad(loss, inp_x, allow_unused=True, retain_graph=True)
        grad_x = torch.autograd.grad(loss, inp_y, allow_unused=True, retain_graph=True)
        assert grad_x[0] is not None, \
            "digitclassifier_loss() does not depend on input x"
        assert grad_y[0] is not None, \
            "digitclassifier_loss() does not depend on labels y"

    train.train_digitclassifier(model, dataset)

    test_logits = model(torch.tensor(dataset.test_images)).data
    test_predicted = np.argmax(test_logits, axis=1).detach().numpy()
    test_accuracy = np.mean(test_predicted == dataset.test_labels)
    print("Test accuracy: {:.2%}".format(test_accuracy))
    assert test_accuracy >= 0.95, \
        "Test accuracy {:.2%} must be ≥ 95%".format(test_accuracy)


@test('q4')
def check_lang_id():
    import models, train, losses

    model = models.LanguageIDModel()
    dataset = backend.LanguageIDDataset(model)

    for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
        start = dataset.dev_buckets[-1, 0]
        inp_xs, inp_y = dataset._encode(
            dataset.dev_x[start:start + batch_size],
            dataset.dev_y[start:start + batch_size])
        inp_xs = torch.tensor(inp_xs[:word_length], requires_grad=True)
        output_node = model(inp_xs)
        verify_node(output_node, 'tensor',
                    (batch_size, len(dataset.language_names)), "LanguageIDModel.forward()")
        grad = torch.autograd.grad(torch.sum(output_node), inp_xs,
                                   allow_unused=True, retain_graph=True)
        assert all(g is not None for g in grad), \
            "LanguageIDModel.forward() output does not depend on all inputs"

    for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
        start = dataset.dev_buckets[-1, 0]
        inp_xs, inp_y = dataset._encode(
            dataset.dev_x[start:start + batch_size],
            dataset.dev_y[start:start + batch_size])
        inp_xs = torch.tensor(inp_xs[:word_length], requires_grad=True)
        loss_node = losses.languageid_loss(model(inp_xs), inp_y)
        grad = torch.autograd.grad(loss_node, inp_xs, allow_unused=True, retain_graph=True)
        assert all(g is not None for g in grad), \
            "languageid_loss() does not depend on all inputs"

    train.train_languageid(model, dataset)

    test_accuracy = dataset.get_validation_accuracy()
    print("Test accuracy: {:.2%}".format(test_accuracy))
    assert test_accuracy >= 0.81, \
        "Test accuracy {:.2%} must be ≥ 81%".format(test_accuracy)


@test('q5')
def check_convolution():
    import models, train, losses

    model = models.DigitConvolutionalModel()
    dataset = backend.DigitClassificationDataset2(model)

    def conv2d(a, f):
        s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
        subM = np.lib.stride_tricks.as_strided(a, shape=s, strides=a.strides * 2)
        return np.einsum('ij,ijkl->kl', f, subM)

    for batch_size in (1, 2, 4):
        inp_x = torch.tensor(dataset[:batch_size]['x'], dtype=torch.float, requires_grad=True)
        inp_y = torch.tensor(dataset[:batch_size]['label'], dtype=torch.float, requires_grad=True)
        loss = losses.digitconvolution_Loss(model(inp_x), inp_y)
        verify_node(loss, 'tensor', (1,), "digitconvolution_Loss()")
        grad_y = torch.autograd.grad(loss, inp_x, allow_unused=True, retain_graph=True)
        grad_x = torch.autograd.grad(loss, inp_y, allow_unused=True, retain_graph=True)
        assert grad_x[0] is not None, "digitconvolution_Loss() does not depend on input x"
        assert grad_y[0] is not None, "digitconvolution_Loss() does not depend on labels y"

    for matrix_size in (2, 4, 6):
        weights = np.random.rand(2, 2)
        inp = np.random.rand(matrix_size, matrix_size)
        student_output = models.Convolve(torch.Tensor(inp), torch.Tensor(weights))
        actual_output = conv2d(inp, weights)
        assert np.isclose(student_output, actual_output).all(), \
            "Convolve() output does not match expected convolution"

    train.Train_DigitConvolution(model, dataset)

    test_logits = model(torch.tensor(dataset.test_images)).data
    test_predicted = np.argmax(test_logits, axis=1).detach().numpy()
    test_accuracy = np.mean(test_predicted == dataset.test_labels)
    print("Test accuracy: {:.2%}".format(test_accuracy))
    assert test_accuracy >= 0.80, \
        "Test accuracy {:.2%} must be ≥ 80%".format(test_accuracy)


@test('q6')
def check_attention():
    import models

    for block_size in [2, 4, 16]:
        layer_size = np.random.randint(2, 10)
        att_block = models.Attention(layer_size, block_size)
        batch_size = np.random.randint(1, 10)
        inp = torch.rand(batch_size, block_size, layer_size)

        k_weight = torch.rand(layer_size, layer_size)
        q_weight = torch.rand(layer_size, layer_size)
        v_weight = torch.rand(layer_size, layer_size)

        with torch.no_grad():
            att_block.k_layer.weight = nn.Parameter(torch.ones((layer_size, layer_size)) * k_weight)
            att_block.q_layer.weight = nn.Parameter(torch.ones((layer_size, layer_size)) * q_weight)
            att_block.v_layer.weight = nn.Parameter(torch.ones((layer_size, layer_size)) * v_weight)

        T = inp.shape[1]
        expected = torch.matmul(
            att_block.q_layer(inp),
            torch.movedim(att_block.k_layer(inp), 1, 2)
        ) / layer_size ** 0.5
        expected = expected.masked_fill(att_block.mask[:, :, :T, :T] == 0, float('-inf'))[0]
        expected = torch.matmul(nn.functional.softmax(expected, dim=-1), att_block.v_layer(inp))

        assert torch.all(torch.isclose(expected, att_block(inp))), \
            "Attention() output does not match expected (block_size={})".format(block_size)

    print("All attention configurations passed.")


if __name__ == '__main__':
    main()