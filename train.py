from torch import no_grad
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim

def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    Iterate through DataLoader in order to retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    converged = False
    while not converged:
        converged = True
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                for xi, yi in zip(x, y):
                    xi = xi.reshape(-1)
                    yi = yi.item()
                    pred = model.get_prediction(xi)
                    if pred != yi:
                        model.w += yi * xi
                        converged = False
                
def train_regression(model, dataset):
    """
    To create batches, create a DataLoader object and pass in `dataset` as well as the required 
    batch size.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    optimizer = optim.Adam(model.parameters(), lr= 0.002)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    converged = False
    while not converged:
        total_loss = 0
        for batch in dataloader:
            x_batch = batch['x']
            y_true = batch['label']
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = regression_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if avg_loss <= 0.02:
            converged = True
        

def train_digitclassifier(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(5):
        for batch in dataloader:
            x_batch = batch['x']
            y_true = batch['label']
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = digitclassifier_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
        if dataset.get_validation_accuracy() >= 0.98:
            break

def train_languageid(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr= 0.005)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(15):
        for batch in dataloader:
            xs = batch['x']
            ys = batch['label']
            xs = xs.movedim(0, 1)
            optimizer.zero_grad()
            y_pred = model(xs)
            loss = languageid_loss(y_pred, ys)
            loss.backward()
            optimizer.step()
        if dataset.get_validation_accuracy() >= 0.89:
            break


def Train_DigitConvolution(model, dataset):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(5):
        for batch in dataloader:
            x_batch = batch['x']
            y_true = batch['label']
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = digitconvolution_Loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
        if dataset.get_validation_accuracy() >= 0.85:
            break
