import time
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_loss(out, data, criterion):
    """
    Function to calculate loss.

    Inputs:
        out (PyTorch tensor): Output of the model.
        data (PyTorch Batch object): Batch of graphs from which one can extract labels and sample weights
        criterion (PyTorch loss function): Loss function which takes model output and labels as inputs. Must have reduction='none'.
    Returns:
        loss (PyTorch Tensor): The calculated loss on which backpropagation can be performed.
    """
    loss = criterion(out, data.y.view(-1, 1))
    loss = torch.div(torch.dot(loss.flatten(), torch.abs(data.w)), torch.mean(torch.abs(data.w))) / 2e5
    return loss


def train(model, loader, cuda, criterion, optimizer):
    """
    Function to run training epoch.

    Inputs:
        model (PyTorch model object)
        loader (PyTorch loader)
        cuda (PyTorch device)
        criterion (PyTorch loss function)
        optimizer (PyTorch optimizer)
    """
    model.train()
    running_loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
        # zero the parameter gradients
        data.to(cuda)
        optimizer.zero_grad()
        # pseudo_mH=data.u*data.pseudo_mH+np.random.randint(4,10)*0.10*(data.u-1)**2 # randomise background pseudomass
        out = model(
            data.x, data.edge_index, data.edge_attr, data.u, data.pseudo_mH, data.batch
        )  # Perform a single forward pass.
        loss = get_loss(out, data, criterion)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        running_loss += loss.item()
    return running_loss


def get_metrics(model, loader, cuda, criterion):
    """
    Function to get testing metrics.

    Inputs:
        model (PyTorch model object)
        loader (PyTorch loader)
        cuda (PyTorch device)
        criterion (PyTorch loss function)
    """
    model.eval()
    running_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(cuda)
        # zero the parameter gradients
        out = model(
            data.x, data.edge_index, data.edge_attr, data.u, data.pseudo_mH, data.batch
        )  # Perform a single forward pass.
        loss = get_loss(out, data, criterion)
        running_loss += loss.item()
    return running_loss


def runTraining(model, loader_train, loader_test, patience, max_epochs, modelname, cuda, criterion, optimizer):
    """
    Function to run training loop.

    Inputs:
        model (PyTorch model object): Input model.
        loader_train (PyTorch loader): Loader containing training dataset.
        loader_test (PyTorch loader): Loader containing testing dataset.
        patience (int): Number of epochs of no improvement to testing loss before training will stop early.
        max_epochs (int): Max number of epochs to train.
        modelname (string): Name of file to save model to.
        cuda (PyTorch device): GPU
        criterion (PyTorch loss function)
        optimizer (PyTorch optimizer)

    Returns:
        model (PyTorch model object): Trained model.
        history (dict): Metrics of training for final model and over training.
    """
    start = time.time()
    epochs = []
    train_losses = []
    test_losses = []
    train_aucs = []
    test_aucs = []
    min_loss = np.inf
    no_improvement_steps = 0
    for epoch in range(1, max_epochs):
        epoch_start = time.time()
        av_loss = train(model, loader_train, cuda, criterion, optimizer)
        epoch_end = time.time()
        test_loss = get_metrics(model, loader_test, cuda, criterion)
        metrics_end = time.time()
        elapsed = time.time() - start
        print(
            "Epoch: {}, \tTraining loss: {:.4f},\tTest loss: {:.4f}, \t\t\tTraining time: {:.2f}s, \tEvaluation time: {:.2f}s, \tTotal time elapsed: {:.2f}s".format(
                epoch, av_loss, test_loss, epoch_end - epoch_start, metrics_end - epoch_end, elapsed
            )
        )
        # Early-stopping / checkpointing
        if test_loss < min_loss:  # if loss is lower than previous best
            no_improvement_steps = 0  # reset patience counter
            print(
                'Test loss improved from {:.4f} to {:.4f}. Saving model to "{}"'.format(min_loss, test_loss, modelname)
            )
            min_loss = test_loss
            torch.save(model, modelname)  # save model
            choice_epoch = epoch  # save epoch
        else:
            print("Test loss did not improve.")
            no_improvement_steps += 1  # else add 1 to the patience counter and dont save
        if no_improvement_steps >= patience:
            print("No improvement for {} epochs. Early stopping now.".format(no_improvement_steps))
            break
        print("\n")
        epochs.append(epoch)
        train_losses.append(av_loss)
        test_losses.append(test_loss)

    model = torch.load(modelname)
    train_loss = get_metrics(model, loader_train, cuda, criterion)
    test_loss = get_metrics(model, loader_test, cuda, criterion)
    print("Reloaded best model .\tTrain loss {:.4f},\tTest Loss {:.4f}".format(train_loss, test_loss))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses, label="Test loss")
    plt.legend(loc="best")
    plt.show()

    history = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
    }

    return model, history
