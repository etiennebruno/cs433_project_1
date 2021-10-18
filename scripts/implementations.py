#=============================================================================================================== HELPER FUNCTIONS

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    N = len(e)
    return e.T @ e / (2 * N)

def compute_loss_sol(y, tx, w):
    """Calculate the mse for vector e."""
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    N = len(e)
    return -1 / N * tx.T @ e

def compute_gradient_sol(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):#                   <------- CHECK IF WE CAN USE THIS FUNCTION!!!!
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



#=============================================================================================================== MAIN FUNCTIONS
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        #computing the gradient of the loss function
        grad = compute_gradient(y, tx, w)
        #iteratively updating the weights w w.r. to the gradient
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_GD_sol(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient_sol(y, tx, w)
        loss = compute_loss(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return ws[-1], losses[-1]

def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            # store w and loss
    loss = compute_loss(y, tx, w)
    return w, loss

def least_squares_SGD_sol(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
    
def least_squares(y, tx):
    """calculate the least squares solution."""
    
    # least squares
    # returns mse, and optimal weights 
    X = tx
    w = np.linalg.solve(X.T @ X, X.T @ y)
    loss = compute_loss(y, X, w)
    
    return w, loss
    
def least_squares_sol(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    # ridge regression
    X = tx
    N = X.shape[0]
    I = np.identity(X.shape[1])
    lambda_p = 2 * N * lambda_
     
    w = np.linalg.solve(X.T @ X + lambda_p * I, X.T @ y)   
    loss = compute_loss(y, X, w)
    
    return w, loss
    
def ridge_regression_sol(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
