import numpy as np

#========================================================================================================================== HELPER FUNCTIONS

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    N = len(e)
    return e.T @ e / (2 * N)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    N = len(e)
    return -1 / N * tx.T @ e


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


def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))


def compute_loss_logistic(y, tx, w):
    """compute the loss: negative log likelihood."""
    ŷ = tx @ w
    loss = -1 * (y.T @ np.log(sigmoid(ŷ)) + (1 - y).T @ np.log(1 - sigmoid(ŷ)))
    return loss


def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)


def logistic_regression_one_iter(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_logistic(y, tx, w)
    grad = compute_gradient_logistic(y, tx, w)
    w -= gamma * grad
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    norm_w = w.T @ w
    loss = compute_loss_logistic(y, tx, w) + lambda_ * norm_w
    norm_gradient = 2 * w       
    gradient = compute_gradient_logistic(y, tx, w) + lambda_ * norm_gradient
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w


#========================================================================================================================== MAIN FUNCTIONS


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

    
def least_squares(y, tx):
    """calculate the least squares solution."""
    X = tx
    w = np.linalg.solve(X.T @ X, X.T @ y)
    loss = compute_loss(y, X, w)
    return w, loss
    
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    X = tx
    N = X.shape[0]
    I = np.identity(X.shape[1])
    lambda_p = 2 * N * lambda_
    w = np.linalg.solve(X.T @ X + lambda_p * I, X.T @ y)   
    loss = compute_loss(y, X, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    threshold = 1e-8
    losses = []

    for iter in range(max_iters):
        loss, w = logistic_regression_one_iter(y, tx, w, gamma)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]
    

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]
 
