import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed

iris_data = None


class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle = True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if(random_state):
            seed(random_state)

    def fit(self, X, y):
        """
        Fit the training data
        :param X: Training vectors
        :param y: Target values
        :return: self
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Fit traing data without initializing the weights
        :param X: Training data
        :param y: Target values
        :return: self
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        Shuffle training data set
        :param X: Training vectors
        :param y: Target values
        :return: Shuffled training data set
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        Initialize weights to zeros
        :param m: The number of data samples
        :return: self
        """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        Apply Adaline learning rule to update the weights
        :param xi: Training vector
        :param target: Target value
        :return: The deviation of the computed value for xi from the target value
        """
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, xi):
        """
        Calculate the net input
        :param xi: Training vector
        :return: The net input
        """
        return np.dot(xi, self.w_[1:]) + self.w_[0]

    def activation(self, xi):
        """
        Compute the linear activation
        :param xi: Training vector
        :return: Linear activation
        """
        return self.net_input(xi)

    def predict(self, xi):
        """
        Return class label after unit step
        :param xi: Training vector
        :return: Class label after unit step
        """
        return np.where(self.activation(xi) >= 0, 1, -1)



class AdalineGD(object):
    """
    ADAptive linear neuron classifier

    Parameters:
    -----------
    eta : float
        Learning rate
    n_iter : int
        Passes over the training data set

    Attributes:
    -----------
    w_  : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=50):
        """
        :param eta: Learning rate between 0.0 and 1.0
        :param n_iter: Passes over the training data set
        :return: self
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit the training data by reduction in gradient-direction
        :param X: Training data sets
        :param y: Target values
        :return: self
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            ## This is the gradient of the cost function
            ## J(w) = sum_i (y_i-w*x_i)
            self.w_[1:] += self.eta * X.T.dot(errors) ## Gradient
            self.w_[0] += self.eta * errors.sum() ## Gradient
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, xi):
        """
        Calculate the net input
        :param xi: Training vector
        :return: the linear transform of xi
        """
        return np.dot(xi, self.w_[1:]) + self.w_[0]

    def activation(self, xi):
        """
        Calculate linear activation
        :param xi: Training vector
        :return: the linear trasform of xi
        """
        return self.net_input(xi)

    def predict(self, xi):
        """
        Return class label after unit step
        :param xi: Training vector
        :return: The class label after unit step
        """
        return np.where(self.activation(xi) >= 0.0, 1, -1)


class Perceptron(object):

    """Perceptron classifier

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 0.1)
    n_iter : int
        Passages over the training dataset

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data
        :param X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples
                and n_features is the number of features
        :param y: {array-like}, shape = [n_samples]
                Target values
        :return: self.object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, xi):
        """
        Calculate net input
        :param xi: Training vector
        :return: Net input
        """
        return np.dot(xi, self.w_[1:]) + self.w_[0]

    def predict(self, xi):
        """
        Return class label after unit step
        :param xi: Training vector
        :return: The class label after unit step
        """
        return np.where(self.net_input(xi) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unicode(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


def get_iris_data():
    import pandas as pd
    global iris_data
    if iris_data is None:
        iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    return iris_data


def get_X_y(standardized = False):
    df = get_iris_data()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    if standardized:
        X_std = np.copy(X)
        X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
        X = X_std
    return X, y


def iris_scatter_plot():
    df = get_iris_data()
    (X, y) = get_X_y()
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()


def plot_performance():
    ppn = Perceptron(eta=0.1, n_iter=10)
    (X, y) = get_X_y()
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()


def show_adaline_learning_rate():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    (X, y) = get_X_y()
    ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum squared errors)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(eta=0.0001, n_iter=10).fit(X,y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum squared errors)')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()


def plot_result(classifier, title, standardized=True):
    X, y = get_X_y(standardized=standardized)
    classifier.fit(X, y)
    plot_decision_regions(X, y, classifier=classifier)
    plt.title(title)
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()
    plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()

def main1():
    (X, y) = get_X_y()
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    ada1 = AdalineGD(eta=0.01, n_iter=10)
    ada2 = AdalineSGD(eta=0.01, n_iter=10)
    plot_result(ada1, 'Adaline GD')
    plot_result(ada2, 'Adaline SGD')

