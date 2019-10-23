import numpy as np
import matplotlib.pyplot as plt

n_dots=200
x=np.linspace(0,1,n_dots)
y=np.sqrt(x)+0.2*np.random.rand(n_dots)-0.1
x=x.reshape(-1,1)
y=y.reshape(-1,1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def polynomial_model(degree):
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression()
    pipeline=Pipeline([('polynomial_features',polynomial_features),
                       ('linear_regression',linear_regression)])
    return pipeline

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_size,train_scores,test_scores=learning_curve(estimator,X,y,
                                                       cv=cv,n_jobs=n_jobs,
                                                       train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_size, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_size, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_size, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == '__main__':
    cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=78)
    titles = ['Learning Curves (Under Fitting)',
              'Learning Curves',
              'Learning Curves (Over Fitting)']
    degrees = [1, 3, 10]

    plt.figure(figsize=(18,4))
    for index,i in enumerate(degrees):
        plt.subplot(1,3,index+1)
        plot_learning_curve(polynomial_model(i),titles[index],x,y,ylim=(0.75, 1.01),cv=cv)

    plt.show()