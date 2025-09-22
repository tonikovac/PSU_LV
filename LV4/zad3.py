import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) \
        - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y, seed=14):
    np.random.seed(seed)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy

def polynomial_regression(x, y, degrees=[2,6,15], train_ratio=0.7):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    
    np.random.seed(12)
    indices = np.random.permutation(len(x))
    n_train = int(train_ratio * len(x))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    xtrain = x[train_idx]
    ytrain = y[train_idx]
    xtest = x[test_idx]
    ytest = y[test_idx]
    
    MSEtrain = []
    MSEtest = []

    x_plot = np.linspace(np.min(x), np.max(x), 300)[:, np.newaxis]
    plt.figure(figsize=(10,6))
    plt.plot(x_plot, non_func(x_plot.flatten()), 'k--', label='True function')
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        Xtrain_poly = poly.fit_transform(xtrain)
        Xtest_poly = poly.transform(xtest)
        Xplot_poly = poly.transform(x_plot)
        
        model = lm.LinearRegression()
        model.fit(Xtrain_poly, ytrain)
        
        ytrain_pred = model.predict(Xtrain_poly)
        ytest_pred = model.predict(Xtest_poly)
        
        mse_train = mean_squared_error(ytrain, ytrain_pred)
        mse_test = mean_squared_error(ytest, ytest_pred)
        
        MSEtrain.append(mse_train)
        MSEtest.append(mse_test)
        
        y_plot_pred = model.predict(Xplot_poly)
        plt.plot(x_plot, y_plot_pred, label=f'Degree {degree}')
    
    plt.scatter(xtrain, ytrain, color='blue', label='Train', alpha=0.6)
    plt.scatter(xtest, ytest, color='red', label='Test', alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Polinomska regresija (train_ratio={train_ratio})')
    plt.legend()
    plt.show()
    
    return np.array(MSEtrain), np.array(MSEtest)

x = np.linspace(1,10,50)
y_true_noisy = add_noise(non_func(x))

MSEtrain, MSEtest = polynomial_regression(x, y_true_noisy, degrees=[2,6,15], train_ratio=0.7)

print("MSE na train skupu:", MSEtrain)
print("MSE na test skupu:", MSEtest)

x_small = np.linspace(1,10,15)
y_small = add_noise(non_func(x_small), seed=21)
MSEtrain_small, MSEtest_small = polynomial_regression(x_small, y_small, degrees=[2,6,15], train_ratio=0.7)

print("MSE na train (malo uzoraka):", MSEtrain_small)
print("MSE na test (malo uzoraka):", MSEtest_small)

x_large = np.linspace(1,10,200)
y_large = add_noise(non_func(x_large), seed=31)
MSEtrain_large, MSEtest_large = polynomial_regression(x_large, y_large, degrees=[2,6,15], train_ratio=0.7)

print("MSE na train (više uzoraka):", MSEtrain_large)
print("MSE na test (više uzoraka):", MSEtest_large)
