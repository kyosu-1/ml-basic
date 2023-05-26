import numpy as np
import matplotlib.pyplot as plt
from linear_model import OLSRegressor, RidgeRegressor, LassoRegressor

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100

# Generate random feature X
X = np.random.uniform(-10, 10, size=n_samples)

# Generate normally distributed noise
noise = np.random.normal(0, 1, size=n_samples)

# Generate target variable y with a linear relationship to X
y = 3 * X + 5 + noise

# Reshape X and y to fit the requirements of sklearn models
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Plot the data
plt.scatter(X, y)
plt.title("Generated linear regression data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Assume X, y are the feature matrix and target vector, respectively.
ols = OLSRegressor()
ols.load_dataset(X, y)
ols.train_model()
ols.evaluate_model()

ridge = RidgeRegressor(alpha=0.5)
ridge.load_dataset(X, y)
ridge.train_model()
ridge.evaluate_model()

lasso = LassoRegressor(alpha=0.5)
lasso.load_dataset(X, y)
lasso.train_model()
lasso.evaluate_model()