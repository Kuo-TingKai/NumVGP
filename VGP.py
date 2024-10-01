import numpy as np
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(0)

# Generate simulated data
def generate_data(N=100):
    X = np.random.rand(N, 1) * 10 - 5
    Y = np.sin(X) + 0.3*np.cos(2*X) + np.random.randn(N, 1) * 0.1
    return X, Y

# RBF kernel function
def rbf_kernel(X, X2, variance, lengthscale):
    dist = np.sum(X**2, 1)[:, None] + np.sum(X2**2, 1) - 2 * np.dot(X, X2.T)
    K = variance * np.exp(-0.5 * dist / (lengthscale**2))
    if X is X2:
        K += 1e-6 * np.eye(X.shape[0])  # Add small diagonal term
    return K

# Variational Gaussian Process model
class VGP:
    def __init__(self, X, Y, num_inducing=10):
        self.X, self.Y = X, Y
        self.num_inducing = num_inducing
        self.Z = np.linspace(X.min(), X.max(), num_inducing).reshape(-1, 1)
        self.q_mu = np.zeros((num_inducing, 1))
        self.q_sqrt = np.eye(num_inducing) * 0.1
        self.likelihood_variance = 0.1
        self.kernel_variance = 1.0
        self.kernel_lengthscale = 1.0

    def predict(self, X_new):
        Kzz = rbf_kernel(self.Z, self.Z, self.kernel_variance, self.kernel_lengthscale)
        Kzx = rbf_kernel(self.Z, X_new, self.kernel_variance, self.kernel_lengthscale)
        Kxx = rbf_kernel(X_new, X_new, self.kernel_variance, self.kernel_lengthscale)

        L = np.linalg.cholesky(Kzz)
        A = np.linalg.solve(L, Kzx)
        
        mean = A.T @ self.q_mu
        var = np.diag(Kxx)[:, None] - np.sum(A**2, axis=0)[:, None] + self.likelihood_variance
        
        return mean.ravel(), var.ravel()  # Ensure returning 1D arrays

    def objective(self, params):
        self.kernel_variance, self.kernel_lengthscale, self.likelihood_variance = np.exp(params[:3])
        self.q_mu = params[3:3+self.num_inducing].reshape(-1, 1)
        self.q_sqrt = np.tril(params[3+self.num_inducing:].reshape(self.num_inducing, self.num_inducing))
        
        # Ensure diagonal elements of q_sqrt are positive
        self.q_sqrt = np.tril(self.q_sqrt)
        self.q_sqrt[np.diag_indices_from(self.q_sqrt)] = np.abs(np.diag(self.q_sqrt))

        Kzz = rbf_kernel(self.Z, self.Z, self.kernel_variance, self.kernel_lengthscale)
        Kzx = rbf_kernel(self.Z, self.X, self.kernel_variance, self.kernel_lengthscale)
        Kxx = rbf_kernel(self.X, self.X, self.kernel_variance, self.kernel_lengthscale)

        L = np.linalg.cholesky(Kzz)
        A = np.linalg.solve(L, Kzx)

        mean = A.T @ self.q_mu  # Calculate mean
        var = np.diag(Kxx)[:, None] - np.sum(A**2, axis=0)[:, None] + self.likelihood_variance

        # KL divergence
        KL = 0.5 * np.sum(self.q_mu**2)
        KL += 0.5 * np.sum(self.q_sqrt**2) - 0.5 * np.sum(np.log(np.diag(self.q_sqrt)**2))
        KL -= 0.5 * self.num_inducing

        # Log likelihood
        epsilon = 1e-6  # Small constant
        LL = -0.5 * np.sum((self.Y - mean)**2 / (var + epsilon))
        LL -= 0.5 * np.sum(np.log(var + epsilon))
        LL -= 0.5 * self.X.shape[0] * np.log(2 * np.pi)

        return KL - LL

    def fit(self):
        initial_params = np.concatenate([
            np.log([self.kernel_variance, self.kernel_lengthscale, self.likelihood_variance]),
            self.q_mu.ravel(),
            self.q_sqrt.ravel()
        ])
        
        # Implement simple gradient descent
        learning_rate = 0.01
        num_iterations = 1000
        
        for _ in range(num_iterations):
            grad = self.compute_gradient(initial_params)
            initial_params -= learning_rate * grad
        
        self.kernel_variance, self.kernel_lengthscale, self.likelihood_variance = np.exp(initial_params[:3])
        self.q_mu = initial_params[3:3+self.num_inducing].reshape(-1, 1)
        self.q_sqrt = np.tril(initial_params[3+self.num_inducing:].reshape(self.num_inducing, self.num_inducing))

    def compute_gradient(self, params):
        # Implement numerical gradient computation
        epsilon = 1e-8
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            grad[i] = (self.objective(params_plus) - self.objective(params_minus)) / (2 * epsilon)
        
        return grad

# Plot results
def plot_results(X, Y, model):
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
    mean, var = model.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(X, Y, color='red', alpha=0.4, label='Observed data')
    plt.plot(X_test, mean, 'b-', label='Predicted mean')
    
    # Add a small constant to avoid negative variance
    epsilon = 1e-6
    plt.fill_between(X_test.ravel(), 
                     mean - 1.96 * np.sqrt(np.maximum(var, epsilon)),
                     mean + 1.96 * np.sqrt(np.maximum(var, epsilon)),
                     color='blue', alpha=0.2, label='95% confidence interval')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Variational Gaussian Process Regression (Pure NumPy Implementation)')
    plt.legend()
    plt.show()

# Main function
def main():
    X, Y = generate_data()
    model = VGP(X, Y)
    model.fit()
    plot_results(X, Y, model)
    
    print("Model parameters:")
    print(f"Kernel variance: {model.kernel_variance}")
    print(f"Kernel lengthscale: {model.kernel_lengthscale}")
    print(f"Likelihood variance: {model.likelihood_variance}")

if __name__ == "__main__":
    main()