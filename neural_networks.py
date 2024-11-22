import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
        # Initialize variables to store activations and gradients
        self.X = None
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def forward(self, X):
        # Forward pass, apply layers to input X
        self.X = X  # store input for backprop
        self.Z1 = np.dot(X, self.W1) + self.b1
        # Apply activation function to Z1
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))
        else:
            raise ValueError("Unsupported activation function")

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # Since it's binary classification, use sigmoid activation
        self.A2 = 1 / (1 + np.exp(-self.Z2))
        return self.A2

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == 'tanh':
            A1 = np.tanh(Z1)
        elif self.activation_fn == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation_fn == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        else:
            raise ValueError("Unsupported activation function")

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for output layer

        return A2

    def backward(self, X, y):
        m = y.shape[0]
        # Compute gradients using chain rule
        dZ2 = self.A2 - y
        self.dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        self.db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)

        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig_Z1 = 1 / (1 + np.exp(-self.Z1))
            dZ1 = dA1 * sig_Z1 * (1 - sig_Z1)
        else:
            raise ValueError("Unsupported activation function")

        self.dW1 = (1 / m) * np.dot(X.T, dZ1)
        self.db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights with gradient descent
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

        # Store gradients for visualization (already stored in self.dW1, self.dW2)

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)

    # Hyperplane visualization in the hidden space
    W2 = mlp.W2.flatten()
    b2 = mlp.b2.flatten()
    if W2[2] != 0:
        # Create grid
        x_vals = np.linspace(min(hidden_features[:, 0]), max(hidden_features[:, 0]), 10)
        y_vals = np.linspace(min(hidden_features[:, 1]), max(hidden_features[:, 1]), 10)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = (-W2[0] * X_grid - W2[1] * Y_grid - b2[0]) / W2[2]
        ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='green')
    else:
        # W2[2] is zero, cannot plot plane
        pass

    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')
    ax_hidden.set_title('Hidden Layer Activations')

    # Distorted input space transformed by the hidden layer
    # Create a grid in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_grid, yy_grid = np.meshgrid(np.linspace(x_min, x_max, 20),
                                   np.linspace(y_min, y_max, 20))

    grid_points = np.c_[xx_grid.ravel(), yy_grid.ravel()]
    # Pass through the hidden layer
    Z1 = np.dot(grid_points, mlp.W1) + mlp.b1
    if mlp.activation_fn == 'tanh':
        A1 = np.tanh(Z1)
    elif mlp.activation_fn == 'relu':
        A1 = np.maximum(0, Z1)
    elif mlp.activation_fn == 'sigmoid':
        A1 = 1 / (1 + np.exp(-Z1))
    else:
        raise ValueError("Unsupported activation function")

    # Reshape A1 for plotting
    A1_x = A1[:, 0].reshape(xx_grid.shape)
    A1_y = A1[:, 1].reshape(xx_grid.shape)
    A1_z = A1[:, 2].reshape(xx_grid.shape)

    # Plot the transformed grid in hidden space
    ax_hidden.plot_wireframe(A1_x, A1_y, A1_z, color='grey', alpha=0.1)

    # Plot input layer decision boundary
    # Create a grid over the input space
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Compute model outputs
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax_input.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_xlabel('Input Feature 1')
    ax_input.set_ylabel('Input Feature 2')
    ax_input.set_title('Input Space Decision Boundary')

    # Visualize features and gradients as circles and edges
    # Nodes positions
    input_dim = mlp.W1.shape[0]
    hidden_dim = mlp.W1.shape[1]
    input_neurons = [(0, y_pos) for y_pos in np.linspace(0, 1, input_dim)]
    hidden_neurons = [(1, y_pos) for y_pos in np.linspace(0, 1, hidden_dim)]
    output_neurons = [(2, 0.5)]

    # Plot neurons
    for x, y_pos in input_neurons:
        circle = Circle((x, y_pos), radius=0.05, fill=True, color='lightblue')
        ax_gradient.add_artist(circle)
    for x, y_pos in hidden_neurons:
        circle = Circle((x, y_pos), radius=0.05, fill=True, color='lightgreen')
        ax_gradient.add_artist(circle)
    for x, y_pos in output_neurons:
        circle = Circle((x, y_pos), radius=0.05, fill=True, color='lightcoral')
        ax_gradient.add_artist(circle)

    # Plot edges from input to hidden layer
    max_grad_w1 = np.max(np.abs(mlp.dW1))
    for i, (x1, y1) in enumerate(input_neurons):
        for j, (x2, y2) in enumerate(hidden_neurons):
            weight_grad = mlp.dW1[i, j]
            lw = (abs(weight_grad) / max_grad_w1) * 5 if max_grad_w1 != 0 else 0.1
            ax_gradient.plot([x1, x2], [y1, y2], 'k-', lw=lw)

    # Plot edges from hidden to output layer
    max_grad_w2 = np.max(np.abs(mlp.dW2))
    for i, (x1, y1) in enumerate(hidden_neurons):
        x2, y2 = output_neurons[0]
        weight_grad = mlp.dW2[i, 0]
        lw = (abs(weight_grad) / max_grad_w2) * 5 if max_grad_w2 != 0 else 0.1
        ax_gradient.plot([x1, x2], [y1, y2], 'k-', lw=lw)

    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.1, 1.1)
    ax_gradient.axis('off')
    ax_gradient.set_title('Network Gradients')

    # Add training step text
    fig = ax_input.get_figure()
    fig.suptitle(f'Training Step: {frame * 10}', fontsize=16)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)