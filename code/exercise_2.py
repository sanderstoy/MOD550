from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
import timeit as it

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

karg = {'observed': observed, 'predicted': predicted}

factory = {'mse_vanilla' : vanilla_mse,
           'mse_numpy' : numpy_mse,
           'mse_sk' : lambda observed, predicted: sk_mse(y_true=observed, y_pred=predicted)
           }

for talker, worker in factory.items():
    exec_time = it.timeit('{worker(**karg)}',
                          globals=globals(), number=100) / 100
    mse = worker(**karg)
    print(f"Mean Squared Error, {talker} :", mse, 
          f"Average execution time: {exec_time} seconds")
    
mse_values = [worker(**karg) for worker in factory.values()]
if mse_values[0] == mse_values[1] == mse_values[2]:
    print("Test successful")



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam

class generate():
    ''' Initialization of the class '''
    def __init__(self, points, noise):
        self.points = points
        self.noise = np.random.rand(self.points)

    def generate_data(self):
        ''' Generate data '''
        x = np.linspace(0, 5, self.points)
        y = np.sin(np.pi*x)
        y_noise = y + self.noise/2
        data = np.column_stack((x, y))
        data_noise = np.column_stack((x, y_noise))
        return data, data_noise
    

    def cluster(self, n_clusters):
        ''' Cluster the data '''
        data, data_noise = self.generate_data()
        print(f'Data generated: {len(data)}, {data.shape} ')
        kmeans = KMeans(n_clusters=n_clusters, random_state = 12)
        kmeans.fit(data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        print(f'Clustering method K-means: Clusters: {n_clusters}, Centroids: {centroids}, Labels: {labels}')
        variance = []
        for i in range(n_clusters):
            y_cluster = data[labels == i, 1]
            variance_cluster = np.var(y_cluster)
            variance.append(variance_cluster)
        for i, var in zip(range(n_clusters), variance):
            print(f'Cluster {i+1} Variance: {var}')
        return data, labels, centroids, variance

    
    def plot_cluster(self, n_clusters):
        data, labels, centroids, variance = self.cluster(n_clusters)
        # plt.figure(figsize=(10, 6))
        # for i, color in zip(range(n_clusters), ['r', 'g', 'b']):
            # plt.scatter(data[labels == i, 0], data[labels == i, 1], c=color, label=f'Cluster {i+1}')
            # plt.scatter(centroids[i, 0], centroids[i, 1], s=100, c='yellow', label='Centroids')
        # plt.title('K-means clustering', c='white')
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()    
        plt.figure()
        plt.scatter(range(1, n_clusters+1), variance)
        plt.title('Variance of each cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Variance')
        plt.grid()
        plt.show()


# Create an instance of the class
data = generate(150, 10)
data.plot_cluster(n_clusters = 3)
data.cluster(n_clusters = 3)


class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1,50)  # 1 input feature, 50 output features
            self.fc2 = nn.Linear(50,50) # 50 input features, 50 output features
            self.fc3 = nn.Linear(50,1)  # 50 input features, 1 output feature

        def forward(self, x):
            x = torch.tanh(self.fc1(x)) # Apply tanh activation function
            x = torch.tanh(self.fc2(x)) # Apply tanh activation function
            return self.fc3(x)
        
def calculate_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)   # Calculate error
        

def train_test():
    real_data, data_noise = data.generate_data()
    x_noise = data_noise[:, 0].reshape(-1, 1)   # Reshape for PyTorch
    y_noise = data_noise[:, 1].reshape(-1, 1)   # Reshape for PyTorch
    y_real = real_data[:, 1].reshape(-1, 1)     # Reshape for PyTorch

    x_tensor = torch.tensor(x_noise, dtype=torch.float32)      # Convert to PyTorch Tensor
    y_tensor = torch.tensor(y_noise, dtype=torch.float32)      # Convert to PyTorch Tensor
    y_true_tensor = torch.tensor(y_real, dtype=torch.float32)  # Convert to PyTorch Tensor
     # Train-test split
    x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test_split(x_tensor, y_tensor, y_true_tensor, test_size=0.4, random_state=12)
    return x_train, x_test, y_train, y_test, y_true_train, y_true_test

def linear_regression():
    x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test()
    x_train_np = x_train.numpy() # Convert to NumPy for sklearn
    y_train_np = y_train.numpy() # Convert to NumPy for sklearn
    x_test_np = x_test.numpy()   # Convert to NumPy for sklearn
    y_test_np = y_test.numpy()   # Convert to NumPy for sklearn

    reg = LinearRegression()            # Initialize model
    reg.fit(x_train_np, y_train_np)     # Fit model to training data
    y_pred_lr = reg.predict(x_test_np)  # Predict on test data
    y_pred_lr = torch.tensor(y_pred_lr, dtype=torch.float32) # Convert to PyTorch Tensor
    # print('Mean Squared Error:', calculate_error(y_true_test, y_pred_lr)) # Calculate error
    fig, ax = plt.subplots(1,3)
    ax[0].scatter(x_test_np, y_test_np)
    ax[0].plot(x_test_np, y_pred_lr, color='red')
    ax[0].set_xlabel("X (Feature)")
    ax[0].set_ylabel("Y (Predicted)")
    ax[0].set_title("Linear Regression")
    ax[1].axhline(calculate_error(y_true_test, y_pred_lr))
    ax[1].set_title("MSE, true vs predicted")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Error")
    ax[1].grid()
    ax[2].axhline(calculate_error(y_test, y_pred_lr), alpha=0.5)
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Error")
    ax[2].set_title("MSE, noisy test data vs predicted")
    ax[2].grid()
    print(f'Task completed: Linear regression')
    plt.show()

def neural_network():
    ##### Neural Network #####
    x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test()  # Train-test split
    # Initialize model
    model = NeuralNetwork() # Initialize model
    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Optimizer
    error_nn = [] # Error list
    loss_nn = []  # Loss list
    counter = []  # Counter list
    predictions_nn = {} # Predictions dictionary
    epochs = 600    # Number of epochs
    # Training loop
    for epoch in range(epochs):
        model.train()   # Set model to training mode
        optimizer.zero_grad()   
        y_pred_train = model(x_train)   # Predict
        loss = criterion(y_pred_train, y_train) # Calculate loss
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
        # plot the regression line every 100 epochs
        if epoch % 100 == 0:
            counter.append(epoch)   # Append counter
            predictions_nn[epoch] = y_pred_train.detach().numpy()   # Append predictions
            model.eval()    # Set model to evaluation mode
            with torch.no_grad():
                y_pred_nn = model(x_test)   # Predict on test data
                error_nn.append(calculate_error(y_true_test, y_pred_nn))    # Calculate error on true test data
                loss_nn.append(criterion(y_pred_nn, y_test))                # Calculate loss on noisy test data


    y_pred_np = y_pred_nn.numpy()   # Convert to NumPy for visualization
    x_test_np = x_test.numpy()      # Convert to NumPy for visualization
    y_test_np = y_test.numpy()      # Convert to NumPy for visualization

    fig, ax = plt.subplots(2,2)
    # plots the evolution of the regression line every 100 epochs.
    for epoch in [0, 50, 100, 200, 300, 400]:
        if epoch in predictions_nn:
            sort_idx = np.argsort(x_train.numpy().flatten())    
            x_sorted = x_train.numpy()[sort_idx]    
            y_sorted = predictions_nn[epoch][sort_idx]
            ax[0,1].scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label="Training Data")
            ax[0,1].plot(x_sorted, y_sorted, label=f'Epoch {epoch}')
            ax[0,1].set_xlabel("X (Feature)")
            ax[0,1].set_ylabel("Y (Predicted)")
            ax[0,1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[0,1].set_title("Neural Network Regression")
            ax[0,1].grid()
    # plots the error with respect to the truth
    ax[1,0].plot(counter, error_nn, label="Error")
    ax[1,0].set_xlabel("Epoch")
    ax[1,0].set_ylabel("error")
    ax[1,0].legend()
    ax[1,0].set_title("Error with respect to the truth")
    ax[1,0].grid()
    ax[1,1].plot(counter, loss_nn, label="Loss")
    ax[1,1].set_xlabel("Epoch")
    ax[1,1].set_ylabel("Loss")
    ax[1,1].legend()
    ax[1,1].set_title("Loss")
    ax[1,1].grid()

    # Sort the predictions for visualization
    # Final state plot
    sort_idx = np.argsort(x_test_np[:, 0])
    x_sorted_, y_pred_sorted = x_test_np[sort_idx], y_pred_np[sort_idx]
    ax[0,0].scatter(x_test_np, y_test_np, label="Test Data", alpha=0.7)
    ax[0,0].plot(x_sorted_, y_pred_sorted, color='red', label=f"Epoch {epochs}")
    ax[0,0].set_xlabel("X (Feature)")
    ax[0,0].set_ylabel("Y (predicted)")
    ax[0,0].legend()
    ax[0,0].set_title("Neural Network Regression")
    ax[0,0].grid()
    print(f'Task completed: Neural network')
    plt.show()


class PINN_model(nn.Module):
    def __init__(self):
        super(PINN_model, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)
    

def pinn():
    x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test()  # Train-test split
    pinn = PINN_model()
    t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)
    criterion = nn.MSELoss()  # Loss function
    optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)  # Learning rate: 0.001

    loss_list = []
    error_pinn = []
    counter = []
    predictions_pinn = {}
    epoch = 4000
    for i in range(epoch):
        pinn.train()
        optimiser.zero_grad()
        y_pred = pinn(x_train)
        loss = criterion(y_pred, y_train)
        u = pinn(t_physics)
        dudx = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(dudx, t_physics, torch.ones_like(dudx), create_graph=True)[0]
        loss_pinn = torch.mean((d2u_dt2 + u) ** 2) 
        total_loss = loss + loss_pinn
        total_loss.backward()
        optimiser.step()
        if i % 100 == 0:
            pinn.eval()
            counter.append(i)
            predictions_pinn[i] = y_pred.detach().numpy()
            with torch.no_grad():
                y_pred_pinn = pinn(x_test)
                error_pinn.append(calculate_error(y_true_test, y_pred_pinn))
                loss_list.append(criterion(y_pred_pinn, y_test).item())
        
    y_pred_np_pinn = y_pred_pinn.numpy()   # Convert to NumPy for visualization
    x_test_np_pinn = x_test.numpy()        # Convert to NumPy for visualization
    y_test_np_pinn = y_test.numpy()        # Convert to NumPy for visualization
        
    sort_idx = np.argsort(x_test_np_pinn[:, 0])
    x_sorted_pinn, y_pred_sorted_pinn = x_test_np_pinn[sort_idx], y_pred_np_pinn[sort_idx]
    
    fix, ax = plt.subplots(2, 2)
    ax[0,0].scatter(x_test_np_pinn, y_test_np_pinn, label="Test Data", alpha=0.7)
    ax[0,0].plot(x_sorted_pinn, y_pred_sorted_pinn, color='red', label=f"Epoch {epoch}")
    ax[0,0].set_xlabel("X")
    ax[0,0].set_ylabel("Y")
    ax[0,0].legend()
    ax[0,0].set_title("PINNs Regression")
    ax[0,0].grid()

    ax[1,1].plot(counter, error_pinn, label="Error")
    ax[1,1].set_xlabel("Epoch")
    ax[1,1].set_ylabel("Error")
    ax[1,1].legend()
    ax[1,1].set_title("Error with respect to the truth")
    ax[1,1].grid()

    ax[1,0].plot(counter, loss_list, label="Loss")
    ax[1,0].set_xlabel("Epoch")
    ax[1,0].set_ylabel("Loss")
    ax[1,0].legend()
    ax[1,0].set_title("Loss")
    ax[1,0].grid()

    for epoch in [0, 300, 600, 900, 1200, 2000, 3000]:
        if epoch in predictions_pinn:
            sort_idx = np.argsort(x_train.numpy().flatten())    
            x_sorted_pinn_ = x_train.numpy()[sort_idx]    
            y_sorted_pinn_ = predictions_pinn[epoch][sort_idx]
            ax[0,1].scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label="Training Data")
            ax[0,1].plot(x_sorted_pinn_, y_sorted_pinn_, label=f'Epoch {epoch}')
            ax[0,1].set_xlabel("X (Feature)")
            ax[0,1].set_ylabel("Y (Predicted)")
            ax[0,1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[0,1].set_title("PINNs Regression")
            ax[0,1].grid()
    print('Task completed: PINNs')
    plt.show()

linear_regression()
neural_network()
pinn()