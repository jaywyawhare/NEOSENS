import numpy as np

def generate_lorenz_data(num_steps=10000, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
    sigma = 10.0
    beta = 8/3
    rho = 28.0
    data = np.zeros((num_steps, 3))
    data[0] = initial_state
    for t in range(1, num_steps):
        x, y, z = data[t-1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        data[t] = data[t-1] + dt * np.array([dx, dy, dz])
    return data

def generate_rossler_data(num_steps=10000, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
    a, b, c = 0.2, 0.2, 5.7
    data = np.zeros((num_steps, 3))
    data[0] = initial_state
    for t in range(1, num_steps):
        x, y, z = data[t-1]
        dx = -y - z
        dy = x + a*y
        dz = b + z*(x - c)
        data[t] = data[t-1] + dt * np.array([dx, dy, dz])
    return data

def generate_mackey_glass_data(num_steps=2000, tau=17, n=10, beta=0.2, gamma=0.1, dt=1.0):
    data = np.zeros(num_steps)
    data[:tau] = 1.2
    for t in range(tau, num_steps):
        data[t] = data[t-1] + dt * (beta * data[t-tau] / (1 + data[t-tau]**n) - gamma * data[t-1])
    return data.reshape(-1, 1)

def generate_sine_data(num_steps=2000):
    x = np.linspace(0, 8*np.pi, num_steps)
    data = np.sin(x)
    return data.reshape(-1, 1)
