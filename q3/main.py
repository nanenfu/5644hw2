import numpy as np
import matplotlib.pyplot as plt

# Generate the vehicle's true position (ensure it's inside the unit circle)
def generate_vehicle_position():
    while True:
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 <= 1:
            return np.array([x, y])

# Generate landmark positions (evenly spaced on the unit circle)
def generate_landmarks(K):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return np.array([[np.cos(a), np.sin(a)] for a in angles])

# Generate measurement data (allow negative measurements)
def generate_measurements(vehicle_pos, landmarks, sigma=0.3):
    true_distances = np.linalg.norm(landmarks - vehicle_pos, axis=1)
    noisy_distances = true_distances + np.random.normal(0, sigma, size=len(landmarks))
    return noisy_distances

# Define the MAP objective function
def map_objective(x, y, measurements, landmarks, sigma=0.3, sigma_x=0.25, sigma_y=0.25):
    vehicle_pos = np.array([x, y])
    predicted_distances = np.linalg.norm(landmarks - vehicle_pos, axis=1)
    measurement_error = np.sum((measurements - predicted_distances)**2) / (2 * sigma**2)
    prior = (x**2) / (2 * sigma_x**2) + (y**2) / (2 * sigma_y**2)
    return measurement_error + prior

# Use grid search to find the MAP estimate
def find_map_estimate(x_vals, y_vals, measurements, landmarks):
    min_value = float('inf')
    best_position = None

    for x in x_vals:
        for y in y_vals:
            value = map_objective(x, y, measurements, landmarks)
            if value < min_value:
                min_value = value
                best_position = np.array([x, y])
    
    return best_position

# Plot the contour graph and save it as a PNG file
def plot_contours(vehicle_pos, landmarks, measurements, K):
    x_vals = np.linspace(-2, 2, 200)
    y_vals = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[map_objective(x, y, measurements, landmarks) for x in x_vals] for y in y_vals])

    # Find the MAP estimate
    map_estimate = find_map_estimate(x_vals, y_vals, measurements, landmarks)

    # Calculate the error
    error = np.linalg.norm(map_estimate - vehicle_pos)
    print(f"K={K} Estimated Position: {map_estimate}, Prediction Error: {error:.4f}")

    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30)
    plt.plot(vehicle_pos[0], vehicle_pos[1], 'r+', markersize=10, label='True Position')
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'bo', label='Landmarks')
    plt.plot(map_estimate[0], map_estimate[1], 'gx', markersize=10, label='MAP Estimate')
    plt.legend()
    plt.title(f'MAP Objective Contour (K={K})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    # Save the graph as a PNG file
    filename = f'map_contour_K1_{K}.png'
    plt.savefig(filename)
    print(f"Saved contour plot for K={K} as '{filename}'")
    plt.close()

# Main function
def main():
    vehicle_pos = generate_vehicle_position()
    print(f"True Position: {vehicle_pos}")

    for K in [1, 2, 3, 4]:
        landmarks = generate_landmarks(K)
        measurements = generate_measurements(vehicle_pos, landmarks)
        print(f"K={K} Landmark Positions:\n{landmarks}")
        print(f"K={K} Measurements: {measurements}\n")
        plot_contours(vehicle_pos, landmarks, measurements, K)

# Execute the main function
if __name__ == "__main__":
    main()
