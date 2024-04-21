import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define the dimensions of the matrix
n = 1000
k = 2  # Use 2D data for easier visualization

# Create the matrix
data = np.random.randn(n, k)

# Re-center the matrix around the first row
recentered_data = data - data[0]

# Define the rotation angle in degrees
rotation_angle_deg = 45
# Convert rotation angle to radians
rotation_angle_rad = np.deg2rad(rotation_angle_deg)

# Define scaling factors for each axis
scaling_x = 1.5
scaling_y = 0.75

# Define shear factors for each axis
shear_x = 0.5
shear_y = 0.2

# Create a 2D rotation matrix
rotation_matrix = np.array([
    [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
    [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]
])

# Create a 2D scaling matrix
scaling_matrix = np.array([
    [scaling_x, 0],
    [0, scaling_y]
])

# Create a 2D shear matrix
shear_matrix = np.array([
    [1, shear_x],
    [shear_y, 1]
])

# Combine rotation, scaling, and shear matrices
transformation_matrix = np.dot(rotation_matrix, np.dot(scaling_matrix, shear_matrix))

# Apply the transformation matrix to the recentered data
transformed_data = np.dot(recentered_data, transformation_matrix)

# Normalize the transformed data to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_transformed_data = scaler.fit_transform(transformed_data)

# Plotting the original and normalized transformed data
plt.figure(figsize=(10, 5))

# Plot original data (re-centered)
plt.subplot(1, 2, 1)
plt.scatter(recentered_data[:, 0], recentered_data[:, 1], color='b', label='Original Data')
plt.scatter(recentered_data[0, 0], recentered_data[0, 1], color='r', marker='*', s=100, label='First Row')
plt.title('Recentered Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Plot normalized transformed data
plt.subplot(1, 2, 2)
plt.scatter(normalized_transformed_data[:, 0], normalized_transformed_data[:, 1], color='r', label='Normalized Transformed Data')
plt.scatter(normalized_transformed_data[0, 0], normalized_transformed_data[0, 1], color='b', marker='*', s=100, label='First Row')
plt.title('Normalized Transformed Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
