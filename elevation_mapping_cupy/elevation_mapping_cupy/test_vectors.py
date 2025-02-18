import numpy as np

# Script that verifies method for generation of coordinate system with z axis aligned with a vector

p = np.array([0, 2, 0])
print("Vector p: ", p)
# get length of vector
d = np.linalg.norm(p)
print("Length of vector p: ", d)

# Define new coordinate system from z
Z = p/d
print("Vector Z: ", Z)
# Choose X such that X is orthogonal to Z
c = 0
if Z[0] != 0:
    b = np.sqrt(1/(1+Z[1]**2/Z[0]**2))
    a = -Z[1]*b/Z[0]
else:
    b = 0
    if Z[2] > 0:
        a = 1
    else:
        a = -1
X = np.array([a, b, c])
print("Vector X: ", X)
# Double check that X is orthogonal to Z
print("Dot product of X and Z: ", np.dot(X, Z))
# Choose Y such that Y is orthogonal to Z and X
Y = np.cross(Z, X)
print("Vector Y: ", Y)
# Double check that Y is orthogonal to Z and X
print("Dot product of Y and Z: ", np.dot(Y, Z))
print("Dot product of Y and X: ", np.dot(Y, X))

# Define a coordinate transform from the original coordinate system to the new coordinate system
T = np.array([X, Y, Z])
print("Coordinate transform matrix T: \n", T)

# Apply the coordinate transform to the original vector p
p_new = np.dot(T, p)
print("Vector p in new coordinate system: ", p_new)

# Find the quaternion representation of the transform T
q0 = np.sqrt(1 + T[0,0] + T[1,1] + T[2,2])/2
# May need to use this approach to avoid numerical issues: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
if q0 != 0:
    q1 = (T[2,1] - T[1,2])/(4*q0)
    q2 = (T[0,2] - T[2,0])/(4*q0)
    q3 = (T[1,0] - T[0,1])/(4*q0)
    q = np.array([q0, q1, q2, q3])
    print("Quaternion representation of the transform T: ", q)

    # Convert to axis-angle representation
    angle = 2*np.arccos(q[0])
    axis = q[1:]/np.sin(angle/2)
    print("Axis-angle representation of the transform T: ", axis, angle*180/np.pi)

# Now apply Transform to covariance matrix
alpha_radial = 1.0
sigma_radial = alpha_radial*d**2
sigma_lateral = 0.0
# Define the covariance matrix in the z axis aligned coordinate system
Sigma = np.array([[sigma_lateral**2, 0, 0], [0, sigma_lateral**2, 0], [0, 0, sigma_radial**2]])
print("Covariance matrix Sigma in D coordinate system: \n", Sigma)
# Apply the coordinate transform to the covariance matrix
# Not T.T first becasue T = C_ds, we need C_sd
Sigma_new = np.dot(np.dot(T.T, Sigma), T)
print("Covariance matrix Sigma in S coordinate System: \n", Sigma_new)

# Assuming sensor has same orientation as map. (not a good assumption, but just testing)
# Project the covariance matrix onto the z axis
sigma_z = np.sqrt(Sigma_new[2,2])
print("Variance of z axis in S/map coordinate system: ", sigma_z)

# How it was being done in EM_cupy before
var_z_old = alpha_radial* p[2]**2
sigma_z_old = np.sqrt(var_z_old)
print("Covariance of z axis in S/map coordinate system (old): ", sigma_z_old)

# How it was being done in EM (no gpu) prior to this change: https://github.com/ANYbotics/elevation_mapping/commit/f27685ba1b0cc9347574daabfeacae5aa69903e9
# Essentially using d instead of z. Simplifying function a bit though
sigma_z_old = alpha_radial* d**2
