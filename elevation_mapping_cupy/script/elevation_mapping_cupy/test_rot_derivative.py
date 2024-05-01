import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# First define the transformation matrix C in terms of
# extrinsically defined Euler angles (roll, pitch, yaw)
# i.e. the rotation matrix is defined as a sequence of
# rotations about the axes of the original coordinate
# system where the axes are fixed in space.
# Basic rotation matrices are defined as:
Rx = lambda a: np.array([[1, 0, 0],
                        [0, np.cos(a), -np.sin(a)],
                        [0, np.sin(a), np.cos(a)]])

Ry = lambda b: np.array([[np.cos(b), 0, np.sin(b)],
                        [0, 1, 0],
                        [-np.sin(b), 0, np.cos(b)]])

Rz = lambda c: np.array([[np.cos(c), -np.sin(c), 0],
                        [np.sin(c), np.cos(c), 0],
                        [0, 0, 1]])

# Skew symmetric matrix of a vector v
SS = lambda v: np.array([[0, -v[2], v[1]],
                                    [v[2], 0, -v[0]],
                                    [-v[1], v[0], 0]])


# Basis vector ei
ei = lambda i: np.array([1 if j == i else 0 for j in range(3)])


def get_ext_euler_angles(C, pitch_estimate=0.0):
    """
    Extract Extrinsically defined Euler angles from rotation matrix
    The transformation matrix C in terms of
    extrinsically defined Euler angles (roll, pitch, yaw)
    is defined as C = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    i.e. first roll about x, then pitch about y, and then yaw about z (in that order)
    i.e. the rotation matrix is defined as a sequence of
    rotations about the axes of the original coordinate
    system where the axes are fixed in space.
    For reference Basic rotation matrices are defined as:
    Rx = lambda a: xp.array([[1, 0, 0],
                            [0, xp.cos(a), xp.sin(a)],
                            [0, -xp.sin(a), xp.cos(a)]])

    Ry = lambda b: xp.array([[xp.cos(b), 0, -xp.sin(b)],
                            [0, 1, 0],
                            [xp.sin(b), 0, xp.cos(b)]])

    Rz = lambda c: xp.array([[xp.cos(c), xp.sin(c), 0],
                            [-xp.sin(c), xp.cos(c), 0],
                            [0, 0, 1]])
    Note that this may be refered to as Extrinsic Tait-Bryan angles in x-y-z order.
    Using Reference: Computing Euler Angles from a Rotation Matrix by Gregory G. Slabaugh
    https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

    2 solutions for pitch always exist. The first always lies between -90 and 90 degrees.
    The second lies between 90 and 270 degrees. We default to the first solution, but allow
    the user to select between them by providing an estimate of the pitch angle.
    """
    if np.abs(C[2, 0]) != 1.0:
        pitch1 = -np.arcsin(C[2,0]) # Different compared to paper due to 
        pitch2 = np.pi - pitch1
        if pitch_estimate > -np.pi/2 and pitch_estimate < np.pi/2:
            pitch = pitch1
        else:
            pitch = pitch2
        roll = np.arctan2(C[2,1]/np.cos(pitch), C[2,2]/np.cos(pitch))
        yaw = np.arctan2(C[1,0]/np.cos(pitch), C[0,0]/np.cos(pitch))
    else: # Gimbal lock: pitch is at -90 or 90 degrees
        yaw = 0.0 # This can be any value, but we choose 0.0 for consistency
        if C[2, 0] == -1.0:
            pitch = np.pi/2
            roll = yaw + np.arctan2(C[0,1], C[0,2])
        else:
            pitch = -np.pi/2
            roll = -yaw + np.arctan2(-C[0,1], -C[0,2])
    return roll, pitch, yaw

# Rotation matrix from extrinsic Euler angles
# For extrinsic rotations is is to roll about x, pitch about y, and yaw about z (in that order)
# This means the multiplication will be in the order Rz*Ry*Rx
ExtEulerRot = lambda roll, pitch, yaw: np.dot(Rz(yaw), np.dot(Ry(pitch), Rx(roll)))

# Change in rotation matrix with respect to small change in Euler angles
def delta_Cr(roll, pitch, yaw, r, drot):
    C1 = ExtEulerRot(roll, pitch, yaw)
    # Compute derivative for each angle
    C2_roll = ExtEulerRot(roll+drot[0], pitch, yaw)
    delta_Cr_roll = (C2_roll.dot(r) - C1.dot(r))/drot[0]
    C2_pitch = ExtEulerRot(roll, pitch+drot[1], yaw)
    delta_Cr_pitch = (C2_pitch.dot(r) - C1.dot(r))/drot[1]
    C2_yaw = ExtEulerRot(roll, pitch, yaw+drot[2])
    delta_Cr_yaw = (C2_yaw.dot(r) - C1.dot(r))/drot[2]
    delta_Cr = np.hstack([delta_Cr_roll[:,np.newaxis], delta_Cr_pitch[:,np.newaxis], delta_Cr_yaw[:,np.newaxis]])
    return delta_Cr

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Taken from: https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    # If eigenvalue is less than a specified tolerance, set it to zero
    eigvals[eigvals < 1e-6] = 0.0
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

# Feel free to change these values that define the rotation matrix
roll_deg = 15.0
pitch_deg = 95.0
# pitch_deg = 10.0
yaw_deg = -5.0
r = np.array([1.0, 2.0, 3.0])

# In implementing the Lucas method the matrix Rx(roll) is required
# If we know the Euler angles then we can compute Rx(roll)
# However, if we only have the rotation matrix C then we need to extract the Euler angles
# Unfortunately, there are two solutions for sets of angles, that produce the same rotation matrix
# If we know the Euler angles (pitch in this case) then we can select the correct set of angles
# If we don't know the Euler angles then we don't know which set of angles to select
# It does affect the Jacobian, but it doesn't seem to affect the error propagation
USE_CORRECT_EULER_QUADRANT = True

# Small value for numerical differentiation
drot = np.array([0.001, 0.001, 0.001])

roll = roll_deg * np.pi/180
pitch = pitch_deg * np.pi/180
yaw = yaw_deg * np.pi/180
C = ExtEulerRot(roll, pitch, yaw)
print("Rotation matrix C: \n", C)

# Now define the derivative of the rotation matrix with respect to the Euler angles
# Numerical differentiation
# Verify that the derivative is correct using numerical differentiation
delta_Cr_num = delta_Cr(roll, pitch, yaw, r, drot)
print("Derivative of rotation matrix C * r with respect to roll, pitch, and yaw using numerical differentiation: \n", delta_Cr_num)

# First using method outlined by James R. Lucas in "Differentiation of the Orientation Matrix by Matrix Multipliers"
# https://www.asprs.org/wp-content/uploads/pers/1963journal/jul/1963_jul_708-715.pdf
roll_est, pitch_est, yaw_est = get_ext_euler_angles(C, pitch_estimate=pitch if USE_CORRECT_EULER_QUADRANT else 0.0)
# Compare the estimated Euler angles with the original Euler angles
print("Original Euler angles: ", roll, pitch, yaw)
print("Estimated Euler angles: ", roll_est, pitch_est, yaw_est)
Qj = (Rx(roll_est).T) @ (SS(ei(1))) @ Rx(roll_est)
dC_dyaw = SS(ei(2)) @ C
dC_dpitch = C @ Qj
dC_droll = C @ (SS(ei(0)))

dCr_droll = (dC_droll @ r)[:,np.newaxis]
dCr_dpitch = (dC_dpitch @ r)[:,np.newaxis]
dCr_dyaw = (dC_dyaw @ r)[:,np.newaxis]

dCr_lucas = np.hstack([dCr_droll, dCr_dpitch, dCr_dyaw])
print("Derivative of rotation matrix C*r with respect to roll, pitch, and yaw using Lucas method: \n", dCr_lucas)

# Now using the method implied by the math in Fankhauser's Robot-centric elevation mapping with uncertainty estimates
# From equation 2 without the projection P
dCr_fankhauser = C @ SS(r)
print("Derivative of rotation matrix C*r with respect to roll, pitch, and yaw using Fankhauser method: \n", dCr_fankhauser)

# Are they the same?
# print("Difference between the two methods: \n", dCr_lucas - dCr_fankhauser)

# Now use the method described by Bloesch et al. in "A Primer on the Differential Calculus of 3D Orientations"
# From equation 27
dC_r_bloesch = -SS(C.dot(r))
print("Derivative of rotation matrix C with respect to r using Bloesch method: \n", dC_r_bloesch)

##########################################################################################
# Now let's test the Jacobian of rotation matrix C applied to vector r
# with respect to extrinsic (fixed-axis) Euler angles (roll, pitch, yaw) x-y-z.
# randomly generate a covariance matrix by defining a lower triangular matrix L
# indx, indy = np.triu_indices(3)
# L = np.zeros((3,3))
# L[indx, indy] = np.random.randn(len(indx))
# cov = L @ L.T
# Could also generate one by rotating a diagonal matrix
vars = np.diag([10.0**2, 1.0**2, 20.0**2])*(np.pi/180)**2
# vars = np.diag([50.0, 1.0, 10.0])*(np.pi/180)
# Rotate the diagonal matrix to produce a full covariance matrix (non-diagonal)
# Set var_ values to 0 to test the diagonal case
# var_r = 0.0 * np.pi/180
# var_p = 0.0 * np.pi/180
# var_y = 0.0 * np.pi/180
var_r = 10.0 * np.pi/180
var_p = 30.0 * np.pi/180
var_y = -7.0 * np.pi/180
R = ExtEulerRot(var_r, var_p, var_y)
Sigma = R @ vars @ R.T

# Error Propagation using numerical Jacobian
sigma_num = delta_Cr_num @ Sigma @ delta_Cr_num.T
print("Error propagation using numerical Jacobian: \n", sigma_num)

# Error Propagation using Lucas method
sigma_lucas = dCr_lucas @ Sigma @ dCr_lucas.T
print("Error propagation using Lucas method: \n", sigma_lucas)

# Error Propagation using Fankhauser method
sigma_fankhauser = dCr_fankhauser @ Sigma @ dCr_fankhauser.T
print("Error propagation using Fankhauser method: \n", sigma_fankhauser)

# Error Propagation using Bloesch method
sigma_bloesch = dC_r_bloesch @ Sigma @ dC_r_bloesch.T
print("Error propagation using Bloesch method: \n", sigma_bloesch)

# Difference between the two methods
print("Difference between the two methods: \n", sigma_num - sigma_lucas)
# Error of each component as a percentage of the sigma_num
print("Percentage difference between the two methods: \n", (sigma_num - sigma_lucas)/sigma_num*100)

# Plot sigma_num and sigma_lucas as 3d Gaussian ellipsoids
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlabel('x')
plt.ylabel('y')
# X1,Y1,Z1 = get_cov_ellipsoid(Sigma, nstd=1)
# ax.plot_wireframe(X1,Y1,Z1, color='r', alpha=0.1)
X2,Y2,Z2 = get_cov_ellipsoid(sigma_num, nstd=1)
ax.plot_wireframe(X2,Y2,Z2, color='b', alpha=0.1)
X3,Y3,Z3 = get_cov_ellipsoid(sigma_lucas, nstd=1)
ax.plot_wireframe(X3,Y3,Z3, color='g', alpha=0.1)
X4,Y4,Z4 = get_cov_ellipsoid(sigma_fankhauser, nstd=1)
ax.plot_wireframe(X4,Y4,Z4, color='k', alpha=0.1)
X5,Y5,Z5 = get_cov_ellipsoid(sigma_bloesch, nstd=1)
ax.plot_wireframe(X5,Y5,Z5, color='m', alpha=0.1)
# Set axes limits to be the same
max_range = np.array([X2.max()-X2.min(), Y2.max()-Y2.min(), Z2.max()-Z2.min()]).max() / 2.0
mid_x = (X2.max()+X2.min()) * 0.5
mid_y = (Y2.max()+Y2.min()) * 0.5
mid_z = (Z2.max()+Z2.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.show()
# Note that projected distribution should be fairly flat in direction of r
# Errors in roll, pitch, and yaw lead to a transformation of the Gaussian ellipsoid
# into a concave "windshield" shape which gets approximated as a very thin Gaussian.