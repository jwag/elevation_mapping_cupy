import numpy as np

# First define the transformation matrix C in terms of
# extrinsically defined Euler angles (roll, pitch, yaw)
# i.e. the rotation matrix is defined as a sequence of
# rotations about the axes of the original coordinate
# system where the axes are fixed in space.
# Basic rotation matrices are defined as:
Rx = lambda a: np.array([[1, 0, 0],
                        [0, np.cos(a), np.sin(a)],
                        [0, -np.sin(a), np.cos(a)]])

Ry = lambda b: np.array([[np.cos(b), 0, -np.sin(b)],
                        [0, 1, 0],
                        [np.sin(b), 0, np.cos(b)]])

Rz = lambda c: np.array([[np.cos(c), np.sin(c), 0],
                        [-np.sin(c), np.cos(c), 0],
                        [0, 0, 1]])

# Skew symmetric matrix of a vector v
SS = lambda v: np.array([[0, -v[2], v[1]],
                                    [v[2], 0, -v[0]],
                                    [-v[1], v[0], 0]])


# Basis vector ei
ei = lambda i: np.array([1 if j == i else 0 for j in range(3)])

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

# Feel free to change these values that define the rotation matrix
roll_deg = 15.0
pitch_deg = 10.0
yaw_deg = -5.0
r = np.array([1.0, 2.0, 3.0])

# Small value for numerical differentiation
drot = np.array([0.005, 0.005, 0.005])

roll = roll_deg * np.pi/180
pitch = pitch_deg * np.pi/180
yaw = yaw_deg * np.pi/180
C = ExtEulerRot(roll, pitch, yaw)
print("Rotation matrix C: \n", C)

# Now define the derivative of the rotation matrix with respect to the Euler angles
# Numerical differentiation
# Verify that the derivative is correct using numerical differentiation
delta_Cr_num = delta_Cr(roll, pitch, yaw, r, drot)
print("Derivative of rotation matrix C with respect to roll, pitch, and yaw using numerical differentiation: \n", delta_Cr_num)

# First using method outlined by James R. Lucas in "Differentiation of the Orientation Matrix by Matrix Multipliers"
# https://www.asprs.org/wp-content/uploads/pers/1963journal/jul/1963_jul_708-715.pdf
Qj = (Rx(roll).T) @ (-SS(ei(1))) @ Rx(roll)
dC_dyaw = -SS(ei(2)) @ C
dC_dpitch = C @ Qj
dC_droll = C @ (-SS(ei(0)))

dCr_droll = (dC_droll @ r)[:,np.newaxis]
dCr_dpitch = (dC_dpitch @ r)[:,np.newaxis]
dCr_dyaw = (dC_dyaw @ r)[:,np.newaxis]

dCr_lucas = np.hstack([dCr_droll, dCr_dpitch, dCr_dyaw])
print("Derivative of rotation matrix C with respect to roll, pitch, and yaw using Lucas method: \n", dCr_lucas)

# Now using the method implied by the math in Fankhauser's Robot-centric elevation mapping with uncertainty estimates
# From equation 2 without the projection P
dCr_fankhauser = C @ SS(r)
print("Derivative of rotation matrix C with respect to roll, pitch, and yaw using Fankhauser method: \n", dCr_fankhauser)

# Are they the same?
# print("Difference between the two methods: \n", dCr_lucas - dCr_fankhauser)

# Now use the method described by Bloesch et al. in "A Primer on the Differential Calculus of 3D Orientations"
# From equation 27
dC_r_bloesch = -SS(C.dot(r))
print("Derivative of rotation matrix C with respect to r using Bloesch method: \n", dC_r_bloesch)
