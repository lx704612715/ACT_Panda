import numpy as np
from pytransform3d.rotations import active_matrix_from_extrinsic_euler_xyz
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from tqdm import tqdm
from pytransform3d.camera import plot_camera

from contact_lfd.LfDusingEC.utils.utils_simple_lfd import generate_uniform_distributed_demo_poses
from contact_lfd.LfDusingEC.utils.utils_robotMath import RtToTrans
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller

arm_ctrl = Arm_Controller(gripper=False)

center_point = np.array([0.706236, 0.0554017, 0])
init_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
init_base_ht_ee = RtToTrans(R=init_R, t=center_point)

poses_1 = generate_uniform_distributed_demo_poses(num_poses=5, radius=0.10, height=0.25)
poses_2 = generate_uniform_distributed_demo_poses(num_poses=10, radius=0.14, height=0.3)
poses_3 = generate_uniform_distributed_demo_poses(num_poses=15, radius=0.18, height=0.35)

poses = init_base_ht_ee @ np.vstack([poses_1, poses_2, poses_3])

for pose in tqdm(poses):
    arm_ctrl.move_to_carte_position(pose, completion_time=5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define camera intrinsic parameters with larger sensor size and focal length
focal_length = 0.004  # Increased to 40mm
sensor_width = 0.0072  # Doubled to 72mm
sensor_height = 0.0048  # Doubled to 48mm
M = np.array([[focal_length, 0, sensor_width / 2],
              [0, focal_length, sensor_height / 2],
              [0, 0, 1]])

# Set plot limits
ax.set_xlim(0.0, 1)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.0, 0.5)

# Plot the center of the circle as the object
ax.plot(center_point[0], center_point[1], center_point[2], 'ro')

for pose in poses:
    # Plot each pose as a camera
    # plot_transform(ax, pose, s=0.05)
    plot_camera(ax, M=M, cam2world=pose, virtual_image_distance=0.03, sensor_size=(sensor_width, sensor_height))

plt.show()
print("test")